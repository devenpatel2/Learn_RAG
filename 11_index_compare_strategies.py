import os
import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_FOLDER = "test_docs"
TEXT_FOLDER = "test_docs_text"
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

STRATEGIES = {
    "raw_500": {
        "collection": "docs_raw_500",
        "type": "raw",
        "chunk_size": 500,
        "overlap": 100,
    },
    "words_1000": {
        "collection": "docs_words_1000",
        "type": "words",
        "chunk_size": 1000,
        "overlap": 150,
    },
    "smart_500": {
        "collection": "docs_smart_500",
        "type": "smart",
        "chunk_size": 500,
        "overlap": 100,
    },
    "smart_800": {
        "collection": "docs_smart_800",
        "type": "smart",
        "chunk_size": 800,
        "overlap": 120,
    },
}


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_txt_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_raw(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_words(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    start = 0

    overlap_words = max(1, overlap // 6)

    while start < len(words):
        current = []
        current_len = 0
        i = start

        while i < len(words) and current_len + len(words[i]) + 1 <= chunk_size:
            current.append(words[i])
            current_len += len(words[i]) + 1
            i += 1

        chunk = " ".join(current).strip()
        if chunk and len(chunk) >= 200:
            chunks.append(chunk)

        start = max(i - overlap_words, start + 1)

    return chunks


def chunk_smart(text: str, chunk_size: int, overlap: int):
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        boundaries = [
            text.rfind(". ", start, end),
            text.rfind("; ", start, end),
            text.rfind(": ", start, end),
        ]
        boundary = max(boundaries)

        if boundary > start + int(chunk_size * 0.65):
            end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_text(text: str, strategy: dict):
    kind = strategy["type"]

    if kind == "raw":
        return chunk_raw(text, strategy["chunk_size"], strategy["overlap"])

    if kind == "words":
        return chunk_words(text, strategy["chunk_size"], strategy["overlap"])

    if kind == "smart":
        return chunk_smart(text, strategy["chunk_size"], strategy["overlap"])

    raise ValueError(f"Unknown strategy type: {kind}")


def load_text_chunks(strategy: dict):
    all_chunks = []

    for filename in sorted(os.listdir(TEXT_FOLDER)):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(TEXT_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = clean_txt_text(f.read())
        if not text:
            continue

        text_chunks = chunk_text(text, strategy)

        for chunk_id, chunk in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{filename}::c{chunk_id}",
                "text": chunk,
                "source": filename,
                "chunk": chunk_id,
                "strategy": strategy["type"],
            })

    return all_chunks


def load_pdf_chunks(strategy: dict):
    all_chunks = []

    for filename in sorted(os.listdir(PDF_FOLDER)):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(PDF_FOLDER, filename)
        reader = PdfReader(filepath)

        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_pdf_text(page.extract_text() or "")

            if not text:
                continue

            page_chunks = chunk_text(text, strategy)

            for chunk_id, chunk in enumerate(page_chunks):
                all_chunks.append({
                    "id": f"{filename}:p{page_num}:c{chunk_id}",
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                    "chunk": chunk_id,
                    "strategy": strategy["type"],
                })

    return all_chunks


def add_in_batches(collection, documents, embeddings, metadatas, ids, batch_size=500):
    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        collection.add(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

        print(f"  added {min(end, len(documents))} / {len(documents)}")


def rebuild_collection(chroma_client, model, strategy_name: str, strategy: dict):
    collection_name = strategy["collection"]

    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        chroma_client.delete_collection(collection_name)

    collection = chroma_client.get_or_create_collection(collection_name)

    # chunks = load_pdf_chunks(strategy)
    chunks = load_text_chunks(strategy)

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = []
    for chunk in chunks:
        meta_data = {
            "source": chunk["source"],
            "chunk": chunk["chunk"],
            "strategy": chunk["strategy"],
        }
        if "page" in chunk:
            meta_data.update({"page": chunk["page"]})

        metadatas.append(meta_data)

    print(f"\nIndexing {strategy_name} into {collection_name}")
    print(f"  chunks: {len(chunks)}")

    embeddings = model.encode(documents).tolist()

    add_in_batches(
        collection=collection,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def main():
    model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)

    for strategy_name, strategy in STRATEGIES.items():
        rebuild_collection(chroma_client, model, strategy_name, strategy)

    print("\nDone. Built collections:")
    for strategy_name, strategy in STRATEGIES.items():
        print(f"- {strategy_name}: {strategy['collection']}")


if __name__ == "__main__":
    main()
