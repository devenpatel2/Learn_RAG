import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_pdfs(folder: str):
    docs = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)

            reader = PdfReader(filepath)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            if text.strip():
                docs.append({
                    "source": filename,
                    "text": text
                })

    return docs


def chunk_text_para_break(text, chunk_size=1200, overlap=200):
    # Looks better, but doesnt quite work well
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)

            if len(para) <= chunk_size:
                current = para
            else:
                # fallback for very long paragraphs: split by words
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= chunk_size:
                        current += (" " if current else "") + word
                    else:
                        chunks.append(current)
                        current = word

    if current:
        chunks.append(current)

    # optional simple overlap by repeating tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                tail = chunks[i - 1][-overlap:]
                overlapped.append(tail + "\n\n" + chunk)
        return overlapped

    return chunks


def chunk_text_words(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        current = []
        current_len = 0
        i = start

        while i < len(words) and current_len + len(words[i]) + 1 <= chunk_size:
            current.append(words[i])
            current_len += len(words[i]) + 1
            i += 1

        chunks.append(" ".join(current))

        overlap_words = max(1, overlap // 6)
        start = max(i - overlap_words, start + 1)

    return chunks


def chunk_text_raw(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_pdf_chunks(folder="datasets/Documents"):
    chunks = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)

            reader = PdfReader(filepath)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if not text.strip():
                    continue

                text_chunks = chunk_text_raw(text)

                for chunk_id, chunk in enumerate(text_chunks):
                    chunks.append({
                        "id": f"{filename}:p{page_num}:c{chunk_id}",
                        "text": chunk,
                        "source": filename,
                        "page": page_num,
                        "chunk": chunk_id,
                    })

    return chunks


def add_in_batches(collection, documents, embeddings, metadatas, ids, batch_size=500):
    # fix for
    # ValueError: Batch size of 6359 is greater than max batch size of 5461
    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        collection.add(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

        print(f"Added {end if end < len(documents) else len(documents)} / {len(documents)}")


# ---- Setup Chroma ----
chroma_client = chromadb.PersistentClient("./chroma_db")


def index_chunks(model: SentenceTransformer):
    collection = chroma_client.get_or_create_collection(name="test_documents")
    # currrent stratety - delete old collection if exists
    if collection.count() > 0:
        chroma_client.delete_collection(name="test_documents")
        collection = chroma_client.get_or_create_collection(name="test_documents")

    print("Indexing into Chroma...")

    chunks = load_pdf_chunks()

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    embeddings = model.encode(documents).tolist()
    add_in_batches(
        collection=collection,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        batch_size=500,
    )
    # collection.add(
    #     documents=documents,
    #     embeddings=embeddings,
    #     metadatas=metadatas,
    #     ids=ids,
    # )


if __name__ == "__main__":
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    index_chunks(_model)
