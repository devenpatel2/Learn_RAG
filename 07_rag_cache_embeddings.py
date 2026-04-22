import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LLAMA_URL = "http://192.168.178.20:11434/api/generate"
INDEX_DIR = "index"
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")

PROMPT = """
You are answering questions about a small personal notes database.

Rules:
- Use only facts explicitly stated in the context.
- Do not add outside knowledge.
- If the answer is not clearly in the context, say exactly: I don't know based on the notes.
- Keep the answer short and direct.

Context:
{context}

Question:
{query}

Answer:
"""


def load_note_chunks(folder="notes"):
    chunks = []

    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:
                        chunks.append(
                            {
                                "text": line,
                                "source": filename,
                                "line": line_no,
                            }
                        )

    return chunks


def build_index(model, folder="notes"):
    chunks = load_note_chunks(folder)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    np.save(EMBEDDINGS_FILE, embeddings)

    return chunks, embeddings


def load_index():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings


def index_exists():
    return os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE)


def llama_req(prompt: str):
    response = requests.post(
        LLAMA_URL,
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    if response.status_code == 200:
        return response.json()["response"].strip()

    print(f"Error: {response.status_code} - {response.text}")
    return "Sorry, I couldn't generate a response."


def retrieve(
    query: str, model, chunks, chunk_embeddings, top_k: int = 2, min_score: float = 0.20
):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        if scores[i] >= min_score:
            results.append(
                {
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    "line": chunks[i]["line"],
                    "score": float(scores[i]),
                }
            )

    return results


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in your notes."

    context = "\n".join(
        f"- [{item['source']}:{item['line']}] {item['text']}" for item in results
    )

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if index_exists():
        print("Loading existing index...")
        chunks, chunk_embeddings = load_index()
    else:
        print("Building index from notes...")
        chunks, chunk_embeddings = build_index(model)

    print("Loaded chunks:")
    for chunk in chunks:
        print(f"- [{chunk['source']}:{chunk['line']}] {chunk['text']}")

    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = retrieve(q, model, chunks, chunk_embeddings)

        print("\n" + generate_answer(q, results))
        print("Supporting notes:")
        if results:
            for item in results:
                print(
                    f"- ({item['score']:.3f}) [{item['source']}:{item['line']}] {item['text']}"
                )
        else:
            print("- No strong match found.")
