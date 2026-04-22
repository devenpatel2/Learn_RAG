import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LLAMA_URL = "http://192.168.178.20:11434/api/generate"

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

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:
                        chunks.append({
                            "text": line,
                            "source": filename,
                            "line": line_no,
                        })

    return chunks


chunks = load_note_chunks()

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(
    [chunk["text"] for chunk in chunks],
    normalize_embeddings=True
)


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


def retrieve(query: str, top_k: int = 2, min_score: float = 0.20):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        if scores[i] >= min_score:
            results.append({
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "line": chunks[i]["line"],
                "score": float(scores[i]),
            })

    return results


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in your notes."

    context = "\n".join(
        f"- {item['text']}" for item in results
    )

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


if __name__ == "__main__":
    print("Loaded chunks:")
    for chunk in chunks:
        print(f"- [{chunk['source']}:{chunk['line']}] {chunk['text']}")

    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = retrieve(q)

        print("\n" + generate_answer(q, results))
        print("Supporting notes:")

        if results:
            for item in results:
                print(
                    f"- ({item['score']:.3f}) "
                    f"[{item['source']}:{item['line']}] "
                    f"{item['text']}"
                )
        else:
            print("- No strong match found.")
