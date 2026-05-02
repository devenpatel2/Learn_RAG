import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LLAMA_URL = "http://192.168.178.20:11434/api/generate"

# PROMPT = """
# You are a helpful assistant.
#
# Use ONLY the information from the context below.
#
# Context:
# {context}
#
# Question:
# {query}
#
# Answer:
# """

PROMPT = """
You are answering questions about a small personal notes database.

Rules:
- Use only facts explicitly stated in the context.
- Do not add outside knowledge.
- If the answer is not clearly in the context, say: "I don't know based on the notes."
- Keep the answer short and direct.

Context:
{context}

Question:
{query}

Answer:
"""


def load_notes(folder="notes"):
    docs = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                content = f.read().strip()

                if content:
                    docs.append(content)

    return docs


# load_note_chunks is an improvement over simple load_notes as it breaks the "docs"
# into smaller chunks (e.g., by line) which can help with retrieval and relevance
def load_note_chunks(folder="notes"):
    docs = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(line)

    return docs


docs = load_note_chunks()
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs, normalize_embeddings=True)


def llama_req(prompt: str):
    response = requests.post(
        LLAMA_URL,
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
        },
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "Sorry, I couldn't generate a response."


def retrieve(query: str, top_k: int = 3, min_score: float = 0.20):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        if scores[i] >= min_score:
            results.append((docs[i], float(scores[i])))

    return results


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in your notes."

    context = "\n".join(f"- {doc}" for doc, _ in results)

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = retrieve(q)

        print("\n" + generate_answer(q, results))

        print("Supporting notes:")
        if results:
            for doc, score in results:
                print(f"- ({score:.3f}) {doc}")
        else:
            print("- No strong match found.")
