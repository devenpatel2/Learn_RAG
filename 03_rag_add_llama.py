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

docs = [
    "I need to call the insurance company on Monday.",
    "My Docker issue was caused by a missing environment variable.",
    "I want to learn AWS ECS but I have not started yet.",
    "I keep delaying bank paperwork.",
    "I need to buy an HDMI cable.",
]


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
