from sentence_transformers import SentenceTransformer
import numpy as np

docs = [
    "I need to call the insurance company on Monday.",
    "My Docker issue was caused by a missing environment variable.",
    "I want to learn AWS ECS but I have not started yet.",
    "I keep delaying bank paperwork.",
    "I need to buy an HDMI cable.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs, normalize_embeddings=True)


def search(query: str, top_k: int = 3, min_score: float = 0.20):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"\nQuestion: {query}\n")
    print("Relevant notes:")

    found = False
    for i in top_indices:
        if scores[i] >= min_score:
            found = True
            print(f"- ({scores[i]:.3f}) {docs[i]}")

    if not found:
        print("- No strong match found.")


if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        search(q)
