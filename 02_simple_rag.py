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

    # For now, this is still a template-based “generator”
    answer = f"""Question: {query}

Context:
{context}

Answer:
Based on your notes, the most relevant information is:
{results[0][0]}
"""
    return answer


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
