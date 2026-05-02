import chromadb
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


chroma_client = chromadb.PersistentClient("./chroma_db")
collection = chroma_client.get_or_create_collection(name="test_documents")

model = SentenceTransformer("all-MiniLM-L6-v2")


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


# ---- Retrieval ----


def retrieve(query: str, top_k: int = 5, max_distance: float = 0.85):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    output = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        if distance > max_distance:
            continue

        output.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            }
        )

    return output


# ---- Answer ----


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in your notes."

    context = "\n".join(
        f"- [{item['source']}:{item['page']}] {item['text']}" for item in results
    )

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


# ---- Main ----

if __name__ == "__main__":

    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = retrieve(q)

        print("\n" + generate_answer(q, results))
        print("Supporting notes:")

        for item in results:
            print(f"- [{item['source']}:{item['page']}] distance={item['distance']:.4f} {item['text']}")
