import os
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

# ---- Load notes as chunks ----


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
                                "id": f"{filename}:{line_no}",
                                "text": line,
                                "source": filename,
                                "line": line_no,
                            }
                        )

    return chunks


# ---- Setup Chroma ----

chroma_client = chromadb.Client()
# for persistant storage, uncomment the following and comment out the above line.
# Note: you might need to have chromadb installed with the "duckdb+parquet" extra for this to work.
# chroma_client = chromadb.Client(
#     Settings(
#         persist_directory="./chroma_db"
#     )
# )

collection = chroma_client.get_or_create_collection(name="notes")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Indexing (only if empty) ----


def index_if_needed():
    if collection.count() > 0:
        print("Chroma already indexed.")
        return

    print("Indexing into Chroma...")

    chunks = load_note_chunks()

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "line": c["line"]} for c in chunks]

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


# ---- LLM ----


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


def retrieve(query: str, top_k: int = 2):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    output = []

    for i in range(len(results["documents"][0])):
        output.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "line": results["metadatas"][0][i]["line"],
            }
        )

    return output


# ---- Answer ----


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in your notes."

    context = "\n".join(
        f"- [{item['source']}:{item['line']}] {item['text']}" for item in results
    )

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


# ---- Main ----

if __name__ == "__main__":
    index_if_needed()

    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        results = retrieve(q)

        print("\n" + generate_answer(q, results))
        print("Supporting notes:")

        for item in results:
            print(f"- [{item['source']}:{item['line']}] {item['text']}")
