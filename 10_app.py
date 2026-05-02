import chromadb
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

LLAMA_URL = "http://192.168.178.20:11434/api/generate"
MODEL_NAME = "llama3.1"

PROMPT = """
You are answering questions about regulatory documents.

Rules:
- Use only facts explicitly stated in the context.
- Do not add outside knowledge.
- If the answer is not clearly in the context, say exactly: I don't know based on the documents.
- Keep the answer short and direct.

Context:
{context}

Question:
{query}

Answer:
"""

app = FastAPI()

chroma_client = chromadb.PersistentClient("./chroma_db")
collection = chroma_client.get_or_create_collection(name="test_documents")
model = SentenceTransformer("all-MiniLM-L6-v2")


class AskRequest(BaseModel):
    question: str


def llama_req(prompt: str) -> str:
    response = requests.post(
        LLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def retrieve(query: str, top_k: int = 5, max_distance: float = 0.85):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    output = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]

        if distance <= max_distance:
            output.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "distance": distance,
            })

    return output


def generate_answer(query: str, results):
    if not results:
        return "I could not find a strong match in the documents."

    context = "\n".join(
        f"- [{item['source']}:{item['page']}] {item['text']}"
        for item in results
    )

    prompt = PROMPT.format(context=context, query=query)
    return llama_req(prompt)


@app.post("/ask")
def ask(req: AskRequest):
    results = retrieve(req.question)
    answer = generate_answer(req.question, results)

    return {
        "answer": answer,
        "sources": results,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mini RAG Assistant</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      max-width: 900px;
      margin: 40px auto;
      padding: 0 20px;
      line-height: 1.5;
    }
    textarea {
      width: 100%;
      height: 90px;
      font-size: 16px;
      padding: 10px;
    }
    button {
      margin-top: 10px;
      padding: 10px 16px;
      font-size: 16px;
      cursor: pointer;
    }
    .answer {
      margin-top: 30px;
      padding: 16px;
      background: #f5f5f5;
      border-radius: 8px;
      white-space: pre-wrap;
    }
    .source {
      margin-top: 12px;
      padding: 12px;
      border-left: 4px solid #ccc;
      background: #fafafa;
    }
    .meta {
      font-size: 13px;
      color: #555;
      margin-bottom: 6px;
    }
    details {
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <h1>Mini RAG Assistant</h1>

  <p>Ask a question about the indexed documents.</p>

  <textarea id="question" placeholder="Ask something..."></textarea>
  <br>
  <button onclick="ask()">Ask</button>

  <div id="result"></div>

  <script>
    async function ask() {
      const question = document.getElementById("question").value;
      const result = document.getElementById("result");

      result.innerHTML = "<p>Thinking...</p>";

      const response = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
      });

      const data = await response.json();

      let html = `
        <div class="answer"><strong>Answer:</strong><br>${escapeHtml(data.answer)}</div>
        <details open>
          <summary>Sources (${data.sources.length})</summary>
      `;

      for (const src of data.sources) {
        html += `
          <div class="source">
            <div class="meta">
              ${escapeHtml(src.source)} — page ${src.page}
              — distance ${src.distance.toFixed(4)}
            </div>
            <div>${escapeHtml(src.text)}</div>
          </div>
        `;
      }

      html += "</details>";
      result.innerHTML = html;
    }

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
