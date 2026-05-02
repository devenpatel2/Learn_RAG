import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


CHROMA_PATH = "./chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
RE_RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTIONS = {
    "raw_500": "docs_raw_500",
    "smart_500": "docs_smart_500",
    "words_1000": "docs_words_1000",
    "smart_800": "docs_smart_800",
}


def rerank_sententce_tf(query: str, results: list, model):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    reranked = []
    for r in results:
        text_emb = model.encode([r["text"]], normalize_embeddings=True)[0]
        score = float(np.dot(query_emb, text_emb))
        r["rerank_score"] = score
        reranked.append(r)

    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


def rerank(query: str, results: list, reranker):
    if not results:
        return results

    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def retrieve(collection, model, query: str, top_k: int = 15):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    output = []

    for i in range(len(results["documents"][0])):
        result = {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk": results["metadatas"][0][i]["chunk"],
            "distance": results["distances"][0][i],
        }
        if "page" in results["metadatas"][0][i]:
            result.update({"page": results["metadatas"][0][i]["page"]})

        output.append(result)

    return output


def print_results(strategy_name: str, results):
    print("\n" + "=" * 80)
    print(strategy_name)
    print("=" * 80)

    for idx, item in enumerate(results, start=1):
        preview = item["text"].replace("\n", " ")
        preview = " ".join(preview.split())

        if len(preview) > 500:
            preview = preview[:500] + "..."

        if "page" in item:
            print(
                f"\n#{idx} "
                f"[{item['source']} p.{item['page']} c.{item['chunk']}] "
                f"distance={item['distance']:.4f}"
            )
        else:
            print(
                f"\n#{idx} "
                f"[{item['source']} c.{item['chunk']}] "
                f"distance={item['distance']:.4f}"
            )

        print(preview)


def main():
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RE_RANKER_MODEL)
    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        search_query = input("Search query override, blank to use same> ").strip()

        if search_query:
            query = search_query

        for strategy_name, collection_name in COLLECTIONS.items():
            collection = chroma_client.get_collection(collection_name)
            results = retrieve(collection, model, query, top_k=5)
            results = rerank(query, results, reranker)[:5]
            for r in results:
                print(r["distance"], "→", r["rerank_score"])
            print_results(strategy_name, results)


if __name__ == "__main__":
    main()
