import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTIONS = {
    "raw_500": "docs_raw_500",
    "smart_500": "docs_smart_500",
    "words_1000": "docs_words_1000",
    "smart_800": "docs_smart_800",
}


def retrieve(collection, model, query: str, top_k: int = 5):
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
            print_results(strategy_name, results)


if __name__ == "__main__":
    main()
