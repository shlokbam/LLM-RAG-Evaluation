from sentence_transformers import CrossEncoder
from src.retriever import load_store
from langsmith import traceable

# Small, fast, Mac M2 CPU-friendly reranker (~90MB).
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k: int = 25, top_k: int = 7, model_name: str = CROSS_ENCODER):
        self.store = load_store()
        self.reranker = CrossEncoder(model_name)
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query: str):
        # 1. OVER-RETRIEVE: fast bi-encoder, fetch candidate chunks
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


def build_reranking_retriever(fetch_k: int = 25, top_k: int = 7):
    """Factory function to build a RerankingRetriever."""
    return RerankingRetriever(fetch_k=fetch_k, top_k=top_k)



if __name__ == "__main__":
    retriever = build_reranking_retriever()
    results = retriever.invoke("what is regression testing?")
    print(f"Retrieved {len(results)} reranked results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Session {r.metadata.get('session', 'N/A')}] {r.page_content[:150]}...\n")