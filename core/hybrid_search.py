from core.bm25 import BM25
from core.dense_search import DenseSearcher


class HybridSearch:
    """Combine dense and BM25 results with reciprocal rank fusion."""

    def __init__(self, dense_searcher: DenseSearcher, bm25: BM25, rrf_k: int = 60) -> None:
        """Initialize the dense searcher, BM25 index, and RRF constant.

        Args:
            dense_searcher (DenseSearcher): The dense searcher instance.
            bm25 (BM25): The BM25 index instance.
            rrf_k (int): The RRF constant for reciprocal rank fusion.

        """
        self.dense_searcher = dense_searcher
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int, score_threshold: float | None, hybrid_top_k: int) -> list[dict]:
        """Return dense and BM25 results merged by point ID and ranked with RRF.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to retrieve from each searcher.
            score_threshold (float | None): The minimum score threshold for dense search results.
            hybrid_top_k (int): The number of top results to return after hybrid ranking.

        """
        dense_documents = self.dense_searcher.search(query, top_k, score_threshold)
        bm25_documents = self.bm25.search(query, top_k)

        documents = {}
        for document in dense_documents + bm25_documents:
            point_id = document["point_id"]
            documents.setdefault(point_id, {}).update(document)

        results = []
        for document in documents.values():
            dense_rank = document.get("dense_rank")
            bm25_rank = document.get("bm25_rank")
            hybrid_score = 0.0

            if dense_rank is not None:
                hybrid_score += 1 / (self.rrf_k + dense_rank)
            if bm25_rank is not None:
                hybrid_score += 1 / (self.rrf_k + bm25_rank)

            document["dense_rank"] = dense_rank
            document["dense_score"] = document.get("dense_score")
            document["bm25_rank"] = bm25_rank
            document["bm25_score"] = document.get("bm25_score")
            document["hybrid_score"] = hybrid_score
            results.append(document)

        results.sort(key=lambda document: document["hybrid_score"], reverse=True)

        for rank, document in enumerate(results[:hybrid_top_k], start=1):
            document["hybrid_rank"] = rank

        return results[:hybrid_top_k]
