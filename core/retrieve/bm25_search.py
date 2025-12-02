from core.embedder.bm25 import BM25


class BM25Searcher:
    """
    BM25 searcher for top-K retrieval.
    """
    
    def __init__(self, bm25: BM25, payloads: list[dict]):
        """
        Initialize searcher with fitted BM25 model.
        
        Args:
            bm25 (BM25): Fitted BM25 instance.
            payloads (list[dict]): Metadata for each document (topic, subtype, relevance, etc.).
        """
        self.bm25 = bm25
        self.payloads = payloads
        if self.bm25 is None: raise ValueError("BM25 instance cannot be None, please fit the model first.")

    def search(self, query: str, k: int = 5) -> tuple[list[dict], list[float]]:
        """
        Search top-K documents for query.
        
        Args:
            query (str): Search query.
            k (int): Number of top results to return.
            
        Returns:
            tuple[list[dict], list[float]]: Top-K results with metadata and all scores.
        """
        query_tokens = self.bm25.tokenize(query)[0]
        scores = self.bm25.bm25_model.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:min(k, len(scores))]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            payload = self.payloads[idx]
            results.append({
                "rank": rank,
                "doc_id": int(idx),
                "score": float(scores[idx]),
                "topic": payload.get("topic"),
                "subtype": payload.get("subtype"),
                "relevance": payload.get("relevance"),
            })
        return results, scores





