from dense_search import DenseSearcher
from bm25_search import BM25Searcher
from typing import List, Dict, Optional
import math


class HybridRRFSearcher:
    """
    Hybrid searcher using Reciprocal Rank Fusion (RRF) to combine
    dense (semantic) rankings and BM25 (lexical) rankings.
    """

    def __init__(self, dense_searcher: DenseSearcher, bm25_searcher: BM25Searcher, k_param: int = 60):
        """
        Args:
            dense_searcher (DenseSearcher):
                Dense semantic searcher, typically backed by Qdrant + BGE.
            bm25_searcher (BM25Searcher):
                BM25 searcher, typically using CKIP tokenization.
            k_param (int):
                RRF hyper-parameter k in the formula:
                    score(d) = sum_s 1 / (k + rank_s(d))
                Larger k reduces the impact of tail ranks. Default to 60.
        """
        self.dense_searcher = dense_searcher
        self.bm25_searcher = bm25_searcher
        self.k_param = k_param

    def search(self, query: str, k: int = 5, dense_k: int = 50, bm25_k: int = 50) -> List[Dict]:
        """
        Perform hybrid search using RRF over the dense and BM25 rankings.

        Args:
            query (str):
                User query in natural language.
            k (int):
                Number of final top results to return.
            dense_k (int):
                Number of top documents to retrieve from the dense searcher
                before fusion (candidate size).
            bm25_k (int):
                Number of top documents to retrieve from the BM25 searcher
                before fusion.

        Returns:
            List[dict]:
                A list of fused result dictionaries, where each entry contains:
                    - 'rank'        : Final hybrid rank (starting from 1).
                    - 'doc_id'      : Document index in the corpus / Qdrant.
                    - 'rrf_score'   : RRF fused score.
                    - 'dense_rank'  : Rank in dense ranking (or None if not present).
                    - 'bm25_rank'   : Rank in BM25 ranking (or None if not present).
                    - 'dense_score' : Raw dense score if available, else 0.0.
                    - 'bm25_score'  : Raw BM25 score if available, else 0.0.
                    - 'topic' / 'subtype' / 'relevance': Metadata fields from payload.
        """
        k_param = self.k_param

        dense_results = self.dense_searcher.search(query, k=dense_k)
        bm25_results = self.bm25_searcher.search(query, k=bm25_k)

        dense_rank: Dict[int, int] = {}
        dense_score: Dict[int, float] = {}
        for r in dense_results:
            doc_id = int(r["doc_id"])
            dense_rank[doc_id] = int(r["rank"])
            dense_score[doc_id] = float(r["score"])

        bm25_rank: Dict[int, int] = {}
        bm25_score: Dict[int, float] = {}
        for r in bm25_results:
            doc_id = int(r["doc_id"])
            bm25_rank[doc_id] = int(r["rank"])
            bm25_score[doc_id] = float(r["score"])

        candidate_ids = sorted(set(dense_rank.keys()) | set(bm25_rank.keys()))
        if not candidate_ids:
            return []

        rrf_scores: Dict[int, float] = {}

        for doc_id in candidate_ids:
            score = 0

            if doc_id in dense_rank:
                score += 1 / (k_param + dense_rank[doc_id])

            if doc_id in bm25_rank:
                score += 1 / (k_param + bm25_rank[doc_id])

            rrf_scores[doc_id] = score

        sorted_ids = sorted(
            candidate_ids,
            key=lambda d: rrf_scores[d],
            reverse=True,
        )
        top_ids = sorted_ids[:k]

        results: List[Dict] = []
        for rank, doc_id in enumerate(top_ids, start=1):
            payload = self.bm25_searcher.payloads[doc_id]

            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "rrf_score": float(rrf_scores[doc_id]),
                    "dense_rank": dense_rank.get(doc_id),
                    "bm25_rank": bm25_rank.get(doc_id),
                    "dense_score": dense_score.get(doc_id, 0.0),
                    "bm25_score": bm25_score.get(doc_id, 0.0),
                    "topic": payload.get("topic"),
                    "subtype": payload.get("subtype"),
                    "relevance": payload.get("relevance"),
                }
            )

        return results
