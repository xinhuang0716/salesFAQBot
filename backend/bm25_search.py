import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
from typing import List, Dict

class BM25Searcher:
    """
    BM25 searcher using CKIP word segmentation.
    """

    def __init__(self, texts: List[str], payloads: List[Dict], ws_model: str = "bert-base", device: int = -1):
        """
        Args:
            texts (List[str]):
                Corpus documents. Index i 對應 payloads[i]。
            payloads (List[Dict]):
                Metadata list, include topic / subtype / relevance / index。
            ws_model (str):
                CKIP word segmenter model name (e.g., "bert-base").
            device (int):
                -1: CPU, 0/1/...: GPU id
        """
        self.texts = texts
        self.payloads = payloads
        self.ws = CkipWordSegmenter(model=ws_model, device=device)

        self.corpus_tokens: List[List[str]] = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"[BM25 Searcher] Built BM25 index: {len(texts)} docs")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single input string using the CKIP word segmenter and return a list of tokens.

        Args:
            text(str)

        """
        return self.ws([text])[0]

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)  # shape: (N,)

        top_idx = np.argsort(scores)[::-1][:k]

        results: List[Dict] = []
        for rank, i in enumerate(top_idx, start=1):
            payload = self.payloads[i]
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(i),
                    "score": float(scores[i]),
                    "topic": payload.get("topic"),
                    "subtype": payload.get("subtype"),
                    "relevance": payload.get("relevance"),
                }
            )
        return results

    def raw_scores(self, query: str) -> np.ndarray:
        """
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        return np.asarray(scores, dtype=np.float32)





