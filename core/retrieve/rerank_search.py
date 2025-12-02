from qdrant_client import QdrantClient
from core.embedder.sentence_transformer_embedder import STEmbedder
from core.reranker.sentence_transformer_reranker import STEReranker


class RerankerSearcher:

    def __init__(self, client: QdrantClient, embedder: STEmbedder, reranker: STEReranker):
        """
        Args:
            client (QdrantClient):
                Connected Qdrant client instance.
            embedder (STEmbedder):
                The same embedding model used during offline indexing.
            reranker (STEReranker):
                The reranker model used to re-rank the retrieved documents.
        """
        self.client = client
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, k: int = 10, score: float = 0.4, top_k: int = 3) -> list[dict]:
        """
        Perform a semantic search over the FAQ collection using Qdrant.

        Args:
            query (str):
                User's natural language query.
            client (QdrantClient):
                Connected Qdrant client instance.
            embedder (STEmbedder):
                The same embedding model used offline.
            k (int):
                Number of top similar documents to retrieve.
            score (float):
                Minimum similarity score threshold for retrieved documents.
            top_k (int):
                Number of top documents to return after reranking.

        Returns:
            Search result object returned by QdrantClient.query_points.
        """

        query_vector = self.embedder.encode(query, encode_type="query")

        search_response = self.client.query_points(
            collection_name="FAQ",
            query=query_vector,
            with_payload=["id", "topic", "subtype", "relevance"],
            limit=k,
            score_threshold=score,
        )

        docs: list[dict] = []
        for rank, point in enumerate(search_response.points, start=1):
            payload = point.payload or {}
            docs.append({
                "rank": rank,
                "doc_id": int(payload.get("id", 0)),
                "score": float(point.score),
                "topic": payload.get("topic"),
                "subtype": payload.get("subtype"),
                "relevance": payload.get("relevance"),
            })

        if len(docs) == 0:
            return []
        else:
            rerank_idx = [i["corpus_id"]for i in self.reranker.rank(query, [doc["relevance"] for doc in docs], top_k) if i["score"] > 0.5]
            results = [docs[i] for i in rerank_idx]
            return results