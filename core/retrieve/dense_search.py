from qdrant_client import QdrantClient
from core.embedder.sentence_transformer_embedder import STEmbedder


class DenseSearcher:
    """
    Dense semantic searcher based on Qdrant + BGE embeddings.
    """

    def __init__(self, client: QdrantClient, embedder: STEmbedder):
        """
        Args:
            client (QdrantClient):
                Connected Qdrant client instance.
            embedder (STEmbedder):
                The same embedding model used during offline indexing.
        """
        self.client = client
        self.embedder = embedder

    def search(self, query: str, k: int = 3, score: float = 0.4) -> list[dict]:
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
                Number of top documents to retrieve.
            score (float):
                Minimum score threshold for retrieved documents.

        Returns:
            Search result object returned by QdrantClient.query_points.
        """

        q_vec = self.embedder.encode(query, encode_type="query")

        res = self.client.query_points(
            collection_name="FAQ",
            query=q_vec,
            with_payload=["id", "topic", "subtype", "relevance"],
            limit=k,
            score_threshold=score,
        )

        results: list[dict] = []
        for rank, p in enumerate(res.points, start=1):
            payload = p.payload or {}
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(payload.get("id", 0)),
                    "score": float(p.score),
                    "topic": payload.get("topic"),
                    "subtype": payload.get("subtype"),
                    "relevance": payload.get("relevance"),
                }
            )

        return results
