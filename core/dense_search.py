from qdrant_client import QdrantClient

from core.embedder import Embedder


class DenseSearcher:
    """Retrieve relevant knowledge documents using dense vectors."""

    def __init__(self, client: QdrantClient, embedder: Embedder, collection_name: str = "FAQ") -> None:
        """Initialize the DenseSearcher with a Qdrant client and an embedder.

        Args:
            client (QdrantClient): The Qdrant client for interacting with the vector database.
            embedder (Embedder): The embedder for generating dense vector representations of queries.
            collection_name (str): The name of the Qdrant collection to search in. Defaults to "FAQ".

        """
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

    def search(self, query: str, top_k: int, score_threshold: float | None) -> list[dict]:
        """Return the highest-scoring documents for a user query.

        Args:
            query (str): The user query to search for.
            top_k (int): The maximum number of results to return.
            score_threshold (float | None): The minimum score threshold, or None to use top-K only.

        Returns:
            list[dict]: A list of dictionaries containing the rank, score, and payload of the retrieved documents.

        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.encode_query(query),
            with_payload=True,
            limit=top_k,
            score_threshold=score_threshold,
        )

        results = []
        for rank, point in enumerate(response.points, start=1):
            results.append({"point_id": point.id, "dense_rank": rank, "dense_score": float(point.score), **(point.payload or {})})

        return results
