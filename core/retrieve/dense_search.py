from qdrant_client import QdrantClient
from embedder.bge_embedder import BGEEmbedder
from typing import List, Dict

class DenseSearcher:
    """
    Dense semantic searcher based on Qdrant + BGE embeddings.
    """

    def __init__(self, client: QdrantClient, embedder: BGEEmbedder, collection_name: str = "FAQ"):
        """
        Args:
            client (QdrantClient):
                Connected Qdrant client instance.
            embedder (BGEEmbedder):
                The same embedding model used during offline indexing.
            collection_name (str):
                Name of the Qdrant collection to search.
        """
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

    def search(self, query: str, k: int = 3) -> List[Dict]:

        """
        Perform a semantic search over the FAQ collection using Qdrant.

        Args:
            query (str):
                User's natural language query.
            client (QdrantClient):
                Connected Qdrant client.
            embedder (BGEEmbedder):
                The same embedding model used offline.
            collection_name (str):
                Target Qdrant collection name.
            k (int):
                Number of top documents to retrieve.

        Returns:
            Search result object returned by QdrantClient.query_points.
        """

        q_vec = self.embedder.encode_query(query, max_length=256)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec.tolist(),
            with_payload=["index", "topic", "subtype", "relevance"],
            limit=k,
        )


        results: List[Dict] = []
        for rank, p in enumerate(res.points, start=1):
            payload = p.payload or {}
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(payload.get("index")),
                    "score": float(p.score),
                    "topic": payload.get("topic"),
                    "subtype": payload.get("subtype"),
                    "relevance": payload.get("relevance"),
                }
            )
        return results
