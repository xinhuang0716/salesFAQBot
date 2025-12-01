import os, shutil
from qdrant_client import QdrantClient, models


def init_Qdrant(collection_name: str, vectors: list[list[float]], payloads: list[dict]) -> QdrantClient:
    """
    Initialize a Qdrant collection and upsert vectors + payloads.

    Args:
        collection_name (str): Target collection name in Qdrant.
        vectors (list[list[float]]): 2D list of shape (N, D), where N is the number of points and D is the embedding dimension.
        payloads (list[dict]): List of payload dictionaries, one per vector, the length must be equal to vectors.shape[0].

    Returns:
        QdrantClient: The Qdrant client instance connected to the given DB path.
    """

    os.makedirs("./db", exist_ok=True)
    client = QdrantClient(path="./db")

    if client.collection_exists(collection_name):
        print(f"[Init Qdrant] collection='{collection_name}' already exists. Skipping creation.")
        return client

    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors.tolist(),
            ids=list(range(len(vectors))),
            payload=payloads,
        )

        print(f"[Init Qdrant] Initialized collection '{collection_name}', with {len(vectors)} points.")
        return client