from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = BASE_DIR / "db"


def initialize_database(collection_name: str, build_index_data: Callable[[], tuple[list[list[float]], list[dict]]]) -> QdrantClient:
    """Return a local Qdrant client with a populated collection.

    Args:
        collection_name (str): Name of the Qdrant collection to initialize.
        build_index_data (Callable): A function that returns a tuple of vectors and payloads.

    Returns:
        QdrantClient: A local Qdrant client with the specified collection.

    """
    DATABASE_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(DATABASE_PATH))

    # Return the client if exists
    if client.collection_exists(collection_name):
        return client

    # Build new collection if it doesn't exist
    vectors, payloads = build_index_data()

    if not vectors:
        raise ValueError("Cannot initialize a Qdrant collection without vectors.")

    if len(vectors) != len(payloads):
        raise ValueError("The number of vectors must match the number of payloads.")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
    )

    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=list(range(len(vectors))),
    )

    return client
