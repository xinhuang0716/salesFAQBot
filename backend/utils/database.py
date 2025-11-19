import os, pandas as pd
from qdrant_client import QdrantClient, models


def initQdrant(collection: str, vecList: list[list], payloadList: list[dict], dimension: int) -> QdrantClient:
    """Initialize Qdrant database and create collection if not exists.

    Args:
        collection (str): Name of the Qdrant collection.
        vecList (list[list]): List of embedding vectors.
        payloadList (list[dict]): List of payload metadata dictionaries, like {"id": ..., "source": ..., etc.}, one per vector.
        dimension (int): Dimension of the embedding vectors.

    Returns:
        QdrantClient: Qdrant client instance.
    """

    # check if dbPath exists, if not create it
    os.makedirs("./db", exist_ok=True)

    # create collection if not exists
    client = QdrantClient(path="./db")
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        timeout=120,
    )

    # load knowledge documents into collection
    client.upload_collection(
        collection_name=collection,
        vectors=vecList,
        ids=list(range(len(vecList))),
        payload=payloadList,
    )

    print(f"Qdrant initialized with collection: {collection}, total points: {len(vecList)}")
    return client

def Qdrant(collection: str) -> None|QdrantClient:
    """Get Qdrant client instance for the specified collection.

    Args:
        collection (str): Name of the Qdrant collection.

    Returns:
        None|QdrantClient: Qdrant client instance if collection exists, else None.
    """

    client = QdrantClient(path="./db")

    if not client.collection_exists(collection):
        return None
    
    return client