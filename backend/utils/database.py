import os, pandas as pd
from qdrant_client import QdrantClient, models


def initQdrant(collection: str, vecList: list[list], payloadList: list[dict], dimension: float) -> QdrantClient:
    """Initialize Qdrant database and create collection if not exists.

    Args:
        collection (str): Name of the Qdrant collection.
        vecList (list[list]): List of embedding vectors.
        payloadList (list[dict]): List of payload metadata dictionaries, like {"id": ..., "source": ..., etc.}, one per vector.
        dimension (float): Dimension of the embedding vectors.
    Returns:
        QdrantClient: _description_
    """

    # check if dbPath exists, if not create it
    os.makedirs("./db", exist_ok=True)

    # check if collection exists
    client = QdrantClient(path="./db")
    if client.collection_exists(collection):
        print(f"Collection {collection} already exists. Skipping creation.")
        return client

    # create collection if not exists
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

    return client
