import os
import numpy as np
from qdrant_client import QdrantClient, models
from typing import Dict, List

def init_Qdrant(collection_name: str, vectors: np.ndarray, payloads: List[Dict], db_path: str = "./db", distance: models.Distance = models.Distance.COSINE, recreate: bool = False) -> QdrantClient:
    """
    Initialize a Qdrant collection and upsert vectors + payloads.

    Args:
        collection_name (str):
            Target collection name in Qdrant.
        vectors (np.ndarray):
            2D array of shape (N, D), where N is the number of points
            and D is the embedding dimension.
        payloads (List[dict]):
            List of payload dictionaries, one per vector.
            The length must be equal to vectors.shape[0].
        db_path (str):
            Local path for embedded Qdrant storage.
        distance (models.Distance):
            Distance metric to use (e.g., COSINE).
        recreate (bool):
            If True:
                - Drop the existing collection (if any) and recreate it.
            If False:
                - Create the collection only when it does not exist,
                  then upsert the vectors into it.

    Returns:
        QdrantClient: The Qdrant client instance connected to the given db_path.
    """
    os.makedirs(db_path, exist_ok=True)
    client = QdrantClient(path=db_path)

    num_points, dim = vectors.shape


    if recreate:
        # Drop and recreate the collection: suitable when you rebuild, the entire index (e.g., re-embed the whole file).
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=distance,
            ),
        )
    else:
        # Only create when the collection does not exist, suitable for incremental upsert.
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=distance,
                ),
            )

    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors.tolist(),               
        ids=list(range(num_points)),            
        payload=payloads,
    )
    print(f"[Init Qdrant] collection='{collection_name}', "f"points={num_points}, dim={dim}, recreate={recreate}")
    return client
