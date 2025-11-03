# import necessary modules
import os, pandas as pd
from qdrant_client import QdrantClient, models


# initialize Qdrant client
def initQdrant(model_name:str, collectionName: str = "FAQ") -> QdrantClient | None:

    # initialize Qdrant client
    os.makedirs("./db", exist_ok = True)
    client = QdrantClient(path="./db")

    if client.collection_exists(collectionName):
        print("Qdrant collection already exists.")
        return client

    client.create_collection(
        collection_name=collectionName,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(model_name), distance=models.Distance.COSINE
        ),
        timeout=30
    )

    # import knowledge docs
    file = os.listdir("./knowledgeDoc")[-1]
    df = pd.read_excel(
        "./knowledgeDoc/" + file,
        usecols=["id", "source", "topic", "subtype", "relevance"],
    )

    client.upload_collection(
        collection_name=collectionName,
        vectors=[
            models.Document(text=doc, model=model_name)
            for doc in df["relevance"].values.tolist()
        ],
        ids=df["id"].values.tolist(),
        payload=df.to_dict(orient="records"),
    )

    print("Qdrant initialized successfully.")
    return client