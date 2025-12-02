import os, pandas as pd
from qdrant_client import QdrantClient
from core.init.database import init_Qdrant
from core.embedder.sentence_transformer_embedder import STEmbedder
from utils.corpus import build_text


def initialize(collection_name: str = "FAQ", repo: str = "BAAI/bge-m3") -> tuple[QdrantClient, STEmbedder]:
    """_summary_

    Args:
        collection_name (str, optional): Target collection name in Qdrant. Defaults to "FAQ".
        repo (str, optional): HuggingFace model repository ID. Defaults to "BAAI/bge-m3".

    Returns:
        tuple[QdrantClient, STEmbedder]: 
    """

    # embedding
    file: str = os.listdir("./knowledgeDoc")[-1]
    payloads: list[dict] = pd.read_excel("./knowledgeDoc/" + file, usecols=["id", "source", "topic", "subtype", "relevance"]).to_dict(orient="records")

    embedder: STEmbedder = STEmbedder(repo)
    vectors: list[list[float]] = embedder.encode(texts=[build_text(row) for row in payloads], encode_type="document")

    # vector database
    client: QdrantClient = init_Qdrant(
        collection_name=collection_name,
        vectors=vectors,
        payloads=payloads
    )

    return client, embedder
