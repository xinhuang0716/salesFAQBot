import os, yaml, pandas as pd
from core.init.database import init_Qdrant
from core.embedder.sentence_transformer_embedder import STEmbedder
from utils.corpus import build_text


def initialize():
    """
    Initialize Qdrant client and embedder based on configuration.

    Raises:
        ValueError: The embedder type is not supported, must be 'sentence_transformer' or 'API'.

    Returns:
        Tuple[QdrantClient, BaseEmbedder]: client: Connected Qdrant client instance. embedder: Initialized embedder instance.
    """

    # configuration
    with open("./config/config.yaml", "r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)

    # embeddors
    if config["embedder"]["type"] == "sentence_transformer":
        embedder: STEmbedder = STEmbedder(repo=config["embedder"]["model_repo"])

    elif config["embedder"]["type"] == "API":
        # TODO: implement API embedder
        pass

    else:
        raise ValueError("Unsupported embedder type. Supported types are: 'sentence_transformer', 'API'.")

    # input data for vector DB
    file: str = os.listdir("./knowledgeDoc")[-1]
    payloads: list[dict] = pd.read_excel("./knowledgeDoc/" + file, usecols=["id", "source", "topic", "subtype", "relevance"]).to_dict(orient="records")
    vectors: list[list[float]] = embedder.encode(texts=[build_text(row) for row in payloads], encode_type="document")

    # vector database
    client: QdrantClient = init_Qdrant(
        collection_name=config["db"]["collection_name"],
        vectors=vectors,
        payloads=payloads
    )

    return client, embedder
