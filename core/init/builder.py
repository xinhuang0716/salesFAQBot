import os, pandas as pd
from qdrant_client import QdrantClient
from core.init.database import init_Qdrant
from core.embedder.sentence_transformer_embedder import STEmbedder
from core.embedder.aoai_embedder import AOAIEmbedder
from core.embedder.base_embedder import BaseEmbedder
from utils.corpus import build_text


def initialize(embedder_config: dict = None, collection_name: str = "FAQ", repo: str = "BAAI/bge-m3") -> tuple[QdrantClient, BaseEmbedder]:
    """Initialize embedder and vector database based on configuration.

    Args:
        embedder_config (dict, optional): Embedder configuration from config.yaml. If None, uses repo parameter.
        collection_name (str, optional): Target collection name in Qdrant. Defaults to "FAQ".
        repo (str, optional): HuggingFace model repository ID (legacy parameter). Defaults to "BAAI/bge-m3".

    Returns:
        tuple[QdrantClient, BaseEmbedder]: Qdrant client and initialized embedder
    """

    # Load data
    file: str = os.listdir("./knowledgeDoc")[-1]
    payloads: list[dict] = pd.read_excel(
        "./knowledgeDoc/" + file, 
        usecols=["id", "source", "topic", "subtype", "relevance"]
    ).to_dict(orient="records")

    # Initialize embedder based on configuration
    if embedder_config is None:
        # Legacy mode: use repo parameter
        embedder: BaseEmbedder = STEmbedder(repo)
    else:
        embedder_type: str = embedder_config.get("type", "sentence_transformer")
        
        if embedder_type == "sentence_transformer":
            embedder: BaseEmbedder = STEmbedder(embedder_config["repo"])
        elif embedder_type == "aoai":
            aoai_config = embedder_config.get("aoai", {})
            embedder: BaseEmbedder = AOAIEmbedder(
                model=aoai_config.get("model", "text-embedding-3-large"),
                dimensions=aoai_config.get("dimensions", 1024)
            )
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")

    # Generate embeddings
    vectors: list[list[float]] = embedder.encode(
        texts=[build_text(row) for row in payloads], 
        encode_type="document"
    )

    # Initialize vector database
    client: QdrantClient = init_Qdrant(
        collection_name=collection_name,
        vectors=vectors,
        payloads=payloads
    )

    return client, embedder
