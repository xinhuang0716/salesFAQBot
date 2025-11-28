from qdrant_client import QdrantClient
from embedder.base_embedder import BaseEmbedder
from embedder.bge_embedder import BGEEmbedder

from reranker.base_reranker import BaseReranker
from reranker.bge_reranker import BGEReranker
from embedder.ckip_tokenizer import CKIPTokenizer
from retrieve.bm25_search import BM25Searcher
from .. import build_corpus_payload
from typing import Literal, Optional, Tuple


SearchType = Literal["dense", "bm25", "hybrid"]
EmbedderType = Literal["bge-m3", "bge-small-zh", "gemma-300m"]
RerankerType = Literal["bge-reranker", "jina-reranker"]


def create_embedder(embedder_type: EmbedderType, device: str = "cpu") -> BaseEmbedder:
    """
    Create an embedder instance based on the given backend type.
    
    Args:
        embedder_type:
            Specifies which embedding backend to use.
        device:
            Device to run the model on, either "cpu" or "cuda".
    
    Returns:
        BaseEmbedder:
            An object implementing the BaseEmbedder interface.
    """
    if embedder_type == "bge-m3":
        return BGEEmbedder(
            model_name="BAAI/bge-m3",
            device=device,
            local_dir="./models/bge-m3",  
        )
    # elif embedder_type == "gemma-300m":
    #     return GemmaEmbedder(
    #         model_name="google/embeddinggemma-300m",
    #         device=device,
    #         local_dir="./models/embeddinggemma-300m",
    #     )
    else:
        raise ValueError(f"Unsupported embedder_type: {embedder_type}")


def create_reranker(use_reranker: bool, reranker_type: RerankerType, device: str = "cpu") -> BaseReranker:
    """
    Create a reranker instance if requested.
    
    Args:
        use_reranker:
            If True, return a BGEReranker instance; if False, return None.
        device:
            Device to run the reranker on, either "cpu" or "cuda".
    
    Returns:
        BaseReranker:
            An object implementing the Basereranker interface.
    """
    if not use_reranker:
        return None

    if use_reranker and reranker_type == 'bge-reranker':
        return BGEReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device=device,
    )


def load_components(search: SearchType = "dense", embedder_type: EmbedderType = "bge-small-zh",use_reranker: bool = False, reranker_type = None, collection_name: str = "FAQ", db_path: str = "./db", device: str = "cpu") -> Tuple[Optional[QdrantClient], Optional[BaseEmbedder], Optional[BaseReranker]]:
    """
    Load and initialize all heavy RAG-related components in a single place.
    
    Args:
        search:
            Retrieval mode:
            - "dense":  use Qdrant dense vector search only.
            - "bm25":   use a pure BM25 flow (no Qdrant).
        embedder_type:
            Specifies which embedding backend to use, e.g.:
            - "bge-m3"
        use_reranker:
            Whether to load a reranker model (e.g. "BAAI/bge-reranker-v2-m3").
        collection_name:
            Name of the Qdrant collection (e.g. "FAQ").
        db_path:
            Local path to the Qdrant database directory (default: "./db").
        device:
            Device to run the models on, either "cpu" or "cuda".
    
    Returns:
        Tuple[Optional[QdrantClient], Optional[BaseEmbedder], Optional[BaseReranker]]:
            - client:   A QdrantClient instance, or None if Qdrant is not needed
                        for the selected search mode.
            - embedder: A BaseEmbedder instance, or None.
            - reranker: A BaseReranker instance, or None.
    """
    client: Optional[QdrantClient] = None
    embedder: Optional[BaseEmbedder] = None

    if search == 'dense':
        client = QdrantClient(path=db_path)
        embedder = create_embedder(embedder_type, device=device)
        print(f"[RAG Init] Qdrant loaded from {db_path}, embedding model: {embedder_type}")
        reranker_model = create_reranker(use_reranker=use_reranker, reranker_type=reranker_type, device=device)
        print(f"[RAG Init] Reranker loaded: {reranker_type}")
        
        return client, embedder, reranker_model
    elif search == 'bm25':
        tokenizer = CKIPTokenizer(
            ws_model="bert-base-chinese",
            device=-1,
            model_dir="./models/ckip-ws",
        )
        texts, payloads = build_corpus_payload(excel_path)
        bm25_searcher = BM25Searcher(
            texts=texts,
            payloads=payloads,
            tokenizer=tokenizer,
        )
        return bm25_searcher
