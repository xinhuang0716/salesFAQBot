import numpy as np
from embedder.bge_embedder import BGEEmbedder
from corpus_builder import build_corpus_payload
from database_builder import init_Qdrant


def build_index(excel_path: str = "知識文件蒐集.xlsx", collection_name: str = "FAQ", db_path: str = "./db", model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
    """
    Build the Qdrant vector index from the given file.

    Args:
        excel_path (str):
            Path to the Excel knowledge file.
        collection_name (str):
            Qdrant collection name to store vectors.
        db_path (str):
            Local Qdrant db path.
        model_name (str):
            Embedding model name (for BGEEmbedder).
        device (str):
            "cpu" or "cuda" depending on your environment.
    """
    texts, payloads = build_corpus_payload(excel_path)

    embedder = BGEEmbedder(model_name=model_name, device=device)
    doc_embs: np.ndarray = embedder.encode_corpus(
        texts,
        max_length=512,
        batch_size=16,
    )
    print(f"[Index Builder] Computed embeddings: shape={doc_embs.shape}")

    init_Qdrant(
        collection_name=collection_name,
        vectors=doc_embs,
        payloads=payloads,
        db_path=db_path,
        recreate=False, 
    )

    print("[Index Builder] Done building index.")


if __name__ == "__main__":
    build_index()