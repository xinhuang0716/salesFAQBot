from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]


class Embedder:
    """Encode documents and queries with SentenceTransformers model."""

    def __init__(self, model_repo: str = "BAAI/bge-m3") -> None:
        """Initialize the embedding model.

        Args:
            model_repo (str): The Hugging Face repository ID of the model to use.

        """
        self.model_repo = model_repo
        self.model_dir = BASE_DIR / "models" / model_repo

        if not (self.model_dir / "modules.json").is_file():
            self.model_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=self.model_repo, local_dir=str(self.model_dir))

        self.model = SentenceTransformer(str(self.model_dir), local_files_only=True)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode knowledge-base documents for indexing."""
        return self.model.encode_document(texts, normalize_embeddings=True).tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode one user query for vector search."""
        return self.model.encode_query(query, normalize_embeddings=True).tolist()
