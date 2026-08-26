from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

from core.context import format_document

BASE_DIR = Path(__file__).resolve().parents[1]


class Reranker:
    """Rerank retrieved documents with a locally cached BGE cross-encoder."""

    def __init__(self, model_repo: str = "BAAI/bge-reranker-base") -> None:
        """Load the reranker from disk, downloading it once when necessary.

        Args:
            model_repo (str, optional): The Hugging Face repository ID of the model to use

        """
        self.model_repo = model_repo
        self.model_dir = BASE_DIR / "models" / model_repo.rsplit("/", maxsplit=1)[-1]

        if not (self.model_dir / "config.json").is_file():
            self.model_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=self.model_repo, local_dir=str(self.model_dir))

        self.model = CrossEncoder(str(self.model_dir), local_files_only=True)

    def rerank(self, query: str, documents: list[dict], top_k: int, score_threshold: float | None = None) -> list[dict]:
        """Return the best matching documents in reranked order.

        Args:
            query (str): The user query to rerank against.
            documents (list[dict]): The retrieved documents to rerank.
            top_k (int): The maximum number of documents to return.
            score_threshold (float | None, optional): Minimum rerank score to include. Defaults to None.

        Returns:
            list[dict]: The reranked documents with updated rank, score and payload fields.

        """
        if not documents:
            return []

        # Reranking.
        paired_documents = [(query, format_document(document)) for document in documents]
        scores = self.model.predict(paired_documents, show_progress_bar=False)
        scored_documents = sorted(zip(documents, scores, strict=True), key=lambda item: float(item[1]), reverse=True)

        # Filter by score threshold if provided.
        if score_threshold is not None:
            scored_documents = [item for item in scored_documents if float(item[1]) >= score_threshold]

        # Limit to top_k results and update rank and score fields.
        reranked_documents = []
        for rank, (document, score) in enumerate(scored_documents[:top_k], start=1):
            reranked_document = dict(document)
            reranked_document["rerank_rank"] = rank
            reranked_document["rerank_score"] = float(score)
            reranked_documents.append(reranked_document)

        return reranked_documents
