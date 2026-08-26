from pathlib import Path

from ckip_transformers.nlp import CkipWordSegmenter
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from core.context import format_document

BASE_DIR = Path(__file__).resolve().parents[1]


class BM25:
    """Build and query an in-memory BM25 index over Qdrant documents."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "FAQ",
        model_repo: str = "ckiplab/albert-tiny-chinese-ws",
    ) -> None:
        """Load CKIP and connect the BM25 index to Qdrant."""
        self.client = client
        self.collection_name = collection_name
        self.model_dir = BASE_DIR / "models" / model_repo.rsplit("/", maxsplit=1)[-1]

        if not (self.model_dir / "config.json").is_file():
            self.model_dir.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_repo, local_dir=str(self.model_dir))

        self.tokenizer = CkipWordSegmenter(model_name=str(self.model_dir), device=-1)
        self.model: BM25Okapi | None = None
        self.point_ids = []

    def tokenize(self, texts: str | list[str]) -> list[list[str]]:
        """Segment text and discard punctuation-only tokens."""
        texts = [texts] if isinstance(texts, str) else texts
        return self.tokenizer([text.casefold() for text in texts], show_progress=False)

    def fit(self) -> None:
        """Build BM25 from Qdrant payloads and retain point IDs."""
        documents = []
        self.point_ids = []

        # Retrieve all points from collection.
        total = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=total,
            with_payload=True,
            with_vectors=False,
        )

        # Retain pointIDs and format payloads.
        for point in points:
            self.point_ids.append(point.id)
            documents.append(format_document(point.payload))

        if not documents:
            raise ValueError("Cannot build a BM25 index without documents.")

        # Build the BM25 model.
        self.model = BM25Okapi(self.tokenize(documents))

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return Qdrant payloads with their BM25 rank and score."""
        if self.model is None:
            raise RuntimeError("BM25 must be fitted before searching.")

        scores = self.model.get_scores(self.tokenize(query)[0])
        indexes = scores.argsort()[::-1][:top_k]
        point_ids = [self.point_ids[int(index)] for index in indexes]
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        payloads = {point.id: point.payload or {} for point in points}

        return [
            {"point_id": point_id, "bm25_rank": rank, "bm25_score": float(scores[index]), **payloads[point_id]}
            for rank, (index, point_id) in enumerate(zip(indexes, point_ids, strict=True), start=1)
        ]
