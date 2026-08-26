import logging
from pathlib import Path

import pandas as pd

from core.context import format_document
from core.embedder import Embedder

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledgeDoc"
REQUIRED_COLUMNS = ["id", "source", "topic", "subtype", "relevance"]

logger = logging.getLogger(__name__)


def build_index_data(embedder: Embedder) -> tuple[list[list[float]], list[dict]]:
    """Read the latest workbook and return dense vectors and payloads."""
    files = list(KNOWLEDGE_DIR.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError("No knowledge workbook was found.")

    workbook_path = max(files, key=lambda path: path.stat().st_mtime)
    dataframe = pd.read_excel(workbook_path, usecols=REQUIRED_COLUMNS).dropna(how="all")

    payloads = dataframe.fillna("").to_dict(orient="records")
    documents = [format_document(payload) for payload in payloads]

    logger.info(
        "Loaded %s documents from %s for indexing.",
        len(documents),
        workbook_path.name,
    )
    vectors = embedder.encode_documents(documents)
    logger.info("Encoded %s dense vectors for indexing.", len(vectors))

    return vectors, payloads
