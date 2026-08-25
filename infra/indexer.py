from pathlib import Path

import pandas as pd

from core.embedder import Embedder

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledgeDoc"
REQUIRED_COLUMNS = ["id", "source", "topic", "subtype", "relevance"]


def build_index_data(embedder: Embedder) -> tuple[list[list[float]], list[dict]]:
    """Read the most recently modified workbook and return vectors and payloads."""
    files = list(KNOWLEDGE_DIR.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError("No knowledge workbook was found.")

    workbook_path = max(files, key=lambda path: path.stat().st_mtime)
    dataframe = pd.read_excel(workbook_path, usecols=REQUIRED_COLUMNS).dropna(how="all")

    payloads = dataframe.fillna("").to_dict(orient="records")
    vectors = embedder.encode_documents(
        [
            f"[主題]{payload['topic']}\n[子題]{payload['subtype']}\n[內容]{payload['relevance']}"
            for payload in payloads
        ]
    )

    return vectors, payloads
