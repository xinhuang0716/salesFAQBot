import pandas as pd
from typing import List, Dict, Tuple

def build_text(row: pd.Series) -> str:    
    """
    Build a normalized text representation from a single row
    in the knowledge Excel file.

    The resulting text will be fed into the embedding model.

    Args:
        row (pd.Series): A row from the DataFrame that contains
            at least 'topic', 'subtype', and 'relevance'.

    Returns:
        str: A single concatenated string, for example:
            "[主題]{topic}\n[子題]{subtype}\n[內容]\n{relevance}"
    """
    topic = str(row['topic']) if not pd.isna(row['topic']) else ""
    subtype = str(row['subtype']) if not pd.isna(row['subtype']) else ""
    relevance = str(row['relevance']) if not pd.isna(row['relevance']) else ""

    return f'[主題]{topic}\n[子題]{subtype}\n[內容]{relevance}'
    
def build_corpus_payload(excel_path: str)-> Tuple[List[str], List[Dict]]:
    """
    Load the knowledge corpus from an Excel file and construct:

      - DataFrame
      - List of texts for embedding
      - List of payload dictionaries for Qdrant

    Args:
        excel_path (str): Path to the Excel knowledge file.

    Returns:
        Tuple[List[str], List[dict]]:
            texts:    List of strings to be embedded.
            payloads: List of payload dicts, each aligned with one text.
    """

    df = pd.read_excel(excel_path)
    corpus = df.apply(build_text, axis=1).to_list()

    payloads  = []

    for idx, (i, row) in enumerate(df.iterrows()):  
        payloads.append(
            {
                'index': idx,  
                'topic': str(row['topic']),
                'subtype':str(row['subtype']),
                'relevance':str(row['relevance'])
            }
        )

    return corpus, payloads