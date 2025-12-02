import os, sys, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.embedder.bm25 import BM25
from core.retrieve.bm25_search import BM25Searcher
from utils.corpus import build_text

def main():
    file: str = os.listdir("./knowledgeDoc")[-1]
    payloads: list[dict] = pd.read_excel("./knowledgeDoc/" + file, usecols=["id", "source", "topic", "subtype", "relevance"]).to_dict(orient="records")

    bm25 = BM25(repo="ckiplab/albert-tiny-chinese-ws", min_length=1)
    bm25.fit([build_text(row) for row in payloads])

    searcher = BM25Searcher(bm25=bm25, payloads=payloads)

    results, scores = searcher.search(query="雙箭頭 按鈕", k=5)
    for _, res in enumerate(results):
        print(f"Rank {res['rank']}: Doc ID {res['doc_id']}, Score {res['score']}, Topic {res['topic']}, Subtype {res['subtype']}, Relevance {res['relevance']}")

    print("\nAll Scores:", scores)

if __name__ == "__main__":
    main()