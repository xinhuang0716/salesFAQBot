import os, sys, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.init.builder import initialize
from core.reranker.sentence_transformer_reranker import STEReranker
from core.retrieve.rerank_search import RerankerSearcher

def main():
    client, embedder = initialize()
    reranker = STEReranker()

    search: RerankerSearcher = RerankerSearcher(client=client, embedder=embedder, reranker=reranker)
    query: str = "若有兩位以上未成年人開戶，法定代理人同意書應簽署幾份？"
    results: list[dict] = search.search(query=query, k=10, score=0.4, top_k=5)

    for res in results:
        print(f"Rank {res['rank']}: Doc ID {res['doc_id']}, Score {res['score']}, Topic {res['topic']}, Subtype {res['subtype']}, Relevance {res['relevance']}")

if __name__ == "__main__":
    main() 