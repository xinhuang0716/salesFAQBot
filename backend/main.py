from qdrant_client import QdrantClient
from embedder.bge_embedder import BGEEmbedder
from dense_search import DenseSearcher
from bm25_search import BM25Searcher
from hybrid_rrf_search import HybridRRFSearcher 
from corpus_builder import build_corpus_payload
import pandas as pd

def find_rank(results, true_id: int, k: int) -> int | None:
    """
    """
    for r in results:
        if int(r["doc_id"]) == true_id:
            return int(r["rank"])
    return None

def main(k: int =20):
    client = QdrantClient(path="./db") 
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="cpu")

    dense_searcher = DenseSearcher(
        client=client,
        embedder=embedder,
        collection_name="FAQ",  
    )

    texts, payloads = build_corpus_payload("知識文件蒐集.xlsx")
    bm25_searcher = BM25Searcher(
        texts=texts,
        payloads=payloads,
        ws_model="bert-base",  
        device=-1,             
    )

    hybrid_searcher = HybridRRFSearcher(
        dense_searcher=dense_searcher,
        bm25_searcher=bm25_searcher,
        k_param=60,  
    )
    
    df = pd.read_excel("nonFAQ.xlsx")
    records = []

    for i, row in df.iterrows():
        query = row.get("Question")

        true_id = int(i)  
        print(f"Evaluating row: {i}, query: {query!r}")

        dense_results = dense_searcher.search(query, k=k)
        bm25_results = bm25_searcher.search(query, k=k)
        hybrid_results = hybrid_searcher.search(query, k=k, dense_k=50, bm25_k=50)

        dense_rank = find_rank(dense_results, true_id, k)
        bm25_rank = find_rank(bm25_results, true_id, k)
        hybrid_rank = find_rank(hybrid_results, true_id, k)

        dense_scores = [round(float(r["score"]),3) for r in dense_results]
        bm25_scores = [round(float(r["score"]),3) for r in bm25_results]
        hybrid_scores = [round(float(r["rrf_score"]),3) if "rrf_score" in r else float(r["score"])
                     for r in hybrid_results]
        

        records.append(
            {
                "row_index": i,
                "query": query,

                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "hybrid_rank": hybrid_rank,

                "dense_scores": dense_scores,
                "bm25_scores": bm25_scores,
                "hybrid_scores": hybrid_scores,

            }
        )

    client.close()

    eval_df = pd.DataFrame(records)
    eval_df.to_excel("nonFAQ_eval.xlsx", index=False)


    def mrr(col: str) -> float:
        ranks = eval_df[col].dropna().astype(int)
        if len(ranks) == 0:
            return 0.0
        return float((1.0 / ranks).mean())

    def hit_rate(col: str) -> float:
        return float(eval_df[col].notna().mean())

    for name in ["dense_rank", "bm25_rank", "hybrid_rank"]:
        print(f"=== {name} ===")
        print(f"  Hit@{k}: {hit_rate(name):.3f}")
        print(f"  MRR@{k}: {mrr(name):.3f}")


    query = "OTP驗證可透過哪些方式接收驗證碼？"

    print("=== Dense ===")
    dense_results = dense_searcher.search(query, k=10)
    for r in dense_results:
        print(f"[{r['rank']}] {r['score']:.3f} | {r['topic']} / {r['subtype']} | Doc_ID: {r['doc_id']}")
        print(r["relevance"])
        print("-" * 80)

    print("=== BM25 ===")
    bm25_results = bm25_searcher.search(query, k=10)
    for r in bm25_results:
        print(f"[{r['rank']}] {r['score']:.3f} | {r['topic']} / {r['subtype']} | Doc_ID: {r['doc_id']}")
        print(r["relevance"])
        print("-" * 80)

    print("=== Hybrid RRF (dense + BM25) ===")
    for r in hybrid_searcher.search(query, k=10, dense_k=20, bm25_k=20):
        print(f"[{r['rank']}] RRF={r['rrf_score']:.4f} | "
            f"dense_rank={r['dense_rank']}, bm25_rank={r['bm25_rank']} | Doc_ID: {r['doc_id']}"
            )
        print(f"{r['topic']} / {r['subtype']}")
        print(r["relevance"])
        print("-" * 80)

if __name__ == "__main__":
    main(k=10)





