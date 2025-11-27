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

def topK(query, k: int, k_reranker: int, search = 'bm25'):
    reranker = BGEReranker() 
    
    if search == 'dense':
        client = QdrantClient(path="./db") 
        embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="cpu",local_dir="./models/")

        dense_searcher = DenseSearcher(
            client=client,
            embedder=embedder,
            collection_name="FAQ",  
        )
        candidates = dense_searcher.search(
                query=query,
                k=k
        )
        final_results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=k_reranker,
            text_key="relevance",
        )
        
        top_k_list = []
        for r in final_results:
            relevance = r.get("relevance")
            top_k_list.append(relevance)


    if search == 'bm25':
        texts, payloads = build_corpus_payload("./docs/知識文件蒐集.xlsx")
        bm25_searcher = BM25Searcher(
            texts=texts,
            payloads=payloads,
            ws_model="bert-base",  
            device=-1,             
        )
        candidates = bm25_searcher.search(
                query=query,
                k=k
        )
        final_results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=k_reranker,
            text_key="relevance",
        )

        top_k_list = []
        for r in final_results:
            relevance = r.get("relevance")
            top_k_list.append(relevance)


    return top_k_list


if __name__ == "__main__":
    main(k=10)






