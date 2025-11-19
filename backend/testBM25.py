from utils.BM25 import BM25Retriever

def main():
    """Example usage of BM25Retriever."""
    corpus = [
        "人工智慧是現代科技的重要領域。",
        "自然語言處理是人工智慧的一個分支。",
        "BM25 是一種資訊檢索演算法。",
        "CKIPTagger 可以處理繁體中文斷詞。"
    ]
    
    retriever = BM25Retriever(minLength=2).fit(corpus)
    
    print("\n斷詞結果：")
    print("=" * 60)
    for i, (original, tokenized) in enumerate(zip(corpus, retriever.tokenizedCorpus)):
        print(f"文件 {i}: {original}")
        print(f"  過濾後: {tokenized}\n")
    
    query = "什麼是自然語言處理？"
    print(f"\n查詢: {query}")
    print("=" * 60)
    
    results = retriever.search(query, top_k=3)
    
    print("\n搜尋結果 (Top 3)：")
    print("=" * 60)
    for rank, (idx, score, text) in enumerate(results, 1):
        print(f"{rank}. 文件 {idx} (分數: {score:.2f})")
        print(f"   {text}\n")


if __name__ == "__main__":
    main()