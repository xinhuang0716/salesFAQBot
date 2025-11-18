import pandas as pd
import numpy as np
import jieba
from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import List, Dict

class KnowledgeSearcher:
    def __init__(self, path: str, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 16, max_length: int = 512):
        """
        excel_path : Path to the RAG file
        model_name : Name of the embedding model to use
        device : "cpu" or "cuda"
        batch_size : Batch size used when encoding to build the index
        max_length : max_length used when encoding to build the index
        """
        self.excel_path = path
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # load RAG file
        self.df = pd.read_excel(path)
        self.texts = self.df.apply(self._build_text, axis=1).tolist() 

        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=True,
            device=self.device,
        )

        self.dense_vecs = self._build_dense_embeddings()
        self.tfidf_vectorizer, self.tfidf_matrix = self._build_tfidf_index()
        self.bm25, self.corpus_tokens = self._build_bm25_index()

    def _build_text(self, row: pd.Series) -> str:
        """
        Construct a normalized text representation from a single knowledge base row.

        Args:
            row (pd.Series): A row from the knowledge DataFrame containing at least
                the columns 'topic', 'subtype', and 'relevance'.

        Returns:
            str: A concatenated text in the format:
            "[主題]{topic}\\n[子題]{subtype}\\n[內容]\\n{relevance}".
        """
        
        topic = str(row["topic"]) if not pd.isna(row["topic"]) else ""
        subtype = str(row["subtype"]) if not pd.isna(row["subtype"]) else ""
        relevance = str(row["relevance"]) if not pd.isna(row["relevance"]) else ""
        return f"[主題]{topic}\n[子題]{subtype}\n[內容]\n{relevance}"


    @staticmethod
    def _jieba_tokenize(text: str):
        """
        Tokenize a Chinese text using jieba.

        This helper is designed to be passed as the tokenizer argument to scikit-learn's TfidfVectorizer.

        Args:
            text (str): Input text to tokenize.

        Returns:
            List[str]: A list of tokens produced by jieba.lcut.
        """
        return list(jieba.lcut(text))

    def _build_dense_embeddings(self) -> np.ndarray:
        """
        Encode all knowledge base texts into dense embeddings.

        Returns:
            np.ndarray: A 2D array of shape (num_docs, embedding_dim) containing dense embeddings for all documents.
        """

        emb = self.model.encode(
            self.texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        dense_vecs = emb["dense_vecs"]  # shape: (N, 1024)
        return dense_vecs

    def _build_tfidf_index(self):
        """
        Build the TF-IDF index over the knowledge base.

        Returns:
            Tuple[TfidfVectorizer, scipy.sparse.csr_matrix]:
                - The fitted TfidfVectorizer instance.
                - The TF-IDF matrix of shape (num_docs, vocab_size).
        """
        vectorizer = TfidfVectorizer(
            tokenizer=self._jieba_tokenize,
            token_pattern=None,
        )
        tfidf_matrix = vectorizer.fit_transform(self.texts)
        return vectorizer, tfidf_matrix

    def _build_bm25_index(self):
        """
        Build the BM25 index over the knowledge base.

        Returns:
            Tuple[BM25Okapi, List[List[str]]]:
                - The BM25Okapi instance fitted on the tokenized corpus.
                - The tokenized corpus, where each document is a list of tokens.
        """
        corpus_tokens = [list(jieba.lcut(text)) for text in self.texts]
        bm25 = BM25Okapi(corpus_tokens)
        return bm25, corpus_tokens

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into a dense embedding vector.

        Args:
            query (str): The user query in natural language.

        Returns:
            np.ndarray: A 1D array of shape (embedding_dim,) representing
                the dense embedding of the query.
        """
        q_emb = self.model.encode(
            [query],
            batch_size=1,
            max_length=256, # it depends
        )["dense_vecs"][0]
        return q_emb

    def search_dense(self, query: str, k: int = 5) -> List[dict]:
        """
        Perform dense semantic search using cosine similarity over embeddings.

        Args:
            query (str): The user query in natural language.
            k (int): Number of top documents to retrieve. Defaults to 5.

        Returns:
            List[dict]: A list of result dictionaries containing:
                - 'rank': Rank starting from 1.
                - 'score': Cosine similarity score.
                - 'topic': Topic of the matched document.
                - 'subtype': Subtype of the matched document.
                - 'relevance': Original relevance text.
                - 'index': Row index in the underlying DataFrame.
        """
        q = self.embed_query(query)

        # cosine similarity
        q_norm = q / np.linalg.norm(q)
        doc_norm = self.dense_vecs / np.linalg.norm(self.dense_vecs, axis=1, keepdims=True)
        scores = doc_norm @ q_norm  # shape: (N,)

        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            row = self.df.iloc[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[i]),
                    "topic": row["topic"],
                    "subtype": row["subtype"],
                    "relevance": str(row["relevance"]),
                    "index": int(i)
                }
            )
        return results

    def search_tfidf(self, query: str, k: int = 5) -> List[dict]:
        """
        Perform sparse retrieval using TF-IDF and jieba-based tokenization.

        Args:
            query (str): The user query in natural language.
            k (int): Number of top documents to retrieve. Defaults to 5.

        Returns:
            List[dict]: A list of result dictionaries containing:
                - 'rank': Rank starting from 1.
                - 'score': TF-IDF dot-product score.
                - 'topic': Topic of the matched document.
                - 'subtype': Subtype of the matched document.
                - 'relevance': Original relevance text.
                - 'index': Row index in the underlying DataFrame.
        """
        q_vec = self.tfidf_vectorizer.transform([query])  # shape: (1, V)
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()  # shape: (N,)

        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            row = self.df.iloc[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[i]),
                    "topic": row["topic"],
                    "subtype": row["subtype"],
                    "relevance": str(row["relevance"]),
                    "index": int(i),
                }
            )
        return results


    def search_bm25(self, query: str, k: int = 5):
        """
        Perform sparse retrieval using BM25 and jieba-based tokenization.

        Args:
            query (str): The user query in natural language.
            k (int): Number of top documents to retrieve. Defaults to 5.

        Returns:
            List[dict]: A list of result dictionaries containing:
                - 'rank': Rank starting from 1.
                - 'score': BM25 relevance score.
                - 'topic': Topic of the matched document.
                - 'subtype': Subtype of the matched document.
                - 'relevance': Original relevance text.
                - 'index': Row index in the underlying DataFrame.
        """
        query_tokens = list(jieba.lcut(query))
        scores = self.bm25.get_scores(query_tokens)  # shape: (N,)

        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            row = self.df.iloc[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[i]),
                    "topic": row["topic"],
                    "subtype": row["subtype"],
                    "relevance": str(row["relevance"]),
                    "index": int(i),
                }
            )
        return results


    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.7) -> List[dict]:
        """
        Perform hybrid retrieval by combining dense and BM25 scores.

        The final score is computed as a weighted linear combination:
            final_score = alpha * dense_score + (1 - alpha) * bm25_score

        Both dense and BM25 scores are rescaled to [0,1] before combining to reduce scale differences.

        Args:
            query (str): The user query in natural language.
            k (int): Number of top documents to retrieve. Defaults to 5.
            alpha (float): Weight for dense scores in [0, 1]. 
            A higher value biases towards semantic similarity, while a lower value biases towards exact keyword match. Defaults to 0.7.

        Returns:
            List[dict]: A list of result dictionaries containing:
                - 'rank': Rank starting from 1.
                - 'score': Final hybrid score.
                - 'dense_score': Normalized dense score component.
                - 'bm25_score': Normalized BM25 score component.
                - 'topic': Topic of the matched document.
                - 'subtype': Subtype of the matched document.
                - 'relevance': Original relevance text.
                - 'index': Row index in the underlying DataFrame.
        """
        q = self.embed_query(query)
        q_norm = q / np.linalg.norm(q)
        doc_norm = self.dense_vecs / np.linalg.norm(
            self.dense_vecs, axis=1, keepdims=True
        )
        dense_scores = doc_norm @ q_norm  # shape (N,)

        query_tokens = list(jieba.lcut(query))
        bm25_scores = self.bm25.get_scores(query_tokens)  # shape (N,)

        # Rescale to [0,1]
        def rescale(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-9:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        dense_norm = rescale(dense_scores)
        bm25_norm = rescale(bm25_scores)

        final_scores = alpha * dense_norm + (1 - alpha) * bm25_norm

        top_idx = np.argsort(final_scores)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            row = self.df.iloc[i]
            results.append(
                {
                    "rank": rank,
                    "score": float(final_scores[i]),
                    "dense_score": float(dense_norm[i]),
                    "bm25_score": float(bm25_norm[i]),
                    "topic": row["topic"],
                    "subtype": row["subtype"],
                    "relevance": str(row["relevance"]),
                    "index": int(i),
                }
            )
        return results


if __name__ == "__main__":
    searcher = KnowledgeSearcher("知識文件蒐集.xlsx", device="cpu")

    query = "'我的趨勢圖'頁面可顯示那些帳戶類別?"

    print("=== Dense ===")
    for r in searcher.search_dense(query, k=5):
        print(f"[{r['rank']}] {r['score']:.3f} | {r['topic']} / {r['subtype']}")
        print(r["relevance"])
        print("-" * 80)

    print("=== TF-IDF ===")
    for r in searcher.search_tfidf(query, k=5):
        print(f"[{r['rank']}] {r['score']:.3f} | {r['topic']} / {r['subtype']}")
        print(r["relevance"])
        print("-" * 80)

    print("=== BM25 ===")
    for r in searcher.search_bm25(query, k=5):
        print(f"[{r['rank']}] {r['score']:.3f} | {r['topic']} / {r['subtype']}")
        print(r["relevance"])
        print("-" * 80)

    print("=== Hybrid (dense + BM25) ===")
    for r in searcher.search_hybrid(query, k=5, alpha=0.7):
        print(
            f"[{r['rank']}] {r['score']:.3f} "
            f"(dense={r['dense_score']:.3f}, bm25={r['bm25_score']:.3f}) "
            f"| {r['topic']} / {r['subtype']}"
        )
        print(r["relevance"])
        print("-" * 80)


'''
Results:
=== Dense ===
[1] 0.763 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，可點選或取消欲檢視之帳戶別按鈕，折線圖及表格的資料會依照用戶所選擇的帳戶別進行顯示。
--------------------------------------------------------------------------------
[2] 0.756 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，選擇欲檢視之帳戶別下的帳路別 button 列表應與檢視條件中有勾選的帳戶別一致。可顯示的帳戶別包含證券、複委託、期貨、信託、衍商。若將其中一種帳 戶的勾取狀態取消時，該帳戶的button則會不顯示。
--------------------------------------------------------------------------------
[3] 0.729 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格中，點擊該年月資料最右側的向下箭頭，可以展開該年月的明細資料。明細資料所顯示的帳戶別會依照欲檢視之帳戶別按鈕列表的選擇狀況進行呈現。   
--------------------------------------------------------------------------------
[4] 0.699 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，趨勢圖呈現的資料範圍為 3 年，預設檢視為當年度的資料，可以使用滑鼠或手指滑掉下方的選取範圍檢視其他年月的資料。
--------------------------------------------------------------------------------
[5] 0.648 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格有三種排序功能，分別為依總資產、依市值、還原預設，且當為升冪排序或降冪排序時，對應的箭頭顯示不同顏色。預設的排列方式為 12 ~ 1 月份降冪 排序的總資產市值。
--------------------------------------------------------------------------------


=== TF-IDF ===
[1] 0.324 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，趨勢圖呈現的資料範圍為 3 年，預設檢視為當年度的資料，可以使用滑鼠或手指滑掉下方的選取範圍檢視其他年月的資料。
--------------------------------------------------------------------------------
[2] 0.296 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，點擊“ 雙箭頭迴圈" 按鈕，可以重新查詢資料，且顯示最新的頁面資料刷新時間。
--------------------------------------------------------------------------------
[3] 0.281 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，可點選或取消欲檢視之帳戶別按鈕，折線圖及表格的資料會依照用戶所選擇的帳戶別進行顯示。
--------------------------------------------------------------------------------
[4] 0.272 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，選擇欲檢視之帳戶別下的帳路別 button 列表應與檢視條件中有勾選的帳戶別一致。可顯示的帳戶別包含證券、複委託、期貨、信託、衍商。若將其中一種帳 戶的勾取狀態取消時，該帳戶的button則會不顯示。
--------------------------------------------------------------------------------
[5] 0.258 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格中，點擊該年月資料最右側的向下箭頭，可以展開該年月的明細資料。明細資料所顯示的帳戶別會依照欲檢視之帳戶別按鈕列表的選擇狀況進行呈現。   
--------------------------------------------------------------------------------


=== BM25 ===
[1] 13.309 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格中，點擊該年月資料最右側的向下箭頭，可以展開該年月的明細資料。明細資料所顯示的帳戶別會依照欲檢視之帳戶別按鈕列表的選擇狀況進行呈現。   
--------------------------------------------------------------------------------
[2] 13.187 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，選擇欲檢視之帳戶別下的帳路別 button 列表應與檢視條件中有勾選的帳戶別一致。可顯示的帳戶別包含證券、複委託、期貨、信託、衍商。若將其中一種帳 戶的勾取狀態取消時，該帳戶的button則會不顯示。
--------------------------------------------------------------------------------
[3] 13.074 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，趨勢圖呈現的資料範圍為 3 年，預設檢視為當年度的資料，可以使用滑鼠或手指滑掉下方的選取範圍檢視其他年月的資料。
--------------------------------------------------------------------------------
[4] 12.903 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，可點選或取消欲檢視之帳戶別按鈕，折線圖及表格的資料會依照用戶所選擇的帳戶別進行顯示。
--------------------------------------------------------------------------------
[5] 12.784 | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，點擊“ 雙箭頭迴圈" 按鈕，可以重新查詢資料，且顯示最新的頁面資料刷新時間。
--------------------------------------------------------------------------------


=== Hybrid (dense + BM25) ===
[1] 0.991 (dense=1.000, bm25=0.969) | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，可點選或取消欲檢視之帳戶別按鈕，折線圖及表格的資料會依照用戶所選擇的帳戶別進行顯示。
--------------------------------------------------------------------------------
[2] 0.984 (dense=0.980, bm25=0.991) | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，選擇欲檢視之帳戶別下的帳路別 button 列表應與檢視條件中有勾選的帳戶別一致。可顯示的帳戶別包含證券、複委託、期貨、信託、衍商。若將其中一種帳 戶的勾取狀態取消時，該帳戶的button則會不顯示。
--------------------------------------------------------------------------------
[3] 0.939 (dense=0.913, bm25=1.000) | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格中，點擊該年月資料最右側的向下箭頭，可以展開該年月的明細資料。明細資料所顯示的帳戶別會依照欲檢視之帳戶別按鈕列表的選擇狀況進行呈現。   
--------------------------------------------------------------------------------
[4] 0.881 (dense=0.837, bm25=0.982) | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，趨勢圖呈現的資料範圍為 3 年，預設檢視為當年度的資料，可以使用滑鼠或手指滑掉下方的選取範圍檢視其他年月的資料。
--------------------------------------------------------------------------------
[5] 0.747 (dense=0.705, bm25=0.845) | 資產總覽 / 功能說明
於【我的趨勢圖】頁面，表格有三種排序功能，分別為依總資產、依市值、還原預設，且當為升冪排序或降冪排序時，對應的箭頭顯示不同顏色。預設的排列方式為 12 ~ 1 月份降冪 排序的總資產市值。
--------------------------------------------------------------------------------
'''
