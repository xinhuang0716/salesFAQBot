import os
from transformers import BertTokenizerFast, AutoModelForTokenClassification
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi

class BM25Retriever:
   
    def __init__(self, repo: str = "ckiplab/albert-tiny-chinese-ws", minLength: int = 2):
        """Initialize the BM25 retriever.
        
        Args:
            repo_id (str, optional): HuggingFace model repository ID for word segmentation
            minlength (int, optional): Minimum token length to keep after segmentation
        """
        self.repo = repo
        self.minLength = minLength
        self.root = "./models"
        self.modelDir = self.root + "/" + self.repo.split("/")[-1]
        
        self.__isDownloaded()
        self.ws_driver = CkipWordSegmenter(model_name=self.modelDir)
        
        self.bm25 = None
        self.corpus = None
        self.tokenizedCorpus = None
    
    def __isDownloaded(self):
        """Download model from HuggingFace Hub if not exists locally."""
        if os.path.exists(self.modelDir): return
        
        os.makedirs(self.modelDir, exist_ok=True)
        print(f"Downloading model {self.repo}...")
        
        tokenizer = BertTokenizerFast.from_pretrained(self.repo)
        model = AutoModelForTokenClassification.from_pretrained(self.repo)
        tokenizer.save_pretrained(self.modelDir)
        model.save_pretrained(self.modelDir)
    
    def __Fitted(self):
        """Check if model has been fitted."""
        if self.bm25 is None:
            raise ValueError("BM25 model not fitted. Please call fit() first.")
    
    def tokenize(self, texts: str|list[str]) -> list[list[str]]:
        """_summary_

        Args:
            texts (str | list[str]): Single text string or list of text strings to be tokenized.

        Returns:
            list[list[str]]: List of tokenized documents.
        """
        texts = [texts] if isinstance(texts, str) else texts
        tokenized = self.ws_driver(texts)
        return [[word for word in doc if len(word) >= self.minLength] for doc in tokenized]
    
    def fit(self, corpus: list[str]) -> "BM25Retriever":
        """Fit the BM25 model on the provided corpus.

        Args:
            corpus (list[str]): List of tokenized documents to fit the BM25 model on.

        Returns:
            BM25Retriever: Self instance after fitting the model.
        """
        self.corpus = corpus
        self.tokenizedCorpus = self.tokenize(corpus)
        self.bm25 = BM25Okapi(self.tokenizedCorpus)
        print(f"BM25 model fitted on {len(corpus)} documents")
        return self
    
    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
        """Conduct BM25 search for a query and return top K results.

        Args:
            query (str): Search query string.
            top_k (int, optional): Retrieve top K results. Defaults to 3.

        Returns:
            list[tuple[int, float, str]]: List of tuples containing (document index, score, document text).
        """
        self.__Fitted()
        tokenizedQuery = self.tokenize(query)[0]
        scores = self.bm25.get_scores(tokenizedQuery)
        topIndices = scores.argsort()[::-1][:min(top_k, len(scores))]
        return [(int(idx), float(scores[idx]), self.corpus[idx]) for idx in topIndices]
