import os
from transformers import BertTokenizerFast, AutoModelForTokenClassification
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi


class BM25:
    """
    BM25 model with CKIP word segmentation tokenizer.
    """

    def __init__(self, repo: str = "ckiplab/albert-tiny-chinese-ws", min_length: int = 1) -> None:
        """
        Initialize BM25 with CKIP tokenizer.
        
        Args:
            repo (str): HuggingFace model repository ID for Token Classification model.
            min_length (int): Minimum token length to keep.
        """
        self.repo = repo
        self.min_length = min_length
        self.model_dir = f"./models/{repo.split('/')[-1]}"
        
        self.__isDownloaded()
        self.tokenizer = CkipWordSegmenter(model_name=self.model_dir)
        self.bm25_model = None

    def __isDownloaded(self):
        """Download model from HuggingFace Hub if not exists locally."""
        if os.path.exists(self.model_dir): return
        
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Downloading model {self.repo}...")
        
        tokenizer = BertTokenizerFast.from_pretrained(self.repo)
        model = AutoModelForTokenClassification.from_pretrained(self.repo)
        tokenizer.save_pretrained(self.model_dir)
        model.save_pretrained(self.model_dir)

    def tokenize(self, texts: str | list[str]) -> list[list[str]]:
        """
        Tokenize text(s) using CKIP word segmenter.
        
        Args:
            texts (str | list[str]): Text or list of texts to tokenize.
            
        Returns:
            list[list[str]]: Tokenized documents.
        """
        if isinstance(texts, str): texts = [texts]
        tokenized = self.tokenizer(texts)
        return [[word for word in doc if len(word) >= self.min_length] for doc in tokenized]

    def fit(self, corpus: list[str]) -> 'BM25':
        """
        Fit BM25 model on corpus.
        
        Args:
            corpus (list[str]): List of documents.
            
        Returns:
            BM25: Self for method chaining.
        """
        self.bm25_model = BM25Okapi(self.tokenize(corpus))
        return self
