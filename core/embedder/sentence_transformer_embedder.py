import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from sentence_transformers import SentenceTransformer
from core.embedder.base_embedder import BaseEmbedder


class STEmbedder(BaseEmbedder):
    """SentenceTransformer Embedder implementation."""

    def __init__(self, repo: str = "BAAI/bge-m3"):
        """
        Initialize the SentenceTransformer Embedder.

        Args:
            repo (str, optional): HuggingFace model repository ID. Defaults to "BAAI/bge-m3".
        """

        self.repo = repo
        self.root = "./models"
        self.model_dir = self.root + "/" + self.repo.split("/")[-1]

        self.__login()
        self.__isDownloaded()

        self.model = SentenceTransformer(self.model_dir)

    def __login(self):
        """Login to HuggingFace Hub using environment variable."""
        try:
            load_dotenv()
            login(os.getenv("HUGGINGFACE_LLM_Model"))
        except:
            raise EnvironmentError("HuggingFace login failed. Please check your HUGGINGFACE_LLM_Model environment variable.")

    def __isDownloaded(self):
        """Download model from HuggingFace Hub if not exists locally."""
        os.makedirs("./models", exist_ok=True)
        if os.path.exists(self.model_dir):  return print(f"Model {self.repo} already exists.")

        print(f"Downloading model {self.repo}...")
        snapshot_download(repo_id=self.repo, local_dir=self.model_dir)

    def encode(self, texts: str | list[str], encode_type: str) -> list[list[float]]:
        """
        The method for encoding texts into embedding vectors.

        Args:
            texts (str | list[list[str]]): Single corpus string or list of corpus strings to be encoded
            encode_type (str): SentenceTransformer supports different encoding methods for 'query' and 'document'. Defaults to "query".

        Returns:
            list[list[float]]: The encoded embedding vector(s).
        """

        handlers = {
            "query": self.model.encode_query,
            "document": self.model.encode_document,
        }

        if encode_type not in handlers:
            raise ValueError("encode_type must be 'query' or 'document'")
        else:
            return handlers[encode_type](texts, normalize_embeddings=True)
