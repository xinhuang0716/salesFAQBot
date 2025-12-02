import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from sentence_transformers import CrossEncoder
from core.reranker.base_reranker import BaseReranker


class STEReranker(BaseReranker):

    def __init__(self, repo="jinaai/jina-reranker-v2-base-multilingual"):
        """
        Initialize the SentenceTransformer Reranker.

        Args:
            repo (str, optional): HuggingFace model repository ID. Defaults to "jinaai/jina-reranker-v2-base-multilingual".
        """
        self.repo: str = repo
        self.root: str = "./models"
        self.model_dir: str = os.path.join(self.root, self.repo.split("/")[-1])

        self.__login()
        self.__isDownloaded()

        self.model = CrossEncoder(self.model_dir, trust_remote_code=True)

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
        if os.path.exists(self.model_dir): return

        print(f"Downloading model {self.repo}...")
        snapshot_download(repo_id=self.repo, local_dir=self.model_dir)

    def rank(self, query: str, docs: list[str], top_k: int) -> list[dict]:
        """Rank documents based on their relevance to the query.

        Args:
            query (str): User query string
            docs (list[str]): Top K documents to be ranked
            top_k (int): Number of top documents to return after reranking.

        Returns:
            list[dict]: The list of dicts containing corpus_id, score, and text of the ranked documents, following the order from highest to lowest score.
        """
        return self.model.rank(query, docs, top_k, return_documents=True)