import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self, repo="jinaai/jina-reranker-v2-base-multilingual"):
        """

        Args:
            repo (str, optional): HuggingFace model repository ID. Defaults to "jinaai/jina-reranker-v2-base-multilingual".
        """

        self.repo = repo
        self.root = "./models"
        self.modelDir = self.root + "/" + self.repo.split("/")[-1]

        self.__login()
        self.__isDownloaded()

        self.model = CrossEncoder(
            self.modelDir,
            trust_remote_code=True,
        )

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
        if os.path.exists(self.modelDir): return

        print(f"Downloading model {self.repo}...")
        snapshot_download(repo_id=self.repo, local_dir=self.modelDir)

    def rank(self, query: str, docs: list[str], **kwargs) -> list[dict]:
        """Rank documents based on their relevance to the query.

        Args:
            query (str): User query string
            docs (list[str]): Top K documents to be ranked

        Returns:
            list[dict]: The list of ranked documents with their scores.
        """
        return self.model.rank(query, docs, **kwargs)