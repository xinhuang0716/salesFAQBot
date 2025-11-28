import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from sentence_transformers import SentenceTransformer


class DenseVector:

    def __init__(self, repo="google/embeddinggemma-300m"):
        """Transform text into dense embedding vectors using SentenceTransformer.

        Args:
            repo (str, optional): HuggingFace model repository ID. Defaults to "google/embeddinggemma-300m".
        """

        self.repo = repo
        self.root = "./models"
        self.modelDir = self.root + "/" + self.repo.split("/")[-1]

        self.__login()
        self.__isDownloaded()

        self.model = SentenceTransformer(self.modelDir)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length

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

    def encode(self, texts: str|list[str], normalize: bool = True, encode_type: str = "query") -> list[float]|list[list[float]]:
        """Encode text(s) into dense embedding vector(s).

        Args:
            texts (str | list[str]): Single text string or list of text strings to be encoded.
            normalize (bool, optional): Whether to normalize the embedding vectors. Defaults to True.
            encode_type (str, optional): SentenceTransformer supports different encoding methods for 'query' and 'document'. Defaults to "query".

        Raises:
            ValueError: Throws error if encode_type is not 'query' or 'document'.

        Returns:
            list[float]|list[list[float]]: The encoded embedding vector(s).
        """
        handlers = {
            "query": self.model.encode_query,
            "document": self.model.encode_document,
        }
        if encode_type not in handlers:
            raise ValueError("encode_type must be 'query' or 'document'")
        else:
            return handlers[encode_type](texts, normalize_embeddings=normalize)
