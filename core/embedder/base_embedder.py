from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """Abstract base class for different embedding backends."""

    @abstractmethod
    def encode(self, texts: str|list[str], encode_type: str) -> list[list[float]]:
        """
        The abstract method for encoding texts into embedding vectors.

        Args:
            texts (str | list[list[str]]): Single corpus string or list of corpus strings to be encoded
            encode_type (str): SentenceTransformer supports different encoding methods for 'query' and 'document'. Defaults to "query".

        Returns:
            list[list[float]]: The encoded embedding vector(s).
        """
        pass