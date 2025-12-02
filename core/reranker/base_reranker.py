from abc import ABC, abstractmethod

class BaseReranker(ABC):
    """Abstract base class for different reranker backends."""

    @abstractmethod
    def rank(self, query: str, docs: list[str], top_k: int) -> list[dict]:
        """
        The abstract method for reranking retrieved documents based on their relevance to the query.

        Args:
            query (str): Single corpus string or list of corpus strings to be encoded.
            docs (list[str]): Top K documents to be ranked, each as a string.
            top_k (int): Number of top documents to return after reranking.


        Returns:
            list[dict]: The list of ranked documents with their idx and scores.
        """
        pass
