from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for different embedding backends."""

    @abstractmethod
    def encode_corpus(self, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        """Encode a list of documents into a 2D dense embedding array."""
        pass

    @abstractmethod
    def encode_query(self, query: str, max_length: int = 256) -> np.ndarray:
        """Encode a single query into a 1D dense embedding vector."""
        pass