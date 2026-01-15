import os
from core.embedder.base_embedder import BaseEmbedder
from openai import AzureOpenAI

class AOAIEmbedder(BaseEmbedder):
    """Azure OpenAI Embedder implementation."""

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 1024,
                 api_version: str = None, azure_endpoint: str = None, api_key: str = None):
        """
        Initialize the Azure OpenAI Embedder.
        
        Args:
            model (str): Azure OpenAI embedding model name. Defaults to "text-embedding-3-large".
            dimensions (int): Output vector dimensions. Defaults to 1024. 
                            text-embedding-3-large supports: 256, 512, 1024, 1536, 3072 (default max).
            api_version (str, optional): API version. Falls back to env var AZURE_OPENAI_API_VERSION.
            azure_endpoint (str, optional): Azure endpoint URL. Falls back to env var AZURE_OPENAI_ENDPOINT.
            api_key (str, optional): API key. Falls back to env var AZURE_OPENAI_API_KEY.
        """
        self.client = AzureOpenAI(
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-ftw-jpe-sit-sandbox-01.openai.azure.com/"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
        )
        self.model = model
        self.dimensions = dimensions
    
    def encode(self, texts: str | list[str], encode_type: str) -> list[list[float]]:
        """
        Encode texts into embedding vectors using Azure OpenAI.
        
        Args:
            texts (str | list[str]): Single text or list of texts to encode.
            encode_type (str): Not used for Azure OpenAI (kept for interface compatibility).
        
        Returns:
            list[list[float]]: The encoded embedding vectors.
        """
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts
        
        response = self.client.embeddings.create(
            input=input_texts,
            model=self.model,
            dimensions=self.dimensions
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if is_single else embeddings