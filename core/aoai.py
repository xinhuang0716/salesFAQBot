"""Azure OpenAI response generation client."""

import logging

import httpx

from core.context import PromptTemplate

logger = logging.getLogger(__name__)


class AOAIClient:
    """Client for generating responses using Azure OpenAI."""

    def __init__(self, endpoint: str, api_key: str, http_client: httpx.AsyncClient, model: str = "gpt-5.2") -> None:
        """Initialize the AOAI client.

        Args:
            endpoint (str): Azure OpenAI API endpoint.
            api_key (str): Azure OpenAI API key.
            http_client (httpx.AsyncClient): Shared async HTTP client for connection pooling.
            model (str, optional): Deployment model name. Defaults to "gpt-5.2".

        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.http = http_client

    async def rag_response(self, query: str, top_k_docs: list[str]) -> str:
        """Generate RAG-based response using AOAI API.

        Args:
            query (str): User's question/query.
            top_k_docs (list[str]): List of top-K retrieved documents.

        Returns:
            str: Generated response from LLM.

        """
        if not top_k_docs:
            logger.info("No retrieved documents available for query: %s", query[:20])
            return "根據目前的知識文件，我沒有找到相關資訊。請聯繫人工客服以獲取進一步的協助。"

        # Construct the AOAI API request payload.
        url = f"{self.endpoint.rstrip('/')}/openai/v1/responses"
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": PromptTemplate.system_prompt()},
                {"role": "user", "content": PromptTemplate.construct(query, top_k_docs)},
            ],
            "temperature": 0.3,
            "top_p": 0.95,
        }

        # Request.
        logger.info("Sending AOAI request to %s with model=%s", url, self.model)
        response = await self.http.post(url, headers=headers, json=payload)

        # Response parsing.
        response.raise_for_status()
        result = response.json()

        try:
            text = result["output"][0]["content"][0]["text"]
            logger.info("AOAI response received successfully: %s", text[:20])

            return text

        except Exception as e:
            logger.error("AOAI response format validation failed.")
            raise ValueError(f"Unexpected API response format: {e}\nResponse: {result}") from e
