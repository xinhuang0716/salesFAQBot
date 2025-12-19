import os, requests
from dotenv import load_dotenv
from core.response.prompt import prompt_template

# Load environment variables from .env file
load_dotenv()


def RAGresponse(query: str, top_k_docs: list[str]) -> str:
    """
    Generate RAG-based response using AOAI API

    Args:
        query (str): User's question/query
        top_k_docs (List[str]): List of top-K retrieved documents from vector database

    Raises:
        ValueError: AOAI_API_KEY or AOAI_ENDPOINT not found in environment variables
        ValueError: Unexpected API response format

    Returns:
        str: Generated response from LLM
    """

    # Get API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not api_key or not endpoint:
        raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not found in environment variables")

    # API configuration
    url = f"{endpoint}/openai/deployments/gpt-4o/chat/completions?api-version={version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": [
            {"role": "system", "content": str(prompt_template.system_prompt)},
            {"role": "user", "content": prompt_template.construct(query, top_k_docs)}
        ],
        "temperature": 0.3,
        "top_p": 0.95
    }

    # Make API request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    # Extract and return the generated text
    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response format: {e}\nResponse: {result}")
