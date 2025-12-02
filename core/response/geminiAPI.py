import os, requests
from dotenv import load_dotenv
from core.response.prompt import prompt_constructor

# Load environment variables from .env file
load_dotenv()


def RAGresponse(query: str, top_k_docs: list[str],  model: str = "gemini-2.0-flash", temperature: float = 0.3) -> str:
    """
    Generate RAG-based response using Gemini API

    Args:
        query (str): User's question/query
        top_k_docs (List[str]): List of top-K retrieved documents from vector database
        model (str, optional): Model name to use. Defaults to "gemini-2.0-flash".
        temperature (float, optional): Controls randomness (0.0-1.0). Defaults to 0.3.

    Raises:
        ValueError: GEMINI_API_KEY not found in environment variables
        ValueError: Unexpected API response format

    Returns:
        str: Generated response from LLM
    """

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found in environment variables")

    # Construct the full prompt
    prompt = prompt_constructor(query, top_k_docs)

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "topP": 0.95, "topK": 40},
    }

    # Make API request
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Extract and return the generated text
    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response format: {e}\nResponse: {result}")