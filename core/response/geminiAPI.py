import os, requests
from dotenv import load_dotenv
from core.response.prompt import prompt_template

# Load environment variables from .env file
load_dotenv()


def RAGresponse(query: str, top_k_docs: list[str], model: str = "gemini-2.5-flash") -> str:
    """
    Generate RAG-based response using Gemini API

    Args:
        query (str): User's question/query
        top_k_docs (List[str]): List of top-K retrieved documents from vector database
        model (str, optional): Model name to use. Defaults to "gemini-2.0-flash".

    Raises:
        ValueError: GEMINI_API_KEY not found in environment variables
        ValueError: Unexpected API response format

    Returns:
        str: Generated response from LLM
    """

    # Decline if no documents retrieved
    if not top_k_docs or top_k_docs == []:
        return "根據目前的知識文件，我沒有找到相關資訊。請聯繫人工客服以獲取進一步的協助。"


    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    data = {
        "systemInstruction": {"parts": [{"text": str(prompt_template.system_prompt)}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_template.construct(query, top_k_docs)}]}],
        "generationConfig": {"temperature": 0.3, "topP": 0.95, "topK": 40},
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
