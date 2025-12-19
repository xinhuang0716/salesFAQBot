import os, requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main(model: str = "gemini-2.5-flash") -> str:
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found in environment variables")

    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    data = {
        "systemInstruction": {"parts": [{"text": "你是一個數學老師，請用繁體中文回答。"}]},
        "contents": [{"role": "user", "parts": [{"text": "1+1=?"}]}],
        "generationConfig": {"temperature": 0.3, "topP": 0.95, "topK": 40},
    }

    # Make API request
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Extract and return the generated text
    result = response.json()

    try:
        print(result["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected API response format: {e}\nResponse: {result}")

if __name__ == "__main__":
    main()