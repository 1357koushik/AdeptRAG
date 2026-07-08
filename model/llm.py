import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai

load_dotenv()  # Ensure .env is loaded even if called before main()

def generate_text(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini", max_retries: int = 3, temperature: float = 0.0) -> str:
    """
    General helper for calling different LLMs.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    base_delay = 2
    
    if model_name == "mock":
        if "Extract keywords" in user_prompt:
            return '{"low_level": ["topic", "documents"], "high_level": ["main concept"]}'
        else:
            return "This is a mock answer based on the provided context."
            
    for attempt in range(max_retries):
        try:
            if model_name.startswith("gpt-"):
                client = OpenAI(timeout=30.0)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif model_name.startswith("claude-"):
                client = Anthropic(timeout=30.0)
                response = client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=4096
                )
                return response.content[0].text
                
            elif model_name.startswith("gemini-"):
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature
                    )
                )
                return response.text
                
            elif model_name.startswith("ollama/"):
                actual_model = model_name.replace("ollama/", "")
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=30.0)
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model prefix for {model_name}")
                
        except Exception as e:
            error_str = str(e).lower()
            if any(err in error_str for err in ["insufficient_quota", "invalid_api_key", "authentication", "unsupported", "not found", "missing credentials"]):
                print(f"  [X] Fatal API Error ({model_name}): {e}")
                return ""
                
            if attempt == max_retries - 1:
                print(f"  [X] Error during LLM generation ({model_name}) after {max_retries} attempts: {e}")
                return ""
                
            delay = base_delay * (2 ** attempt)
            error_msg = str(e).splitlines()[0][:100] if str(e) else "Unknown Error"
            print(f"  [!] API Error [{type(e).__name__}]: {error_msg}")
            print(f"      Retrying (attempt {attempt + 2}/{max_retries}) in {delay}s...")
            time.sleep(delay)
            print()

    return ""
