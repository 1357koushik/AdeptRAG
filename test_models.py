import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Add the current directory to sys.path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.extractor import extract_entities

TEST_CHUNK = """
In late 2021, Elon Musk founded xAI, an artificial intelligence company based in the San Francisco Bay Area. 
The company's mission is to understand the true nature of the universe. 
Musk recruited Igor Babuschkin, a former deep learning engineer at DeepMind, to help lead the technical efforts. 
By July 2023, xAI officially announced its team and its first AI model, Grok.
"""

MODELS_TO_TEST = [
    "gpt-4o-mini", 
    "gpt-4o", 
    "claude-3-5-sonnet-20240620", 
    "gemini-2.5-flash-lite", 
    "ollama/llama3", 
    "ollama/mistral-nemo"
]

def run_tests():
    print("========================================")
    print("   Testing LLM Extraction Pipeline      ")
    print("========================================\n")
    print("Test Chunk:\n" + TEST_CHUNK.strip() + "\n")
    print("-" * 40)
    
    for model in MODELS_TO_TEST:
        print(f"\n[Testing Model: {model}]")
        try:
            results = extract_entities(TEST_CHUNK, model_name=model)
            
            entities = [r for r in results if r['type'] == 'entity']
            relations = [r for r in results if r['type'] == 'relation']
            
            if not results:
                print("  -> Result: FAILED (Returned empty list. Possibly missing API key or Ollama server down?)")
            else:
                print(f"  -> Result: SUCCESS! Extracted {len(entities)} Entities, {len(relations)} Relations.")
                print("     Sample Entities:")
                for e in entities[:3]:
                    print(f"       - {e['name']} ({e.get('entity_type', 'Unknown')})")
                print("     Sample Relations:")
                for r in relations[:3]:
                    print(f"       - {r['source']} -> {r['target']} ({r.get('keywords', '')})")
                    
        except Exception as e:
            print(f"  -> ERROR: {e}")

if __name__ == "__main__":
    run_tests()
