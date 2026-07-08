import os
import time
from openai import OpenAI
from anthropic import Anthropic
from google import genai

# Set API key for OpenAI if needed (litellm uses standard env vars)
# os.environ["OPENAI_API_KEY"] = "your-key"

ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` section of user prompt.

---Instructions---
1. **Entity Extraction:**
  - Identify clearly defined and meaningful entities only in the current user prompt's fenced `---Input Text---` section.
  - For each entity, extract:
    - `entity_name`: The name of the entity. Capitalize the first letter of each significant word (title case). Ensure consistent naming.
    - `entity_type`: Categorize the entity (e.g., Person, Organization, Location, Event, Concept).
    - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes.

2. **Relationship Extraction:**
  - Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - For each binary relationship, extract:
    - `source_entity`: The name of the source entity.
    - `target_entity`: The name of the target entity.
    - `relationship_keywords`: One or more high-level keywords summarizing the relationship.
    - `relationship_description`: A concise explanation of the nature of the relationship.

3. **Record Types:**
  - `entity` is used only for entity rows and those rows always contain exactly 4 tuple parts total.
  - `relation` is used only for relationship rows and those rows always contain exactly 5 tuple parts total.

4. **Output Format:**
  - Entity row: `entity<|TUPLE_DELIMITER|>entity_name<|TUPLE_DELIMITER|>entity_type<|TUPLE_DELIMITER|>entity_description`
  - Relation row: `relation<|TUPLE_DELIMITER|>source_entity<|TUPLE_DELIMITER|>target_entity<|TUPLE_DELIMITER|>relationship_keywords<|TUPLE_DELIMITER|>relationship_description`

5. **Delimiter Usage:**
  - The `<|TUPLE_DELIMITER|>` is a complete, atomic marker and must not be filled with content. It serves strictly as a field separator.

6. **Output Order:**
  - Output all extracted entities first, followed by all extracted relationships.

7. **Completion Signal:** Output the literal string `<|COMPLETION|>` only after all entities and relationships have been completely extracted and outputted.
"""

ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the `---Input Text---` section below.

---Instructions---
1. Strictly adhere to all format requirements for entity and relationship lists.
2. Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks.
3. Output `<|COMPLETION|>` as the final line after all relevant entities and relationships have been extracted and presented.

---Input Text---
```
{input_text}
```

---Output---
"""

def extract_entities(chunk_text: str, model_name: str = "gpt-4o-mini") -> list[dict]:
    """
    Extract entities and relationships from a text chunk using litellm.
    """
    if model_name == "mock":
        llm_output = (
            'entity<|TUPLE_DELIMITER|>Elon Musk<|TUPLE_DELIMITER|>person<|TUPLE_DELIMITER|>Founder of xAI.\n'
            'entity<|TUPLE_DELIMITER|>xAI<|TUPLE_DELIMITER|>organization<|TUPLE_DELIMITER|>An artificial intelligence company based in the San Francisco Bay Area.\n'
            'relation<|TUPLE_DELIMITER|>Elon Musk<|TUPLE_DELIMITER|>xAI<|TUPLE_DELIMITER|>founder<|TUPLE_DELIMITER|>Elon Musk founded xAI in late 2021.'
        )
        return parse_extraction_output(llm_output)

    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": ENTITY_EXTRACTION_USER_PROMPT.format(input_text=chunk_text)}
    ]
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            content = ""
            
            if model_name.startswith("gpt-"):
                client = OpenAI(timeout=20.0)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message.content
                
            elif model_name.startswith("claude-"):
                client = Anthropic(timeout=20.0)
                response = client.messages.create(
                    model=model_name,
                    system=ENTITY_EXTRACTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": ENTITY_EXTRACTION_USER_PROMPT.format(input_text=chunk_text)}],
                    temperature=0.0,
                    max_tokens=4096
                )
                content = response.content[0].text
                
            elif model_name.startswith("gemini-"):
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=model_name,
                    contents=ENTITY_EXTRACTION_USER_PROMPT.format(input_text=chunk_text),
                    config=genai.types.GenerateContentConfig(
                        system_instruction=ENTITY_EXTRACTION_SYSTEM_PROMPT,
                        temperature=0.0
                    )
                )
                content = response.text
                
            elif model_name.startswith("ollama/"):
                actual_model = model_name.replace("ollama/", "")
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=20.0)
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model prefix for {model_name}")

            return parse_extraction_output(content)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Fail fast on non-recoverable errors (invalid API key, out of credits, unsupported model)
            if any(err in error_str for err in ["insufficient_quota", "invalid_api_key", "authentication", "unsupported", "not found", "missing credentials"]):
                print(f"  [X] Fatal API Error ({model_name}): {e}")
                print(f"      -> Skipping retries. Please check your API key and billing status.")
                return []
                
            if attempt == max_retries - 1:
                print(f"  [X] Error during LLM extraction ({model_name}) after {max_retries} attempts: {e}")
                return []
                
            delay = base_delay * (2 ** attempt)
            import sys
            
            error_msg = str(e).splitlines()[0][:100] if str(e) else "Unknown Error"
            print(f"  [!] API Error [{type(e).__name__}]: {error_msg}")
            
            # Safe cross-platform live countdown loop
            sys.stdout.write(f"      Waiting before attempt {attempt + 2}/{max_retries}: ")
            sys.stdout.flush()
            for remaining in range(delay, 0, -1):
                sys.stdout.write(f"{remaining}.. ")
                sys.stdout.flush()
                time.sleep(1)
            print() # Newline after countdown finishes

def parse_extraction_output(llm_output: str) -> list[dict]:
    """
    Parses the raw text output from the LLM into structured entity and relation dictionaries.
    """
    results = []
    lines = llm_output.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line == "<|COMPLETION|>":
            continue
            
        parts = line.split("<|TUPLE_DELIMITER|>")
        
        if parts[0] == "entity" and len(parts) >= 4:
            results.append({
                "type": "entity",
                "name": parts[1].strip(),
                "entity_type": parts[2].strip(),
                "description": parts[3].strip()
            })
        elif parts[0] == "relation" and len(parts) >= 5:
            results.append({
                "type": "relation",
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "keywords": parts[3].strip(),
                "description": parts[4].strip()
            })
            
    return results
