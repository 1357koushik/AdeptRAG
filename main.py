import os
import sys
import json
import tiktoken
import subprocess
from dotenv import load_dotenv
from classifier import classify_text
from model.extractor import extract_entities
from db.graph_store import GraphStore

# Load environment variables (API keys)
load_dotenv()

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

class AragCompleter(Completer):
    def __init__(self, available_models):
        self.commands = ['/mount', '/model', '/serve', '/quit', '/exit']
        self.models = available_models
        
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        parts = text.split(' ')
        if len(parts) == 1:
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
        elif len(parts) == 2 and parts[0] == '/model':
            word = parts[1]
            for m in self.models:
                if m.startswith(word):
                    yield Completion(m, start_position=-len(word))

# Enable ANSI escape sequences on Windows
if os.name == 'nt':
    os.system("")

#--------------| terminal utils |-------------
def enter_alt_screen():
    sys.stdout.write('\033[?1049h')
    sys.stdout.write('\033[H\033[J')
    sys.stdout.flush()

def exit_alt_screen():
    sys.stdout.write('\033[?1049l')
    sys.stdout.flush()

#--------------| main execution |-------------
def main():
    try:
        enter_alt_screen()
        
        current_model = "gpt-4o-mini"
        available_models = [
            "gpt-4o-mini", 
            "gpt-4o", 
            "claude-3-5-sonnet-20240620", 
            "gemini-2.5-flash-lite", 
            "ollama/llama3", 
            "ollama/mistral-nemo"
        ]
        
        graph_store = GraphStore()
        
        print("==================================================")
        print("            AdeptRAG Interactive CLI              ")
        print("==================================================")
        print("Available commands:")
        print("  /mount <folder> - Ingest documents from a folder")
        print("  /model <name>   - Change the extraction LLM model")
        print("  /serve          - Start the 3D Graph Web UI")
        print("  /quit           - Exit the CLI\n")
        
        session = PromptSession(
            completer=AragCompleter(available_models),
            complete_while_typing=True
        )
        
        while True:
            try:
                user_input = session.prompt("arag> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
                
            if not user_input:
                continue
                
            command = user_input.lower()
            
            if command in ['/quit', '/exit']:
                print("Exiting...")
                break
            elif command.startswith('/mount '):
                folder = user_input[7:].strip()
                if folder:
                    print(f"\n[Mounting folder: {folder}...]")
                    if os.path.exists(folder) and os.path.isdir(folder):
                        enc = tiktoken.get_encoding("cl100k_base")
                        chunk_size = 1200
                        chunk_overlap = 100
                        
                        allowed_exts = {'.txt', '.md', '.json', '.csv', '.py', '.js', '.html'}
                        for root, _, files in os.walk(folder):
                            for file in files:
                                if not any(file.endswith(ext) for ext in allowed_exts):
                                    continue
                                filepath = os.path.join(root, file)
                                try:
                                    with open(filepath, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    
                                    tokens = enc.encode(content)
                                    if not tokens:
                                        continue
                                        
                                    for i in range(0, len(tokens), chunk_size - chunk_overlap):
                                        chunk_tokens = tokens[i:i + chunk_size]
                                        chunk_text = enc.decode(chunk_tokens)
                                        
                                        result_json = classify_text(chunk_text)
                                        result = json.loads(result_json)
                                        
                                        label = result.get('label', 'Uncertain')
                                        
                                        if label == 'Data Dump':
                                            print(f"- {file} (Chunk {i // (chunk_size - chunk_overlap) + 1}): Dump Data (Skipping LLM)")
                                        else:
                                            print(f"- {file} (Chunk {i // (chunk_size - chunk_overlap) + 1}): Useful -> Extracting with {current_model}...")
                                            extracted = extract_entities(chunk_text, current_model)
                                            entities_count = sum(1 for e in extracted if e['type'] == 'entity')
                                            relations_count = sum(1 for e in extracted if e['type'] == 'relation')
                                            print(f"  -> Extracted {entities_count} entities, {relations_count} relations.")
                                            for item in extracted:
                                                if item['type'] == 'entity':
                                                    graph_store.upsert_entity(item['name'], item.get('entity_type', ''), item.get('description', ''))
                                                    print(f"     [Entity] {item['name']} ({item.get('entity_type', '')})")
                                                elif item['type'] == 'relation':
                                                    graph_store.upsert_relation(item['source'], item['target'], item.get('keywords', ''), item.get('description', ''))
                                                    print(f"     [Relation] {item['source']} -> {item['target']} ({item.get('keywords', '')})")
                                                    
                                except Exception as e:
                                    print(f"  Skipping {file}: {e}")
                                    
                        # Save graph to disk after processing folder
                        graph_store.save_to_disk()
                        
                    else:
                        print(f"Error: Directory '{folder}' not found.")
                    print("\n[Mount complete]\n")
                else:
                    print("Usage: /mount <folder_path>\n")
            elif command == '/mount':
                print("Usage: /mount <folder_path>\n")
            elif command.startswith('/model'):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    new_model = parts[1].strip()
                    current_model = new_model
                    print(f"\n[Model updated to: {current_model}]\n")
                else:
                    print(f"\nCurrent Model: {current_model}")
                    print("Best models for this job:")
                    for m in available_models:
                        print(f"  - {m}")
                    print("Usage: /model <model_name>\n")
            elif command == '/serve':
                print("\n[Starting Web Server on http://localhost:8000]")
                print("Press Ctrl+C to stop the server and return to CLI.\n")
                try:
                    # Launch uvicorn
                    subprocess.run([sys.executable, "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"])
                except KeyboardInterrupt:
                    pass
                print("\n[Server stopped]\n")
            else:
                print(f"Unknown command. Try '/mount <folder>', '/model <name>', '/serve', or '/quit'.\n")
    finally:
        # Guarantee we restore the terminal even if it crashes
        exit_alt_screen()

if __name__ == "__main__":
    main()