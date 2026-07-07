import os
import sys
import json
import tiktoken
from classifier import classify_text

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
        
        print("==================================================")
        print("            AdeptRAG Interactive CLI              ")
        print("==================================================")
        print("Available commands:")
        print("  /mount <folder> - Ingest documents from a folder")
        print("  /quit           - Exit the CLI\n")
        
        while True:
            try:
                user_input = input("arag> ").strip()
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
                        chunk_size = 512
                        
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
                                    for i in range(0, len(tokens), chunk_size):
                                        chunk_tokens = tokens[i:i + chunk_size]
                                        chunk_text = enc.decode(chunk_tokens)
                                        
                                        result_json = classify_text(chunk_text)
                                        result = json.loads(result_json)
                                        
                                        label = result.get('label', 'Uncertain')
                                        
                                        if label == 'Data Dump':
                                            final_output = 'Dump Data'
                                        else:
                                            final_output = 'Useful'
                                            
                                        print(f"- {file} (Chunk {i//chunk_size + 1}): {final_output}")
                                except Exception as e:
                                    print(f"  Skipping {file}: {e}")
                    else:
                        print(f"Error: Directory '{folder}' not found.")
                    print("\n[Mount complete]\n")
                else:
                    print("Usage: /mount <folder_path>\n")
            elif command == '/mount':
                print("Usage: /mount <folder_path>\n")
            else:
                print(f"Unknown command. Try '/mount <folder>' or '/quit'.\n")
    finally:
        # Guarantee we restore the terminal even if it crashes
        exit_alt_screen()

if __name__ == "__main__":
    main()