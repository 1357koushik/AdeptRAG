import os
import sys
from server_downloader import ensure_server_exists
from model_downloader import ensure_model_exists
from llm_server import LocalLlama

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
    SERVER_EXE = r"./bin/llama-server.exe" 
    
    # Check dependencies before taking over the screen so any download progress is visible
    ensure_server_exists(SERVER_EXE)
    MODEL_FILE = ensure_model_exists("./model")

    try:
        enter_alt_screen()
        
        print("==================================================")
        print("            AdeptRAG Interactive CLI              ")
        print("==================================================")
        print("Starting local AI Server... please wait.\n")
        
        with LocalLlama(SERVER_EXE, MODEL_FILE) as llm:
            # Clear the screen once the server is successfully booted
            sys.stdout.write('\033[H\033[J')
            sys.stdout.flush()
            
            print("==================================================")
            print("            AdeptRAG Interactive CLI              ")
            print("==================================================")
            print("Server is Online! Available commands:")
            print("  /msg <text> - Chat with the model")
            print("  /quit       - Exit and shut down the server\n")
            
            while True:
                try:
                    user_input = input("arag> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nShutting down...")
                    break
                    
                if not user_input:
                    continue
                    
                command = user_input.lower()
                
                if command in ['/quit', '/exit']:
                    print("Shutting down server and exiting...")
                    break
                elif command.startswith('/msg '):
                    prompt = user_input[5:].strip()
                    if prompt:
                        print("\n[AI thinking...]")
                        answer = llm.ask(prompt)
                        print(f"\n[AI]: {answer}\n")
                    else:
                        print("Usage: /msg <your message>\n")
                elif command == '/msg':
                    print("Usage: /msg <your message>\n")
                else:
                    print(f"Unknown command. Try '/msg <text>' or '/quit'.\n")
    finally:
        # Guarantee we restore the terminal even if it crashes
        exit_alt_screen()

if __name__ == "__main__":
    main()