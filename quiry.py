import os
from server_downloader import ensure_server_exists
from model_downloader import ensure_model_exists
from llm_server import LocalLlama

SERVER_EXE = r"./bin/llama-server.exe" 


def main():
    ensure_server_exists(SERVER_EXE)
    MODEL_FILE = ensure_model_exists("./model")
    
    with LocalLlama(SERVER_EXE, MODEL_FILE) as llm:
        print("\nSending prompt...")
        answer = llm.ask("Explain quantum computing in one sentence.")
        
        print("\n--- Result ---")
        print(answer)
        print("--------------\n")

if __name__ == "__main__":
    main()