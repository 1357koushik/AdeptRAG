import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: 'huggingface_hub' library is not installed.")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

#--------------| configuration |-------------
MODEL_DIR = "model"
MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q8_0.gguf"
REPO_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

#--------------| model initialization |-------------
def init_model():
    if not os.path.exists(MODEL_DIR):
        print(f"Directory '{MODEL_DIR}' not found. Creating it...")
        os.makedirs(MODEL_DIR)
    else:
        print(f"Directory '{MODEL_DIR}' already exists.")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_FILE}' not found.")
        print(f"Starting download from Hugging Face using huggingface_hub...")
        
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILE,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            print("\nDownload completed successfully!")
            
        except Exception as e:
            print(f"\nAn error occurred during the download: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            sys.exit(1)
    else:
        print(f"Model file '{MODEL_FILE}' already exists in '{MODEL_DIR}/'. Ready to go!")

if __name__ == "__main__":
    init_model()