import os
import urllib.request
import glob
import sys



def ensure_model_exists(model_dir="./model"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    existing_models = glob.glob(os.path.join(model_dir, "*.gguf"))
    if existing_models:
        return existing_models[0]
        
    print("No model found! Downloading default Qwen2.5-Coder model (~1.9GB)...")
    default_model_url = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q8_0.gguf"
    model_path = os.path.join(model_dir, "qwen2.5-coder-1.5b-instruct-q8_0.gguf")
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            progress_size = int(count * block_size)
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\rDownloading... {percent}% ({progress_size / (1024*1024):.2f} MB)")
            sys.stdout.flush()

    req = urllib.request.Request(default_model_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', -1))
        block_size = 8192
        count = 0
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            count += 1
            reporthook(count, block_size, total_size)
            
    print("\nDownload complete.")
    return model_path
