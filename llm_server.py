import os
import time
import json
import subprocess
import urllib.request
import shutil

#--------------| vram detection |-------------
def get_vram_mb():
    try:
        if shutil.which("nvidia-smi") is not None:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            vram = sum(int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip().isdigit())
            return vram
    except Exception:
        pass
    return 0

#--------------| local server class |-------------
class LocalLlama:
    def __init__(self, server_path, model_path, port=8080):
        self.server_path = server_path
        self.model_path = model_path
        self.port = port
        self.process = None

    def __enter__(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Cannot find your .gguf model file at: {self.model_path}")

        print("Starting llama-server...")
        cmd = [self.server_path, "-m", self.model_path, "--port", str(self.port)]
        
        vram_mb = get_vram_mb()
        if vram_mb > 0:
            print(f"Detected {vram_mb} MB of VRAM. Instructing server to load model into VRAM...")
            cmd.extend(["-ngl", "99"])
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        health_url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(60): 
            try:
                with urllib.request.urlopen(health_url) as response:
                    if response.getcode() == 200:
                        print("Server is ready!")
                        return self
            except Exception:
                time.sleep(1)
                
        self.stop()
        raise RuntimeError("Server failed to start. Check if the model is corrupted or if the server crashed.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if self.process:
            print("Shutting down server...")
            self.process.terminate()
            self.process.wait()
            self.process = None

    def ask(self, prompt, max_tokens=2048):
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        req = urllib.request.Request(
            url, data=json.dumps(data).encode('utf-8'), headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']