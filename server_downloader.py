import os
import json
import shutil
import zipfile
import urllib.request

#--------------| fetch and extract server |-------------
def ensure_server_exists(server_path):
    if os.path.exists(server_path):
        return 

    print("llama-server.exe not found! Downloading the latest release...")
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        
    target_asset = next((a for a in data['assets'] if "bin-win" in a['name'].lower() and "cpu-x64" in a['name'].lower()), None)
    if not target_asset:
        raise RuntimeError("Could not find a suitable Windows release on GitHub.")
        
    zip_filename = target_asset['name']
    print(f"Downloading {zip_filename}...")
    urllib.request.urlretrieve(target_asset['browser_download_url'], zip_filename)
    
    print(" Extracting binaries and libraries...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            filename = os.path.basename(file_info.filename)
            if not filename:
                continue
            if filename.endswith('.exe') or filename.endswith('.dll'):
                target_dir = os.path.dirname(os.path.abspath(server_path))
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, filename)
                with zip_ref.open(file_info) as source, open(target_file, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
    os.remove(zip_filename)
    print("Download complete.\n")