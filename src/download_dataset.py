import os
import urllib.request
import time

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

BASE_URL = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/"
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'MachineLearningCVE')

def download_file(filename, url, out_path):
    print(f"Downloading {filename}...")
    start_time = time.time()
    
    # We use urllib to avoid external dependencies like requests
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
    elapsed = time.time() - start_time
    size_mb = len(data) / (1024 * 1024)
    print(f"  -> Saved {size_mb:.1f} MB in {elapsed:.1f} seconds.")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Target directory: {OUT_DIR}")
    
    for filename in FILES:
        out_path = os.path.join(OUT_DIR, filename)
        if os.path.exists(out_path):
            print(f"Skipping {filename} (already exists)")
            continue
            
        url = BASE_URL + filename + "?download=true"
        download_file(filename, url, out_path)

    print("\nAll dataset files downloaded successfully.")

if __name__ == '__main__':
    main()
