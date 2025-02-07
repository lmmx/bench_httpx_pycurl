import subprocess
import time

def download_curl(url):
    start_time = time.time()
    result = subprocess.run(['curl', '-O', url], capture_output=True)
    duration = time.time() - start_time
    return duration, result.stdout

url = "https://docs.pola.rs/py-polars/html/objects.inv"
duration, content = download_curl(url)
print(f"curl download time: {duration:.4f} seconds")
