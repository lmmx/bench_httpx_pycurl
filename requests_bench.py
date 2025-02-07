import requests
from concurrent.futures import ThreadPoolExecutor
import time

def download_requests(url):
    start_time = time.time()
    response = requests.get(url)
    duration = time.time() - start_time
    return duration, response.content

def main():
    url = "https://docs.pola.rs/py-polars/html/objects.inv"
    with ThreadPoolExecutor() as executor:
        future = executor.submit(download_requests, url)
        duration, content = future.result()
        print(f"requests download time: {duration:.4f} seconds")

main()
