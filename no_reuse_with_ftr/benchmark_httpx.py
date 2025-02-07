
import time
import httpx
url = "https://docs.pola.rs/py-polars/html/objects.inv"

def download():
    start_time = time.time()
    with httpx.Client() as client:
        response = client.get(url)
    return time.time() - start_time

if __name__ == "__main__":
    durations = [download() for _ in range(1)]
    avg_duration = sum(durations) / len(durations)
    print(f"httpx average download time: {avg_duration:.4f} seconds")
