import time
import pycurl
from io import BytesIO

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"
num_runs = 5  # Number of times to perform the download

def download():
    """
    Download the content at the given URL using pycurl,
    and return the time taken for the download.
    """
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    
    # Start the timer, perform the download, and then stop the timer.
    start_time = time.time()
    c.perform()
    duration = time.time() - start_time
    
    c.close()
    return duration

if __name__ == "__main__":
    durations = []
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")
        duration = download()
        durations.append(duration)
        print(f"Duration: {duration:.4f} seconds\n")
        
    avg_duration = sum(durations) / len(durations)
    print(f"pycurl average download time over {num_runs} run(s): {avg_duration:.4f} seconds")
