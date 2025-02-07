import os
import re
import subprocess

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"
num_runs = 1  # Number of times to run each test

# We'll store the results in this list so that we can sort and print them afterwards
benchmark_results = []

def run_benchmark(library, code):
    """
    Run the benchmark for a specific library using a subprocess.
    Write the code to a temporary file, run it, display its output,
    then parse the output and store the results.
    """
    script_filename = f"benchmark_{library}.py"
    with open(script_filename, "w") as script_file:
        script_file.write(code)

    result = subprocess.run(["python", script_filename], capture_output=True, text=True)
    # Print the library output for transparency
    print(result.stdout)

    # Remove the temporary file
    os.remove(script_filename)

    # Parse the average time from the output (if any)
    # We look for lines like: "<library> average download time: 0.1466 seconds"
    # or an error line if something went wrong.
    lines = result.stdout.strip().split("\n")
    duration = float("inf")  # default in case of errors
    for line in lines:
        match = re.search(r"average download time:\s+([0-9.]+)\s+seconds", line)
        if match:
            duration_str = match.group(1)
            try:
                duration = float(duration_str)
            except ValueError:
                duration = float("inf")
            break

    # Store the (library, duration) pair
    benchmark_results.append((library, duration))

def benchmark():
    """
    Define all library benchmarks in a dictionary, run them,
    then sort and display the results with relative speed.
    """
    libraries = {
        "httpx": f'''
import time
import httpx

url = "{url}"

def download():
    start_time = time.time()
    response = httpx.get(url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"httpx average download time: {{avg_duration:.4f}} seconds")
''',
        "pycurl": f'''
import time
import pycurl
from io import BytesIO

url = "{url}"

def download():
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    
    start_time = time.time()
    c.perform()
    duration = time.time() - start_time
    c.close()
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"pycurl average download time: {{avg_duration:.4f}} seconds")
''',
        "aiohttp": f'''
import time
import aiohttp
import asyncio

url = "{url}"

async def download():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.get(url) as response:
            await response.read()
            duration = time.time() - start_time
            return duration

if __name__ == "__main__":
    durations = []

    async def main():
        for _ in range({num_runs}):
            durations.append(await download())
        avg_duration = sum(durations) / len(durations)
        print(f"aiohttp average download time: {{avg_duration:.4f}} seconds")
    
    asyncio.run(main())
''',
        "requests": f'''
import time
import requests

url = "{url}"

def download():
    start_time = time.time()
    response = requests.get(url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"requests average download time: {{avg_duration:.4f}} seconds")
''',
        "urllib3": f'''
import time
import urllib3

url = "{url}"

def download():
    http = urllib3.PoolManager()
    start_time = time.time()
    response = http.request('GET', url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"urllib3 average download time: {{avg_duration:.4f}} seconds")
''',
        "httpcore": f'''
import time
import httpcore
from urllib.parse import urlparse

url = "{url}"

def download():
    try:
        parsed = urlparse(url)
        # Build the full URL as a string
        full_url = parsed.scheme + "://" + parsed.netloc + parsed.path
        if parsed.query:
            full_url += "?" + parsed.query

        start_time = time.time()
        with httpcore.ConnectionPool() as http:
            response = http.request("GET", full_url)
            body = response.read()
            response.close()

        duration = time.time() - start_time
        return duration
    except Exception as e:
        print(f"Error in httpcore download: {{e}}")
        return float('inf')

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    if all(isinstance(d, float) for d in durations):
        print(f"httpcore average download time: {{avg_duration:.4f}} seconds")
    else:
        print("httpcore encountered errors during download.")
'''
    }

    # Run benchmarks for each library
    for library, code in libraries.items():
        run_benchmark(library, code)

    # Sort results based on duration
    sorted_results = sorted(benchmark_results, key=lambda x: x[1])

    # Print sorted results, showing the fastest library's time and
    # a ratio for the rest.
    fastest_time = sorted_results[0][1]
    print("\nBenchmark Results (sorted fastest to slowest):")
    for i, (lib, duration) in enumerate(sorted_results, start=1):
        if i == 1:
            # Fastest: print its time and label as (1x)
            print(f"{i}. {lib} (1x) = {duration:.3f}s")
        else:
            if fastest_time == 0:
                ratio = float('inf')
            else:
                ratio = duration / fastest_time
            print(f"{i}. {lib} ({ratio:.1f}x)")

if __name__ == "__main__":
    benchmark()
