import time
import httpx
import pycurl
import aiohttp
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import asyncio

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"
num_runs = 5  # Number of times to run each test

# Function to download using httpx
def download_httpx(url):
    start_time = time.time()
    response = httpx.get(url)
    duration = time.time() - start_time
    return duration

# Function to download using pycurl
def download_pycurl(url):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    
    start_time = time.time()
    c.perform()
    duration = time.time() - start_time
    c.close()
    
    return duration

# Function to download using aiohttp
async def download_aiohttp(url):
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.get(url) as response:
            await response.read()  # Read the response
            duration = time.time() - start_time
            return duration

# Function to download using requests with threading
def download_requests(url):
    start_time = time.time()
    response = requests.get(url)
    duration = time.time() - start_time
    return duration

# Function to run aiohttp in an event loop
def run_aiohttp(url):
    return asyncio.run(download_aiohttp(url))

# Benchmarking function
def benchmark():
    results = {
        "httpx": [],
        "pycurl": [],
        "aiohttp": [],
        "requests": []
    }

    # Run httpx
    for _ in range(num_runs):
        duration = download_httpx(url)
        results["httpx"].append(duration)

    # Run pycurl
    for _ in range(num_runs):
        duration = download_pycurl(url)
        results["pycurl"].append(duration)

    # Run aiohttp
    for _ in range(num_runs):
        duration = run_aiohttp(url)
        results["aiohttp"].append(duration)

    # Run requests with threading
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_requests, url) for _ in range(num_runs)]
        for future in futures:
            duration = future.result()
            results["requests"].append(duration)

    # Print results
    for library, durations in results.items():
        avg_duration = sum(durations) / len(durations)
        print(f"{library} average download time: {avg_duration:.4f} seconds")

# Run the benchmark
if __name__ == "__main__":
    benchmark()
