import os
import subprocess

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"
num_runs = 5  # Number of times to run each test


def run_benchmark(library, code):
    """Run the benchmark for a specific library using a subprocess."""
    # Write the script to a temporary file
    script_filename = f"benchmark_{library}.py"
    with open(script_filename, "w") as script_file:
        script_file.write(code)

    # Run the script in a subprocess
    result = subprocess.run(["python", script_filename], capture_output=True, text=True)

    # Print the output from the subprocess
    print(result.stdout)

    # Clean up the temporary script file
    os.remove(script_filename)


def benchmark():
    libraries = {
        "httpx": """
import time
import httpx

url = '"""
        + url
        + """'

def download():
    start_time = time.time()
    response = httpx.get(url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range("""
        + str(num_runs)
        + """):
        duration = download()
        durations.append(duration)
    avg_duration = sum(durations) / len(durations)
    print(f"httpx average download time: {avg_duration:.4f} seconds")
""",
        "pycurl": """
import time
import pycurl
from io import BytesIO

url = '"""
        + url
        + """'

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
    for _ in range("""
        + str(num_runs)
        + """):
        duration = download()
        durations.append(duration)
    avg_duration = sum(durations) / len(durations)
    print(f"pycurl average download time: {avg_duration:.4f} seconds")
""",
        "aiohttp": """
import time
import aiohttp
import asyncio

url = '"""
        + url
        + """'

async def download():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.get(url) as response:
            await response.read()  # Read the response
            duration = time.time() - start_time
            return duration

if __name__ == "__main__":
    durations = []
    async def main():
        for _ in range("""
        + str(num_runs)
        + """):
            duration = await download()
            durations.append(duration)
        avg_duration = sum(durations) / len(durations)
        print(f"aiohttp average download time: {avg_duration:.4f} seconds")
    
    asyncio.run(main())
""",
        "requests": """
import time
import requests

url = '"""
        + url
        + """'

def download():
    start_time = time.time()
    response = requests.get(url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range("""
        + str(num_runs)
        + """):
        duration = download()
        durations.append(duration)
    avg_duration = sum(durations) / len(durations)
    print(f"requests average download time: {avg_duration:.4f} seconds")
""",
        "urllib3": """
import time
import urllib3

url = '"""
        + url
        + """'

def download():
    http = urllib3.PoolManager()
    start_time = time.time()
    response = http.request('GET', url)
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    durations = []
    for _ in range("""
        + str(num_runs)
        + """):
        duration = download()
        durations.append(duration)
    avg_duration = sum(durations) / len(durations)
    print(f"urllib3 average download time: {avg_duration:.4f} seconds")
""",
    }

    for library, code in libraries.items():
        run_benchmark(library, code)


# Run the benchmark
if __name__ == "__main__":
    benchmark()
