import os
import re
import subprocess

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"

def run_benchmark(library, code):
    """
    Write the provided code to a temporary file, run it via subprocess,
    print the output, remove the file, and parse the average download time.
    """
    script_filename = f"benchmark_{library}.py"
    with open(script_filename, "w") as script_file:
        script_file.write(code)
    result = subprocess.run(["python", script_filename], capture_output=True, text=True)
    print(result.stdout)  # For transparency
    os.remove(script_filename)
    
    # Parse the average time from a line like:
    # "<library> average download time: 0.1466 seconds"
    lines = result.stdout.strip().split("\n")
    duration = float("inf")
    for line in lines:
        match = re.search(r"average download time:\s+([0-9.]+)\s+seconds", line)
        if match:
            try:
                duration = float(match.group(1))
            except ValueError:
                duration = float("inf")
            break
    return duration

def benchmark_for_condition(num_runs):
    """
    For the given number of runs, build the benchmark code for each library,
    run the benchmarks, and return a dict mapping library names to the average time.
    """
    libraries_code = {
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
    durations = [download() for _ in range({num_runs})]
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
    durations = [download() for _ in range({num_runs})]
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
    async def main():
        durations = [await download() for _ in range({num_runs})]
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
    durations = [download() for _ in range({num_runs})]
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
    durations = [download() for _ in range({num_runs})]
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
    durations = [download() for _ in range({num_runs})]
    avg_duration = sum(durations) / len(durations)
    print(f"httpcore average download time: {{avg_duration:.4f}} seconds")
'''
    }
    
    results = {}
    for lib, code in libraries_code.items():
        results[lib] = run_benchmark(lib, code)
    return results

if __name__ == "__main__":
    # Benchmark conditions: 1, 10, and 100 requests.
    conditions = [1, 10, 100]
    all_results = {}  # Map num_runs -> {library: avg_duration}
    for n in conditions:
        print(f"\nRunning benchmarks for {n} request(s)...")
        results = benchmark_for_condition(n)
        all_results[n] = results

    # Prepare data for plotting.
    # Get list of libraries (assume same ordering for all conditions)
    libraries = list(all_results[conditions[0]].keys())
    conditions_sorted = sorted(all_results.keys())  # [1, 10, 100]
    # Build a 2D array: rows = libraries, columns = conditions.
    data = []
    for lib in libraries:
        lib_times = [all_results[cond][lib] for cond in conditions_sorted]
        data.append(lib_times)
    
    # --- Import graphing libraries only after benchmarks are complete ---
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.array(data)  # shape: (num_libraries, num_conditions)
    num_libs = len(libraries)
    num_conds = len(conditions_sorted)
    bar_width = 0.2
    x = np.arange(num_libs)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cond in enumerate(conditions_sorted):
        ax.bar(x + i * bar_width, data[:, i], width=bar_width, label=f"{cond} request(s)")

    ax.set_xlabel("Library")
    ax.set_ylabel("Average Download Time (seconds)")
    ax.set_title("Benchmark Leaderboards for 1, 10, and 100 Requests")
    ax.set_xticks(x + bar_width * (num_conds - 1) / 2)
    ax.set_xticklabels(libraries)
    ax.legend()
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig("benchmark_leaderboards.png")
    print("Graph saved as 'benchmark_leaderboards.png'")
