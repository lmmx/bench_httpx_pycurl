import os
import re
import subprocess

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"

# --- Benchmark functions ---
def run_benchmark(library, code):
    """
    Write the provided code to a temporary file, run it via subprocess,
    print the output, remove the file, and return the measured average time.
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

def benchmark_repeats(num_runs, repeats):
    """
    For a given condition (num_runs requests), repeat the test 'repeats' times.
    Returns a dict mapping each library to a list of measured average times.
    """
    # Code for each library benchmark (using the same code as before)
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
        results[lib] = []
        for i in range(repeats):
            res = run_benchmark(lib, code)
            results[lib].append(res)
    return results

# --- Main benchmarking and plotting section ---
if __name__ == "__main__":
    # Define the conditions (number of requests) and number of repeats for each condition.
    conditions = [1, 10, 100]
    repeats = 30  # number of repeated measurements per condition
    all_results = {}  # mapping: condition -> {library: list of measurements}
    for cond in conditions:
        print(f"\nRunning benchmarks for {cond} request(s), {repeats} repeats...")
        all_results[cond] = benchmark_repeats(cond, repeats)
    
    # Now compute statistics (mean and 95% confidence interval error) for each library under each condition.
    from statsmodels.stats.weightstats import DescrStatsW
    cond_stats = {}  # mapping: condition -> {library: (mean, error)}
    for cond in conditions:
        cond_stats[cond] = {}
        for lib, measurements in all_results[cond].items():
            ds = DescrStatsW(measurements)
            mean = ds.mean
            ci_low, ci_upp = ds.tconfint_mean()  # 95% CI by default
            error = (ci_upp - ci_low) / 2.0  # half-width of CI
            cond_stats[cond][lib] = (mean, error)
    
    # Use the 1-request condition to sort libraries (fastest first)
    sorted_libraries = sorted(cond_stats[1].keys(), key=lambda lib: cond_stats[1][lib][0])
    
    # Build arrays for means and errors for each condition in sorted order.
    import numpy as np
    means_arr = []
    errs_arr = []
    for cond in conditions:
        means = [cond_stats[cond][lib][0] for lib in sorted_libraries]
        errs = [cond_stats[cond][lib][1] for lib in sorted_libraries]
        means_arr.append(means)
        errs_arr.append(errs)
    # Convert arrays so that rows = libraries, columns = conditions.
    means_arr = np.array(means_arr).T  # shape: (num_libraries, num_conditions)
    errs_arr = np.array(errs_arr).T      # same shape

    # --- Import graphing libraries only after benchmarks are complete ---
    import polars as pl
    import matplotlib.pyplot as plt

    # For record, build a Polars DataFrame of the means.
    data_list = []
    for i, lib in enumerate(sorted_libraries):
        row = {"library": lib}
        for j, cond in enumerate(conditions):
            row[str(cond)] = means_arr[i, j]
        data_list.append(row)
    df = pl.DataFrame(data_list)
    print("Sorted benchmark means (in seconds):")
    print(df)

    # Plot a grouped bar chart with error bars.
    num_libs = len(sorted_libraries)
    num_conds = len(conditions)
    bar_width = 0.2
    x = np.arange(num_libs)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cond in enumerate(conditions):
        ax.bar(x + i * bar_width, means_arr[:, i], width=bar_width,
               yerr=errs_arr[:, i], capsize=5, label=f"{cond} request(s)")

    ax.set_xlabel("Library")
    ax.set_ylabel("Average Download Time (seconds)")
    ax.set_title("Benchmark Leaderboards for 1, 10, and 100 Requests\n(Fastest on the Left)")
    ax.set_xticks(x + bar_width * (num_conds - 1) / 2)
    ax.set_xticklabels(sorted_libraries)
    ax.legend()
    plt.tight_layout()
    
    # Save the figure to a file.
    plt.savefig("benchmark_leaderboards.png")
    print("Graph saved as 'benchmark_leaderboards.png'")
