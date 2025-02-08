import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm

# =============================================================================
# Global configuration
# =============================================================================

# Number of requests for each run. E.g. 1 => we do the block of GET requests once,
# 10 => repeat it 10 times in a row, etc.
CONDITIONS = [1, 10, 100]

# Adaptive sampling parameters
TOLERANCE = 0.01  # How precise we want each library’s mean estimate
CONFIDENCE = 0.95  # Confidence level for confidence intervals
MIN_REPEATS = 2  # 30  # Minimum repeats before we check if we can stop
MAX_REPEATS = 2  # 50  # Maximum repeats to avoid infinite loops

# Our set of URLs (each run will fetch them all)
URLS = [
    "https://docs.pola.rs/py-polars/html/objects.inv",
    "https://pandas.pydata.org/docs/objects.inv",
    "https://numpy.org/doc/stable/objects.inv",
    "https://pytorch.org/docs/stable/objects.inv",
    "https://docs.pydantic.dev/latest/",
]

# =============================================================================
# Benchmarking logic
# =============================================================================


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


def adaptive_benchmark(library, code, tolerance, confidence, min_repeats, max_repeats):
    """
    Run the benchmark repeatedly until the CI half-width of the mean
    is below the given tolerance or we hit max_repeats.

    Returns:
        measurements (list[float]): All measured durations
        mean (float): The mean of the measurements
        half_width (float): The half-width of the (confidence) CI
    """
    measurements = []

    # 1) Run an initial batch
    for _ in tqdm(range(min_repeats), desc=f"[{library}] Initial repeats"):
        measurements.append(run_benchmark(library, code))

    ds = DescrStatsW(measurements)
    ci_low, ci_upp = ds.tconfint_mean(alpha=1 - confidence)
    half_width = (ci_upp - ci_low) / 2.0

    # 2) Keep collecting data until half-width < tolerance or we run out of repeats
    with tqdm(
        total=max_repeats - min_repeats, desc=f"[{library}] Adaptive repeats"
    ) as pbar:
        while half_width > tolerance and len(measurements) < max_repeats:
            measurements.append(run_benchmark(library, code))
            pbar.update(1)
            ds = DescrStatsW(measurements)
            ci_low, ci_upp = ds.tconfint_mean(alpha=1 - confidence)
            half_width = (ci_upp - ci_low) / 2.0

    return measurements, ds.mean, half_width


def build_library_code(num_runs):
    """
    Return a dict mapping each library name to code that performs
    'num_runs' loops, and in each loop fetches ALL URLs from the URLS list.
    """
    # We embed the Python array of URLs in the code snippet
    urls_py = "[" + ", ".join(f'"{u}"' for u in URLS) + "]"

    return {
        "httpx": f"""
import time
import httpx

URLS = {urls_py}
client = httpx.Client()

def download():
    start_time = time.time()
    for url in URLS:
        resp = client.get(url)
        resp.read()
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"httpx average download time: {{avg_duration:.4f}} seconds")
    client.close()
""",
        "pycurl": f"""
import time
import pycurl
from io import BytesIO

URLS = {urls_py}
c = pycurl.Curl()

def download():
    start_time = time.time()
    for url in URLS:
        buffer = BytesIO()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, buffer)
        c.perform()
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"pycurl average download time: {{avg_duration:.4f}} seconds")
    c.close()
""",
        "aiohttp": f"""
import time
import aiohttp
import asyncio

URLS = {urls_py}

async def download(session):
    start_time = time.time()
    for url in URLS:
        async with session.get(url) as resp:
            await resp.read()
    return time.time() - start_time

async def main(num_runs):
    durations = []
    async with aiohttp.ClientSession() as session:
        for _ in range(num_runs):
            durations.append(await download(session))
    return durations

if __name__ == "__main__":
    durations = asyncio.run(main({num_runs}))
    avg_duration = sum(durations) / len(durations)
    print(f"aiohttp average download time: {{avg_duration:.4f}} seconds")
""",
        "requests": f"""
import time
import requests

URLS = {urls_py}
session = requests.Session()

def download():
    start_time = time.time()
    for url in URLS:
        resp = session.get(url)
        resp.content  # ensure it's read
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"requests average download time: {{avg_duration:.4f}} seconds")
    session.close()
""",
        "urllib3": f"""
import time
import urllib3

URLS = {urls_py}
http = urllib3.PoolManager()

def download():
    start_time = time.time()
    for url in URLS:
        response = http.request('GET', url)
        _ = response.data
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"urllib3 average download time: {{avg_duration:.4f}} seconds")
""",
        "httpcore": f"""
import time
import httpcore
from urllib.parse import urlparse

URLS = {urls_py}
pool = httpcore.ConnectionPool()

def download():
    start_time = time.time()
    for url in URLS:
        parsed = urlparse(url)
        full_url = parsed.scheme + "://" + parsed.netloc + parsed.path + (("?" + parsed.query) if parsed.query else "")
        response = pool.request("GET", full_url)
        _ = response.read()
        response.close()
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"httpcore average download time: {{avg_duration:.4f}} seconds")
    pool.close()
""",
        "hyperfastparallel": f"""
import time
import hyperfastparallel

URLS = {urls_py}

def download():
    start_time = time.time()
    hyperfastparallel.download_all(URLS)
    return time.time() - start_time

if __name__ == "__main__":
    durations = []
    for _ in range({num_runs}):
        durations.append(download())
    avg_duration = sum(durations) / len(durations)
    print(f"hyperfastparallel average download time: {{avg_duration:.4f}} seconds")
""",
    }


if __name__ == "__main__":
    all_results = {}  # condition -> {library -> list[float]}

    for cond in CONDITIONS:
        print(f"\n=== Running adaptive benchmarks for {cond} run(s) ===")
        libs_code = build_library_code(cond)
        all_results[cond] = {}

        for lib, code in libs_code.items():
            print(
                f"\n--- Starting adaptive benchmark for {lib} (each run = {cond} block(s) of all URLs) ---"
            )
            measurements, mean, half_width = adaptive_benchmark(
                library=lib,
                code=code,
                tolerance=TOLERANCE,
                confidence=CONFIDENCE,
                min_repeats=MIN_REPEATS,
                max_repeats=MAX_REPEATS,
            )
            all_results[cond][lib] = measurements
            print(f"[{lib}] final mean: {mean:.4f} s | half-width ~ {half_width:.4f} s")

    # -------------------------------------------------------------------------
    # Compute final descriptive statistics for each library & condition.
    # -------------------------------------------------------------------------
    cond_stats = {}
    for cond in CONDITIONS:
        cond_stats[cond] = {}
        for lib, measurements in all_results[cond].items():
            ds = DescrStatsW(measurements)
            mean = ds.mean
            ci_low, ci_upp = ds.tconfint_mean(alpha=1 - CONFIDENCE)
            error = (ci_upp - ci_low) / 2.0
            cond_stats[cond][lib] = (mean, error)

    # Sort libraries by speed under the 1-run condition
    sorted_libraries = sorted(
        cond_stats[1].keys(), key=lambda lib: cond_stats[1][lib][0]
    )

    # Build arrays for means/errors
    means_arr = []
    errs_arr = []
    for cond in CONDITIONS:
        means = [cond_stats[cond][lib][0] for lib in sorted_libraries]
        errs = [cond_stats[cond][lib][1] for lib in sorted_libraries]
        means_arr.append(means)
        errs_arr.append(errs)
    means_arr = np.array(means_arr).T
    errs_arr = np.array(errs_arr).T

    # Create a Polars DataFrame of the final means
    data_list = []
    for i, lib in enumerate(sorted_libraries):
        row = {"library": lib}
        for j, cond in enumerate(CONDITIONS):
            row[f"{cond}_mean"] = means_arr[i, j]
        data_list.append(row)
    df_means = pl.DataFrame(data_list)
    print("\n===== Sorted benchmark means (in seconds) =====")
    print(df_means)

    # -------------------------------------------------------------------------
    # Pairwise significance tests (Welch's t-test) for each condition
    # -------------------------------------------------------------------------
    sig_results_list = []

    for cond in CONDITIONS:
        libs = list(all_results[cond].keys())
        libs.sort()

        for i in range(len(libs)):
            for j in range(i + 1, len(libs)):
                libA = libs[i]
                libB = libs[j]
                t_stat, p_val = stats.ttest_ind(
                    all_results[cond][libA], all_results[cond][libB], equal_var=False
                )
                sig = p_val < 0.05
                sig_results_list.append(
                    {
                        "condition": cond,
                        "libraryA": libA,
                        "libraryB": libB,
                        "p_value": p_val,
                        "significant": sig,
                    }
                )

    sig_df = pl.DataFrame(sig_results_list)
    print("\n===== Pairwise Significance Results (Welch's t-test) =====")
    print(sig_df.sort(["condition", "p_value"]))

    # -------------------------------------------------------------------------
    # Plot a grouped bar chart showing means & confidence intervals
    # -------------------------------------------------------------------------
    num_libs = len(sorted_libraries)
    num_conds = len(CONDITIONS)
    bar_width = 0.2
    x = np.arange(num_libs)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cond in enumerate(CONDITIONS):
        ax.bar(
            x + i * bar_width,
            means_arr[:, i],
            width=bar_width,
            yerr=errs_arr[:, i],
            capsize=5,
            label=f"{cond} run(s)",
        )

    ax.set_xlabel("Library")
    ax.set_ylabel("Average Download Time (seconds)")
    ax.set_title("Adaptive Benchmark Leaderboards (Multiple URLs Per Run)")
    ax.set_xticks(x + bar_width * (num_conds - 1) / 2)
    ax.set_xticklabels(sorted_libraries)
    ax.legend()
    plt.tight_layout()

    plt.savefig("benchmark_leaderboards.png")
    print("\nGraph saved as 'benchmark_leaderboards.png'")
