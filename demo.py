import time

# URL to download
url = "https://docs.pola.rs/py-polars/html/objects.inv"

# Function to download using httpx
def download_httpx(url):
    import httpx  # Importing inside the function
    start_time = time.time()
    response = httpx.get(url)
    duration = time.time() - start_time
    return duration, response.content

# Function to download using pycurl
def download_pycurl(url):
    import pycurl  # Importing inside the function
    from io import BytesIO
    
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    
    start_time = time.time()
    c.perform()
    duration = time.time() - start_time
    c.close()
    
    return duration, buffer.getvalue()

# Benchmarking httpx
httpx_duration, httpx_content = download_httpx(url)
print(f"httpx download time: {httpx_duration:.4f} seconds")

# Benchmarking pycurl
pycurl_duration, pycurl_content = download_pycurl(url)
print(f"pycurl download time: {pycurl_duration:.4f} seconds")

# Optionally, you can check if the content is the same
if httpx_content == pycurl_content:
    print("Both downloads are identical.")
else:
    print("Downloads differ.")
