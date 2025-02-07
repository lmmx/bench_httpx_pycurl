import httpx
import asyncio
import time

async def download_httpx_async(url):
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        response = await client.get(url)
        duration = time.time() - start_time
        return duration, response.content

async def main():
    url = "https://docs.pola.rs/py-polars/html/objects.inv"
    duration, content = await download_httpx_async(url)
    print(f"httpx async download time: {duration:.4f} seconds")

asyncio.run(main())
