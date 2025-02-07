import aiohttp
import asyncio
import time

async def download_aiohttp(url):
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.get(url) as response:
            content = await response.read()
            duration = time.time() - start_time
            return duration, content

async def main():
    url = "https://docs.pola.rs/py-polars/html/objects.inv"
    duration, content = await download_aiohttp(url)
    print(f"aiohttp download time: {duration:.4f} seconds")

asyncio.run(main())
