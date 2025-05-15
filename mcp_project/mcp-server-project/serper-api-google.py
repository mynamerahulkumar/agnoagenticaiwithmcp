import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

async def google_serper_search_results(query: str) -> str:
    """Fetch search results from Yahoo using Serper.dev and httpx"""
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload
        )
        return response.text


if __name__ == "__main__":
    import asyncio

    async def main():
        search_data = await google_serper_search_results("Nvidia stock price data")
        print(search_data)

    asyncio.run(main())
