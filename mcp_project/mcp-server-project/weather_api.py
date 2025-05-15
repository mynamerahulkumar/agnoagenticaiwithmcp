import os
import httpx
from dotenv import load_dotenv

load_dotenv()

weather_api_key = os.getenv("WEATHER_API_KEY")

import httpx

async def get_weather(city:str)->dict:
    """Fetch current weather data for a city"""
    async with httpx.AsyncClient() as client:
        response= await client.get(f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}&aqi=no")
        return response.text
    

if __name__ == "__main__":
    import asyncio
    city = "New York"
    weather_data = asyncio.run(get_weather(city))
    print(weather_data)