import httpx
from bs4 import BeautifulSoup

# Define the URL for a specific ticker
ticker = "NVDA"
url = f"https://finance.yahoo.com/quote/{ticker}"

# Send a GET request using httpx with headers
headers = {"User-Agent": "Mozilla/5.0"}

response = httpx.get(url, headers=headers, timeout=10)

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Extract the stock price
price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
if price_tag:
    price = price_tag.text
    print(f"{ticker} Current Price: {price}")
else:
    print(f"Could not find stock price for {ticker}")
