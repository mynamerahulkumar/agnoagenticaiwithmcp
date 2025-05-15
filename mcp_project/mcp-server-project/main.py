from mcp.server.fastmcp import FastMCP
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

weather_api_key = os.getenv("WEATHER_API_KEY")
serper_api_key =  os.getenv("SERPER_API_KEY")

# Create an MCP server
mcp = FastMCP("Serper Search")

NOTES_FILE = os.path.join(os.path.dirname(__file__), "result_notes.txt")

def ensure_file():
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as f:
            f.write("")

@mcp.tool()
def add_note(message: str) -> str:
    """
    Append a new note to the note file.

    Args:
        message (str): The note content to be added.

    Returns:
        str: Confirmation message indicating the note was saved.
    """
    ensure_file()
    with open(NOTES_FILE, "a") as f:
        f.write(message + "\n")
    return "Note saved!"

@mcp.tool()
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

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}&aqi=no")
        return response.text


@mcp.tool()
def read_notes() -> str:
    """
    Read and return all notes from the note file.

    Returns:
        str: All notes as a single string separated by line breaks.
             If no notes exist, a default message is returned.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        content = f.read().strip()
    return content or "No notes yet."

@mcp.resource("notes://latest")
def get_latest_note() -> str:
    """
    Get the most recently added note from the note file.

    Returns:
        str: The last note entry. If no notes exist, a default message is returned.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        lines = f.readlines()
    return lines[-1].strip() if lines else "No notes yet."

@mcp.prompt()
def note_summary_prompt() -> str:
    """
    Generate a prompt asking the AI to summarize all current notes.

    Returns:
        str: A prompt string that includes all notes and asks for a summary.
             If no notes exist, a message will be shown indicating that.
    """
    ensure_file()
    with open(NOTES_FILE, "r") as f:
        content = f.read().strip()
    if not content:
        return "There are no notes yet."

    return f"Summarize the current notes: {content}"