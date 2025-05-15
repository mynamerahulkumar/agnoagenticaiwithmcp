import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools, MultiMCPTools
import os
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# This is the URL of the MCP server we want to use.
server_url = "http://localhost:8000/sse"


async def run_agent(message: str) -> None:
    async with MCPTools(transport="sse", url=server_url) as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)


# Using MultiMCPTools, we can connect to multiple MCP servers at once, even if they use different transports.
# In this example we connect to both our example server (SSE transport), and a different server (stdio transport).
async def run_agent_with_multimcp(message: str) -> None:
    async with MultiMCPTools(
        commands=["npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt"],
        urls=[server_url],
    ) as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)


if __name__ == "__main__":
    asyncio.run(run_agent("Do we have parents anniversary this week?"))
    asyncio.run(
        run_agent_with_multimcp(
            "Can you check when is my parent's anniversary, and if there are any AirBnb listings in SF for two people for that day?"
        )
    )