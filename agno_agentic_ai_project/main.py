from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

agent=Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an financial assistant,Please reply based on the query",
    tools=[DuckDuckGoTools()],
    markdown=True
)

agent.print_response("What is Nvidia currently investing on",stream=True)