
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv
load_dotenv()

"""
This code uses the agno library to create a team of AI agents 
that can work together to answer your question about the market 
outlook and financial performance of NVIDIA and Tesla.
"""

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

web_agent=Agent(
    name="Web Agent",
    role="Search the web for the information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent=Agent(
    name="Finance Agent",
    role="Get Financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_info=True)],
    show_tool_calls=True,
    markdown=True,
)

agent_team=Agent(
    team=[web_agent,finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources","use tables to display data","Always use tools"],
    show_tool_calls=True,
    markdown=True,
)


# agent_team.print_response("analyze the market outlook and financial performance of AI semiconductor compnay NVDA and Microsoft and suggest wheather I have to invest or not ?",stream=True)

agent_team.print_response("what is recent development in AI companies like OpenAI?",stream=True)

