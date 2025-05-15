from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

agent=Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Tech Financial expert!",
    instructions=[
        "Search your knowledge base for Amazon company investment and revenue.",
        "If the question is better suited for the web, search the web to fill in gaps.Just directly search web don't ask for permission",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
         urls=["https://s2.q4cdn.com/299287126/files/doc_financials/2024/ar/Amazon-com-Inc-2023-Shareholder-Letter.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True,
    )
)

if agent.knowledge is not None:
    agent.knowledge.load()
    
agent.print_response("What is revenue distribution of Amazon in 2023",stream=True)


