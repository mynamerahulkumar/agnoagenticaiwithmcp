{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'mcp'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01masyncio\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mmcp\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ClientSession, StdioServerParameters\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mmcp\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mclient\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mstdio\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m stdio_client\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mlangchain_mcp_adapters\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mtools\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_mcp_tools\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'mcp'"
     ]
    }
   ],
   "source": [
    "import asyncio\n",
    "from mcp import ClientSession, StdioServerParameters\n",
    "from mcp.client.stdio import stdio_client\n",
    "from langchain_mcp_adapters.tools import load_mcp_tools\n",
    "from langgraph.prebuilt import create_react_agent\n",
    "from langchain_ollama import ChatOllama\n",
    "from langchain_ibm import ChatWatsonx\n",
    "#Using Ollama\n",
    "model = ChatOllama(model=\"llama3.2\")  # Use a model available via Ollama\n",
    "#Using WatsonX\n",
    "parameters = {\n",
    "    \"decoding_method\": \"sample\",\n",
    "    \"max_new_tokens\": 4000,\n",
    "    \"min_new_tokens\": 1,\n",
    "    \"temperature\": 1,\n",
    "    \"top_k\": 50,\n",
    "    \"top_p\": 1,\n",
    "}\n",
    "\n",
    "watsonx_llm = ChatWatsonx(\n",
    "    model_id = \"ibm/granite-3-8b-instruct\",\n",
    "    url=\"\",\n",
    "    project_id=\"\",\n",
    "    apikey=\"\",\n",
    "    params=parameters,\n",
    ")\n",
    "async def main():\n",
    "    server_params = StdioServerParameters(\n",
    "        command=\"python\",\n",
    "        args=[\"string_tools_server.py\"],  # Update this path\n",
    "    )\n",
    "    async with stdio_client(server_params) as (read, write):\n",
    "        async with ClientSession(read, write) as session:\n",
    "            await session.initialize()\n",
    "            tools = await load_mcp_tools(session)\n",
    "            #Use \"watsonx_llm\" instead of \"model\" to use WatsonX\n",
    "            agent = create_react_agent(model, tools)\n",
    "            # Try out the tools via natural language\n",
    "            msg1 = {\"messages\": \"Reverse the string 'hello world'\"}\n",
    "            msg2 = {\"messages\": \"How many words are in the sentence 'Model Context Protocol is powerful'?\"}\n",
    "            res1 = await agent.ainvoke(msg1)\n",
    "            # print(\"Reversed string result:\", res1)\n",
    "            for m in res1['messages']:\n",
    "                m.pretty_print()\n",
    "            res2 = await agent.ainvoke(msg2)\n",
    "            # print(\"Word count result:\", res2)\n",
    "            for m in res2['messages']:\n",
    "                m.pretty_print()\n",
    "if __name__ == \"__main__\":\n",
    "    asyncio.run(main())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
