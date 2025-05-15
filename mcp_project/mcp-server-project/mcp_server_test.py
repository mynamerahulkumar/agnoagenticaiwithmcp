from mcp.server.fastmcp import FastMCP

#CREATE MCP SERVER
mcp=FastMCP("CALCULATORMCP")

#ADD A TOOL

@mcp.tool()
def mul(a:int ,b:int)->int:
    """
    Multiply two integers
    """
    return a*b




