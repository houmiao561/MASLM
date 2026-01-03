import asyncio
from fastmcp import Client
from fastmcp.client import NodeStdioTransport

async def main():
    # 启动 MCP Server（Node）
    client = Client(NodeStdioTransport("/Users/houmiao/Desktop/MASLM/web-search-mcp/dist/index.js"))

    async with client:
        # 列出 server 提供的工具
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print("-", t.name)

        # 调用搜索工具（名称以 list_tools 为准）
        result = await client.call_tool(
            name="get-web-search-summaries",
            arguments={
                "query": "什么是人工智能",
                "limit": 3,
                "includeContent": False
            }
        )

        print("\nSearch result:")
        print(result)

if __name__ == "__main__":
    # 启动 MCP Server（Node）   
    
    asyncio.run(main())


# content=[
#     TextContent(
#         type='text', 
#         text='Search completed for "FastMCP MCP protocol" with 3 results:\n\n**Status:** Search engine: Browser Brave; 3 result requested/3 obtained; PDF: 0; 3 followed\n\n**1. GitHub github.com › jlowin  › fastmcp   GitHub - jlowin/fastmcp: 🚀 The fast, Pythonic way to build MCP servers and clients**\nURL: https://github.com/jlowin/fastmcp\nDescription: No description available\n\n---\n\n**2. FastMCP gofastmcp.com › getting-started  › welcome   Welcome to FastMCP 2.0! - FastMCP**\nURL: https://gofastmcp.com/getting-started/welcome\nDescription: No description available\n\n---\n\n**3. DataCamp datacamp.com › tutorial  › building-mcp-server-client-fastmcp   Building an MCP Server and Client with FastMCP 2.0 | DataCamp**\nURL: https://www.datacamp.com/tutorial/building-mcp-server-client-fastmcp\nDescription: No description available\n\n---\n\n', 
#         annotations=None, 
#         meta=None)
#         ], 
# structured_content=None, 
# meta=None, 
# data=None, 
# is_error=False