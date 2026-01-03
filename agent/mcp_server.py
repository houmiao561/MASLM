from fastmcp import FastMCP

# 1. 创建 MCP Server
mcp = FastMCP(name="demo-mcp-server")

# 2. 定义一个工具（Tool）
@mcp.tool()
def web_search(query: str, top_k: int = 3) -> dict:
    """
    模拟一个搜索工具
    """
    # 真实场景中这里可以换成 requests / serpapi / bing 等
    results = [
        {"title": f"Result {i}", "content": f"About {query} #{i}"}
        for i in range(1, top_k + 1)
    ]
    return {
        "query": query,
        "results": results
    }
@mcp.tool()
def get_sum(a: int, b: int) -> str:
    """
    如果需要计算就用这个
    """
    print("这里调用了工具！！！！！！！！")
    return f"这是一个计算工具demo测试{a} + {b} = {a + b}"
# 3. 启动 MCP Server
if __name__ == "__main__":
    mcp.run()
