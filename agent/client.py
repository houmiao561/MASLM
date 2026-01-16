import requests
import uuid
from typing import Any, Dict, Optional

class MCPClient:
    def __init__(self, server_url: str, api_key: str = ""):
        self.server_url = server_url
        self.api_key = api_key

    # ===== MCP JSON-RPC 调用封装 =====
    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method, # 一般情况下是 "tools/call"
            "params": params or {} 
            # 一般情况下是 
            # { 
            #   "name": "resolve-library-id", 
            #   "arguments": {
            #       "query": "numpy.compare_chararrays", 
            #       "libraryName": "numpy"
            #   }
            # }
            # 或者是
            # { 
            #   "name": "query-docs", 
            #   "arguments": {
            #       "libraryId": "numpy", 
            #       "query": "numpy.compare_chararrays not available in numpy 2.0"
            #   }
            # }
            # 二选一
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key: # 有无 api_key 都可以运行
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            self.server_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    mcp_client123 = MCPClient(
        server_url="https://mcp.context7.com/mcp",
        api_key=""   # 可空
    )
    # print("list tools::::::::\n",mcp_client123.call("tools/list"))

    # result = mcp_resolve_library_id("numpy.compare_chararrays", "numpy", mcp_client123)
    # print(result["result"]["content"][0].get("text"))
    # result_1 = mcp_query_docs("/numpy/numpy", "numpy.compare_chararrays not available in numpy 2.0", mcp_client123)
    # print(result_1["result"]["content"][0].get("text"))
    print(mcp_client123.call("tools/call", {"name": "resolve-library-id", "arguments": {"query": "numpy", "libraryName": "numpy"}})) 
    # print(mcp_client123.call("tools/call", {"name": "query-docs", "arguments": {"libraryId": "/numpy/numpy", "query": "numpy.compare_chararrays not available in numpy 2.0"}}))