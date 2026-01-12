# mcp_client.py
import subprocess
import json
import threading
import queue
import time
from camel.toolkits import FunctionTool

class MCPClient:
    def __init__(self, cmd):
        # print("MCPClient init")
        # print("cmd:", cmd)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.responses = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self._init()

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
                self.responses.put(msg)
            except Exception:
                pass

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _init(self):
        self._send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "camel-client", "version": "0.1"}
            }
        })
        response = self.responses.get(timeout=5)
        # print(f"Initialize response: {response}")
        self._send({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
        time.sleep(0.5)

    def list_tools(self):
        self._send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        response = self.responses.get(timeout=5)
        # print(f"list_tools response: {response}")
        return response["result"]["tools"]

    def call_tool(self, name, arguments):
        self._send({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        })
        return self.responses.get(timeout=10)["result"]



def make_mcp_tool(tool_def, mcp_client):
    tool_name = tool_def["name"]
    schema = tool_def["inputSchema"]

    # Python 层的 wrapper（名字可以随便，但必须合法）
    def _mcp_tool_wrapper(**kwargs):
        return mcp_client.call_tool(tool_name, kwargs)

    return FunctionTool(
        func=_mcp_tool_wrapper,
        openai_tool_schema={
            "type": "function",
            "function": {
                "name": tool_name,          # 这里仍然用 MCP tool name
                "description": tool_def.get("description", ""),
                "parameters": schema,
            },
        },
    )
