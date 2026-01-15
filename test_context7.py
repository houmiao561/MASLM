import requests
import json
import uuid
import sys

MCP_SERVER = "https://mcp.context7.com/mcp"
# API_KEY = "ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"  # ← 换成你自己的
API_KEY = ""


def mcp_call(method: str, params: dict | None = None):
    """
    最基础的 MCP JSON-RPC 调用封装
    """
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {}
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {API_KEY}",
    }

    resp = requests.post(
        MCP_SERVER,
        headers=headers,
        json=payload, 
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def list_tools():
    """
    调用 MCP 的 tools/list
    """
    result = mcp_call("tools/list")
    print(result)
    sys.exit()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result["result"]["tools"]


if __name__ == "__main__":
    tools = list_tools()

    print(f"Found {len(tools)} tools:\n")

    for tool in tools:
        print("=" * 60)
        print(f"Name        : {tool['name']}")
        print(f"Description : {tool.get('description', '')}")

        print("Input Schema:")
        print(json.dumps(tool.get("inputSchema", {}), indent=2, ensure_ascii=False))


"""
RESULT: 已经保存在 garbage.json 中


Found 2 tools:

============================================================
Name        : resolve-library-id
Description : Resolves a package/product name to a Context7-compatible library ID and returns matching libraries.

You MUST call this function before 'query-docs' to obtain a valid Context7-compatible library ID UNLESS the user explicitly provides a library ID in the format '/org/project' or '/org/project/version' in their query.

Selection Process:
1. Analyze the query to understand what library/package the user is looking for
2. Return the most relevant match based on:
- Name similarity to the query (exact matches prioritized)
- Description relevance to the query's intent
- Documentation coverage (prioritize libraries with higher Code Snippet counts)
- Source reputation (consider libraries with High or Medium reputation more authoritative)
- Benchmark Score: Quality indicator (100 is the highest score)

Response Format:
- Return the selected library ID in a clearly marked section
- Provide a brief explanation for why this library was chosen
- If multiple good matches exist, acknowledge this but proceed with the most relevant one
- If no good matches exist, clearly state this and suggest query refinements

For ambiguous queries, request clarification before proceeding with a best-guess match.

IMPORTANT: Do not call this tool more than 3 times per question. If you cannot find what you need after 3 calls, use the best result you have.
Input Schema:
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The user's original question or task. This is used to rank library results by relevance to what the user is trying to accomplish. IMPORTANT: Do not include any sensitive or confidential information such as API keys, passwords, credentials, or personal data in your query."
    },
    "libraryName": {
      "type": "string",
      "description": "Library name to search for and retrieve a Context7-compatible library ID."
    }
  },
  "required": [
    "query",
    "libraryName"
  ],
  "additionalProperties": false,
  "$schema": "http://json-schema.org/draft-07/schema#"
}
============================================================
Name        : query-docs
Description : Retrieves and queries up-to-date documentation and code examples from Context7 for any programming library or framework.

You must call 'resolve-library-id' first to obtain the exact Context7-compatible library ID required to use this tool, UNLESS the user explicitly provides a library ID in the format '/org/project' or '/org/project/version' in their query.

IMPORTANT: Do not call this tool more than 3 times per question. If you cannot find what you need after 3 calls, use the best information you have.
Input Schema:
{
  "type": "object",
  "properties": {
    "libraryId": {
      "type": "string",
      "description": "Exact Context7-compatible library ID (e.g., '/mongodb/docs', '/vercel/next.js', '/supabase/supabase', '/vercel/next.js/v14.3.0-canary.87') retrieved from 'resolve-library-id' or directly from user query in the format '/org/project' or '/org/project/version'."
    },
    "query": {
      "type": "string",
      "description": "The question or task you need help with. Be specific and include relevant details. Good: 'How to set up authentication with JWT in Express.js' or 'React useEffect cleanup function examples'. Bad: 'auth' or 'hooks'. IMPORTANT: Do not include any sensitive or confidential information such as API keys, passwords, credentials, or personal data in your query."
    }
  },
  "required": [
    "libraryId",
    "query"
  ],
  "additionalProperties": false,
  "$schema": "http://json-schema.org/draft-07/schema#"
}

"""