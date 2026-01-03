import asyncio
from fastmcp import Client
from openai import AsyncOpenAI

# async def run():
#     client = Client("agent/mcp_server.py")
#     async with client:
#         tools = await client.list_tools()
#         tool = tools[1]
#         tools_result = await client.call_tool(tool.name, {"a": 1, "b": 2})
#         print(tools_result)


# if __name__ == "__main__":
#     asyncio.run(run())

#     os.environ["OPENAI_API_KEY"] = "sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl"
    # os.environ["OPENAI_API_BASE_URL"] = "https://api.siliconflow.cn/v1"

class UserClient:
    def __init__(self,script: str):
        self.model = "deepseek-ai/DeepSeek-V3"
        self.script = script
        self.openai_client = AsyncOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl"
        )
        
    async def prepare_tools(self, mcp_client):
        tools = await mcp_client.list_tools()
        tools = [
            {
                "type":"function",
                "function":{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools
        ]
        return tools
    
    async def chat(self):
        async with Client(self.script) as mcp_client:
            tools = await self.prepare_tools(mcp_client)
            messages = [
                {
                    "role": "user",
                    "content": "Hello,请帮我计算一下1283979+78235等于多少，并且必须使用mcp工具"
                }
            ]
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools
            )
            print(f"Response: \n{response}")  # 打印响应response)
    
    
async def main():
    user_client = UserClient("agent/mcp_server.py")
    await user_client.chat()


if __name__ == "__main__":
    asyncio.run(main())





"""
[Tool(
    name='web_search', 
    title=None, 
    description='模拟一个搜索工具', 
    inputSchema={
        'properties': {
            'query': {'type': 'string'}, 
            'top_k': {'default': 3, 'type': 'integer'}
        }, 
        'required': ['query'], 
        'type': 'object'
    }, 
    outputSchema={
        'additionalProperties': True, 
        'type': 'object'
    }, 
    icons=None, 
    annotations=None, 
    meta={'_fastmcp': {'tags': []}}, 
    execution=None)
]
"""