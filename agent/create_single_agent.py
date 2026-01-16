from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents.chat_agent import ChatAgent,FunctionTool
from typing import Optional, Dict, Any
from client import MCPClient
import os

def create_agent(
    system_prompt: str,
    api_key: Optional[str] = None,
    server_url: Optional[str] = None
):
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type="deepseek-ai/DeepSeek-V3",
        model_config_dict={
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    )
    mcp_client = MCPClient(
        server_url=server_url,
        api_key=api_key
    )

    # Tools
    def resolve_library_id(query: str, libraryName: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve a package or product name to a Context7-compatible library ID.

        Args:
            query: The symbol, module, or API name to resolve.
            libraryName: Optional library name hint.
        """

        arguments = {"query": query}
        if libraryName:
            arguments["libraryName"] = libraryName

        return mcp_client.call(
            method="tools/call",
            params={
                "name": "resolve-library-id",
                "arguments": arguments,
            }
        )
    resolve_library_id_tool = FunctionTool( func=resolve_library_id)
    
    # Tools
    def query_docs(libraryId: str, query: str) -> Dict[str, Any]:
        """
        Query up-to-date documentation from Context7.

        Args:
            libraryId: Context7-compatible library ID (must be resolved first).
            query: Documentation or API question.
        """
        return mcp_client.call(
            method="tools/call",
            params={
                "name": "query-docs",
                "arguments": {
                    "libraryId": libraryId,
                    "query": query,
                },
            }
        )
    query_docs_tool = FunctionTool(func=query_docs)
    
    agent = ChatAgent(
        system_message=system_prompt,
        model=model,
        tools=[
            resolve_library_id_tool,
            query_docs_tool,
        ]
    )
    return agent



if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "sk-rttlzkrvwxmfnolcmadlkeczxxnkmwolfprvyfnfwpfursjl"
    os.environ["OPENAI_API_BASE_URL"] = "https://api.siliconflow.cn/v1"
    
    SYSTEM_PROMPT = """
You are an AI assistant that uses two tools for Context7 documentation lookup:

Tool: resolve-library-id
- Purpose: Finds a Context7-compatible library ID for a given library name.
- Required parameters:
    - libraryName (string): the library name to look up (e.g., "numpy")
    - query (string): the user's question or context for relevance

Tool: query-docs
- Purpose: Retrieves up-to-date documentation and examples for a library ID.
- Required parameters:
    - libraryId (string): a Context7-compatible library ID (e.g., "/numpy/numpy")
    - query (string): the documentation query or topic

Always call these tools by naming them exactly and filling all required arguments correctly. Do not guess; if unsure, ask first for the correct parameters.
"""
    agent_test = create_agent(system_prompt=SYSTEM_PROMPT, api_key="", server_url="https://mcp.context7.com/mcp")
    result = agent_test.step(
        """
You are answering a question about a Python library API change.

You MUST use the provided tools to answer.

Tool usage rules:
1. First, call the tool "resolve_library_id".
   - Always provide BOTH arguments:
     - query: the library name or symbol (string)
     - libraryName: the library name as a string (do NOT omit this field)
   - Example:
     resolve_library_id(query="numpy", libraryName="numpy")

2. From the resolve result, select the most relevant Context7-compatible library ID
   (for NumPy, this is typically "/numpy/numpy").

3. Then call the tool "query_docs".
   - Arguments:
     - libraryId: the resolved Context7 library ID (string)
     - query: a precise documentation question (string)

4. Use ONLY the documentation returned by Context7 to form your final answer.
   Do NOT rely on prior knowledge.

User question:
numpy.compare_chararrays is not available in NumPy 2.0.
Explain what changed and how to fix existing code.

Output requirements:
- Output MUST be valid JSON
- Output format exactly:

{
  "ai_api_change": "Natural language explanation of the API change and reliable fix guidance."
}

Do NOT explain your reasoning steps.
Do NOT include tool call logs in the output.
        """
    )
    print(result)



"第一次执行结果:"
# msgs=[
#     BaseMessage(
#         role_name='Assistant', 
#         role_type=<RoleType.ASSISTANT: 'assistant'>, 
#         meta_dict={}, 
#         content='{\n    "ai_api_change": "The function `numpy.compare_chararrays` has been relocated from the main NumPy namespace to the `numpy.char` submodule and is now accessed as `numpy.char.compare_chararrays`. This change was introduced in NumPy 2.0. For better performance and future compatibility, consider migrating to the new `numpy.strings` namespace for string operations."\n}', 
#         video_bytes=None, 
#         image_list=None, 
#         image_detail='auto', 
#         video_detail='low', 
#         parsed=None
#     )
# ] 
# terminated=False 
# info={
#     'id': '019bc625ecd03540fc71c12389a46a9d', 
#     'usage': {
#         'completion_tokens': 84, 
#         'prompt_tokens': 2120, 
#         'total_tokens': 2204, 
#         'completion_tokens_details': {
#             'accepted_prediction_tokens': None, 
#             'audio_tokens': None, 
#             'reasoning_tokens': 0, 
#             'rejected_prediction_tokens': None
#         }, 
#         'prompt_tokens_details': None
#     }, 
#     'termination_reasons': ['stop'], 
#     'num_tokens': 1696, 
#     'tool_calls': [
#         ToolCallingRecord(
#             tool_name='resolve_library_id', 
#             args={'query': 'numpy', 'libraryName': None}, 
#             result={
#                 'result': {
#                     'content': [
#                         {
#                             'type': 'text', 
#                             'text': 'MCP error -32602: Input validation error: Invalid arguments for tool resolve-library-id: [\n  {\n    "code": "invalid_type",\n    "expected": "string",\n    "received": "undefined",\n    "path": [\n      "libraryName"\n    ],\n    "message": "Required"\n  }\n]'
#                         }
#                     ], 
#                     'isError': True
#                 }, 
#                 'jsonrpc': '2.0', 
#                 'id': 'adf9c4c3-bce9-4e77-835b-019c5e5de292'
#             }, 
#             tool_call_id='019bc625c614db8985117b739ccb37db'
#         ), 
#         ToolCallingRecord(
#             tool_name='resolve_library_id', 
#             args={'query': 'numpy', 'libraryName': 'numpy'}, 
#             result={
#                 'result': {
#                     'content': [
#                         {
#                             'type': 'text', 
#                             'text': 'Available Libraries:\n\nEach result includes:\n- Library ID: Context7-compatible identifier (format: /org/project)\n- Name: Library or package name\n- Description: Short summary\n- Code Snippets: Number of available code examples\n- Source Reputation: Authority indicator (High, Medium, Low, or Unknown)\n- Benchmark Score: Quality indicator (100 is the highest score)\n- Versions: List of versions if available. Use one of those versions if the user provides a version in their query. The format of the version is /org/project/version.\n\nFor best results, select libraries based on name match, source reputation, snippet coverage, benchmark score, and relevance to your use case.\n\n----------\n\n- Title: NumPy\n- Context7-compatible library ID: /numpy/numpy\n- Description: NumPy is the fundamental package for scientific computing with Python, providing a powerful N-dimensional array object and tools for linear algebra, Fourier transforms, and random number capabilities.\n- Code Snippets: 3593\n- Source Reputation: Unknown\n- Benchmark Score: 84.1\n- Versions: v2.3.1, v2.1.3\n----------\n- Title: CPMpy\n- Context7-compatible library ID: /cpmpy/cpmpy\n- Description: CPMpy is a constraint programming and modeling library in Python, based on NumPy, offering direct solver access for combinatorial problems.\n- Code Snippets: 382\n- Source Reputation: Medium\n- Benchmark Score: 94.5\n----------\n- Title: npcpy\n- Context7-compatible library ID: /npc-worldwide/npcpy\n- Description: npcpy is the core library of the NPC Toolkit, designed to enhance natural language processing pipelines and agent tooling for building advanced LLM applications and research.\n- Code Snippets: 99\n- Source Reputation: Medium\n- Benchmark Score: 69.4\n----------\n- Title: MusPy\n- Context7-compatible library ID: /salu133445/muspy\n- Description: MusPy is an open-source Python library that provides essential tools for symbolic music generation, including dataset management, data I/O, data preprocessing, and model evaluation.\n- Code Snippets: 918\n- Source Reputation: High\n- Benchmark Score: 80.5\n----------\n- Title: Musicpy\n- Context7-compatible library ID: /websites/musicpy_readthedocs_io_en\n- Description: Musicpy is a Python-based music programming language designed to write, analyze, and generate music using music theory and algorithms, supporting MIDI file operations and flexible syntax for expressing musical elements.\n- Code Snippets: 537\n- Source Reputation: High\n- Benchmark Score: 68.8'
#                         }
#                     ]
#                 }, 
#                 'jsonrpc': '2.0', 
#                 'id': '1039869d-01e0-44b5-b48c-2c33d29630a9'
#                 }, 
#             tool_call_id='019bc6277d6e5fd70c799fea6ce984a5'
#         ), 
#         ToolCallingRecord(
#             tool_name='query_docs', 
#             args={
#                 'libraryId': '/numpy/numpy', 
#                 'query': 'numpy.compare_chararrays not available in numpy 2.0'
#             }, 
#             result={
#                 'result': {
#                     'content': [
#                         {
#                             'type': 'text', 
#                             'text': "### Relocate numpy.compare_chararrays to numpy.char.compare_chararrays\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\n\nThe function `np.compare_chararrays` has been removed from the main NumPy namespace. It should now be accessed via the `np.char` submodule as `np.char.compare_chararrays`.\n\n```python\nimport numpy as np\n\n# Old (removed from main namespace) usage:\n# result = np.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\n\n# New (recommended) usage:\nresult = np.char.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\nprint(result)\n\nresult_ne = np.char.compare_chararrays(['hello', 'world'], ['hello', 'python'], '!=')\nprint(result_ne)\n```\n\n--------------------------------\n\n### Use numpy.strings namespace for string operations\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nNumPy 2.0 introduces a new numpy.strings namespace with string operations implemented as ufuncs. The old numpy.char namespace is still available but recommended to migrate to numpy.strings for better performance.\n\n```python\n# Old way (still works)\nimport numpy as np\nnp.char.upper(['hello', 'world'])\n\n# New recommended way\nimport numpy as np\nnp.strings.upper(['hello', 'world'])\n```\n\n### Character arrays\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/arrays.classes.rst\n\nThe `~numpy.char.chararray` class exists for backwards compatibility with Numarray, it is not recommended for new development. Starting from numpy 1.4, if one needs arrays of strings, it is recommended to use arrays of `dtype` `object_`, `bytes_` or `str_`, and use the free functions in the `numpy.char` module for fast vectorized string operations.\n\n--------------------------------\n\n### String functionality > Integration with numpy.char\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.strings.rst\n\nThe `numpy.strings` module universal functions are also used in `numpy.char`, which provides the `numpy.char.chararray` array subclass. This integration allows string routines to benefit from the performance optimizations of universal functions. Prior to NumPy 2.0, all string functionality was contained in `numpy.char`, which only operated on fixed-width strings. The `numpy.char` module will not be receiving updates and will be deprecated in the future.\n\n--------------------------------\n\n### String functionality > Comparison\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.strings.rst\n\nThe `numpy.strings` module exports comparison universal functions that can operate on string arrays. These comparison functions include `equal`, `not_equal`, `greater_equal`, `less_equal`, `greater`, and `less`, enabling element-wise comparison of string arrays and returning boolean arrays as results."
#                         }
#                     ]
#                 }, 
#                 'jsonrpc': '2.0', 
#                 'id': 'dc408fdd-32fc-46df-ac0f-52195b3f8c93'
#             }, 
#             tool_call_id='019bc625e59170d1d45ee8651c42d8f0'
#         )
#     ], 
#     'external_tool_call_requests': None
# }





"第二次执行结果，修改了prompt格式以及step中的任务描述"
# msgs=[
#     BaseMessage(
#         role_name='Assistant', 
#         role_type=<RoleType.ASSISTANT: 'assistant'>, meta_dict={}, 
#         content='{\n  "ai_api_change": "The function `numpy.compare_chararrays` has been removed from the main NumPy namespace in version 2.0. It has been relocated to the `numpy.char` submodule and should now be accessed as `numpy.char.compare_chararrays`. To fix existing code, replace instances of `np.compare_chararrays` with `np.char.compare_chararrays`. Example: `result = np.char.compare_chararrays([\'a\', \'b\'], [\'a\', \'c\'], \'==\')`."\n}', 
#         video_bytes=None, image_list=None, image_detail='auto', video_detail='low', parsed=None
#     )
# ] 
# terminated=False 
# info={
#     'id': '019bc6361236ce0950ecd23d1e135a50', 
#     'usage': {
#         'completion_tokens': 110, 'prompt_tokens': 2166, 
#         'total_tokens': 2276, 
#         'completion_tokens_details': {
#             'accepted_prediction_tokens': None, 
#             'audio_tokens': None, 
#             'reasoning_tokens': 0, 
#             'rejected_prediction_tokens': None
#         }, 
#     'prompt_tokens_details': None
#     }, 
#     'termination_reasons': ['stop'], 'num_tokens': 1746, 
#     'tool_calls': [
#         ToolCallingRecord(
#             tool_name='resolve_library_id', 
#             args={'query': 'numpy.compare_chararrays', 'libraryName': 'numpy'}, 
#             result={
#                 'result': {
#                     'content': [
#                         {
#                             'type': 'text', 
#                             'text': 'Available Libraries:\n\nEach result includes:\n- Library ID: Context7-compatible identifier (format: /org/project)\n- Name: Library or package name\n- Description: Short summary\n- Code Snippets: Number of available code examples\n- Source Reputation: Authority indicator (High, Medium, Low, or Unknown)\n- Benchmark Score: Quality indicator (100 is the highest score)\n- Versions: List of versions if available. Use one of those versions if the user provides a version in their query. The format of the version is /org/project/version.\n\nFor best results, select libraries based on name match, source reputation, snippet coverage, benchmark score, and relevance to your use case.\n\n----------\n\n- Title: NumPy\n- Context7-compatible library ID: /numpy/numpy\n- Description: NumPy is the fundamental package for scientific computing with Python, providing a powerful N-dimensional array object and tools for linear algebra, Fourier transforms, and random number capabilities.\n- Code Snippets: 3593\n- Source Reputation: Unknown\n- Benchmark Score: 84.1\n- Versions: v2.3.1, v2.1.3\n----------\n- Title: CPMpy\n- Context7-compatible library ID: /cpmpy/cpmpy\n- Description: CPMpy is a constraint programming and modeling library in Python, based on NumPy, offering direct solver access for combinatorial problems.\n- Code Snippets: 382\n- Source Reputation: Medium\n- Benchmark Score: 94.5\n----------\n- Title: npcpy\n- Context7-compatible library ID: /npc-worldwide/npcpy\n- Description: npcpy is the core library of the NPC Toolkit, designed to enhance natural language processing pipelines and agent tooling for building advanced LLM applications and research.\n- Code Snippets: 99\n- Source Reputation: Medium\n- Benchmark Score: 69.4'
#                         }
#                     ]
#                 }, 
#                 'jsonrpc': '2.0', 'id': 'c57ae2ce-be9c-4080-9fb4-3cf52c0110ae'
#             }, 
#             tool_call_id='019bc635f1dcded4e0bfde77a8ff0770'
#         ), 
#         ToolCallingRecord(
#             tool_name='query_docs', 
#             args={
#                 'libraryId': '/numpy/numpy', 
#                 'query': 'numpy.compare_chararrays removal in NumPy 2.0 and alternative'
#             }, 
#             result={
#                 'result': {
#                     'content': [
#                         {
#                             'type': 'text', 
#                             'text': "### Relocate numpy.compare_chararrays to numpy.char.compare_chararrays\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\n\nThe function `np.compare_chararrays` has been removed from the main NumPy namespace. It should now be accessed via the `np.char` submodule as `np.char.compare_chararrays`.\n\n```python\nimport numpy as np\n\n# Old (removed from main namespace) usage:\n# result = np.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\n\n# New (recommended) usage:\nresult = np.char.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\nprint(result)\n\nresult_ne = np.char.compare_chararrays(['hello', 'world'], ['hello', 'python'], '!=')\nprint(result_ne)\n```\n\n--------------------------------\n\n### Handle Deprecation of chararray in Main NumPy Namespace\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\n\nThe `chararray` type in the main NumPy namespace (`np.chararray`) is deprecated. While it can still be imported from `np.char.chararray` without a deprecation warning for now, users should be aware of its planned full removal.\n\n```python\nimport numpy as np\n\n# Deprecated in main namespace, but still importable from np.char\nfrom numpy.char import chararray\n\n# Creating a chararray\narr = chararray((3,), itemsize=5)\narr[:] = 'test'\nprint(arr)\n\n# Accessing an element\nprint(arr[0])\n```\n\n### Release Notes > Removals > Namespace Changes\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\n\nSeveral functions have been removed from the main NumPy namespace and relocated to submodules. ``np.compare_chararrays`` is now accessible as ``np.char.compare_chararrays``, ``np.format_parser`` has moved to ``np.rec.format_parser``, and ``np.round_`` has been replaced with ``np.round``. The ``charrarray`` class in the main namespace is deprecated and should be imported from ``np.char.chararray`` instead.\n\n--------------------------------\n\n### Character arrays\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/arrays.classes.rst\n\nThe `~numpy.char.chararray` class exists for backwards compatibility with Numarray, it is not recommended for new development. Starting from numpy 1.4, if one needs arrays of strings, it is recommended to use arrays of `dtype` `object_`, `bytes_` or `str_`, and use the free functions in the `numpy.char` module for fast vectorized string operations.\n\n--------------------------------\n\n### Legacy fixed-width string functionality > Comparison > Whitespace Handling\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.char.rst\n\nUnlike the standard numpy comparison operators, the comparison operators in the `char` module strip trailing whitespace characters before performing the comparison. This behavior is useful when comparing strings that may have different amounts of padding, as it focuses the comparison on the actual string content rather than whitespace differences."
#                         }
#                     ]
#                 }, 
#                 'jsonrpc': '2.0', 'id': '9315516a-407f-4c52-9afa-ccfe63d245d2'
#             }, 
#             tool_call_id='019bc63603928f7175ac34d23e287248'
#         )
#     ], 
#     'external_tool_call_requests': None
# }