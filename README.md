### 原始函数

```python
def find_largest_equal_substring(arr1, arr2):
    import numpy
    max_len = 0
    max_substring = ''
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            length = 0
            while (i + length < len(arr1) and j + length < len(arr2) and
                numpy.compare_chararrays(arr1[i + length], arr2[j + length], '==', True)):
                    length += 1
                    if length > max_len:
                        max_len = length
                        max_substring = arr1[i:i + length]
    return max_substring
```

### 命题人建议的修改结果

```json
"compare_signature": "numpy.char.compare_chararrays(char1, char2, cmp, assume_equal)"
```

### baseline 的修改结果

```python
def find_largest_equal_substring(arr1, arr2):
    import numpy
    max_len = 0
    max_substring = ''
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            length = 0
            while (i + length < len(arr1) and j + length < len(arr2)
                   and
                   arr1[i + length] == arr2[j + length]):
                length += 1
                if length > max_len:
                    max_len = length
                    max_substring = arr1[i:i + length]
                    return max_substring
```

### MASLM

```py
# MASLM
def find_largest_equal_substring(arr1, arr2):
    import numpy
    max_len = 0
    max_substring = ''
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            length = 0
            while (i + length < len(arr1) and j + length < len(arr2)
                   and
                   numpy.char.compare_chararrays(arr1[i + length], arr2[j + length], '==', True)):
                length += 1
                if length > max_len:
                    max_len = length
                    max_substring = arr1[i:i + length]
                    return max_substring
```

### MASLM 证据

````json
{
  "ast_structure": [
    {
      "function_name": "find_largest_equal_substring",
      "lineno": 1,
      "api_calls": [
        { "api": "range", "lineno": 5, "context": "expression" },
        { "api": "len", "lineno": 5, "context": "expression" },
        { "api": "range", "lineno": 6, "context": "expression" },
        { "api": "len", "lineno": 6, "context": "expression" },
        { "api": "len", "lineno": 8, "context": "expression" },
        { "api": "len", "lineno": 8, "context": "expression" },
        {
          "api": "numpy.compare_chararrays",
          "lineno": 9,
          "context": "expression"
        }
      ]
    }
  ],
  "ai_api_wrong": "numpy.compare_chararrays",
  "line_number": 9,
  "natural_language_questions": "Why is numpy.compare_chararrays not available in 2.0?",
  "ai_api_answer_change": {
    "what_changed": "The function `numpy.compare_chararrays` was moved from the main NumPy namespace to the `numpy.char` submodule.",
    "why_it_breaks": "The code breaks because it attempts to access `numpy.compare_chararrays` directly, which is no longer available in the main namespace.",
    "how_to_fix": "Replace `numpy.compare_chararrays` with `numpy.char.compare_chararrays` in the code."
  },
  "reason_type": "Removed",
  "mcp_raw": {
    "resolve_library_id": "{'result': {'content': [{'type': 'text', 'text': 'Available Libraries:\\n\\nEach result includes:\\n- Library ID: Context7-compatible identifier (format: /org/project)\\n- Name: Library or package name\\n- Description: Short summary\\n- Code Snippets: Number of available code examples\\n- Source Reputation: Authority indicator (High, Medium, Low, or Unknown)\\n- Benchmark Score: Quality indicator (100 is the highest score)\\n- Versions: List of versions if available. Use one of those versions if the user provides a version in their query. The format of the version is /org/project/version.\\n\\nFor best results, select libraries based on name match, source reputation, snippet coverage, benchmark score, and relevance to your use case.\\n\\n----------\\n\\n- Title: NumPy\\n- Context7-compatible library ID: /numpy/numpy\\n- Description: NumPy is the fundamental package for scientific computing with Python, providing a powerful N-dimensional array object and tools for linear algebra, Fourier transforms, and random number capabilities.\\n- Code Snippets: 3593\\n- Source Reputation: Unknown\\n- Benchmark Score: 84.1\\n- Versions: v2.3.1, v2.1.3'}]}, 'jsonrpc': '2.0', 'id': '66a65663-be10-49b8-9cd2-37f0e9512200'}",
    "query_docs": "{'result': {'content': [{'type': 'text', 'text': \"### Relocate numpy.compare_chararrays to numpy.char.compare_chararrays\\n\\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\\n\\nThe function `np.compare_chararrays` has been removed from the main NumPy namespace. It should now be accessed via the `np.char` submodule as `np.char.compare_chararrays`.\\n\\n```python\\nimport numpy as np\\n\\n# Old (removed from main namespace) usage:\\n# result = np.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\\n\\n# New (recommended) usage:\\nresult = np.char.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\\nprint(result)\\n\\nresult_ne = np.char.compare_chararrays(['hello', 'world'], ['hello', 'python'], '!=')\\nprint(result_ne)\\n```\\n\\n--------------------------------\\n\\n### Inspect NumPy API surface and modules in Python\\n\\nSource: https://github.com/numpy/numpy/blob/main/doc/neps/nep-0052-python-api-cleanup.rst\\n\\nThis Python code snippet inspects the NumPy namespace to identify the total number of public API objects and lists all available modules. It demonstrates the large API surface (562 public objects across 14 modules) that motivated the cleanup effort in NEP 52.\\n\\n```python\\n>>> objects_in_api = [s for s in dir(np) if not s.startswith('_')]\\n>>> len(objects_in_api)\\n562\\n>>> modules = [s for s in objects_in_api if inspect.ismodule(eval(f'np.{s}'))]\\n>>> modules\\n['char', 'compat', 'ctypeslib', 'emath', 'fft', 'lib', 'linalg', 'ma', 'math', 'polynomial', 'random', 'rec', 'testing', 'version']\\n>>> len(modules)\\n14\\n```\\n\\n### String functionality > Integration with numpy.char\\n\\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/routines.strings.rst\\n\\nThe `numpy.strings` module universal functions are also used in `numpy.char`, which provides the `numpy.char.chararray` array subclass. This integration allows string routines to benefit from the performance optimizations of universal functions. Prior to NumPy 2.0, all string functionality was contained in `numpy.char`, which only operated on fixed-width strings. The `numpy.char` module will not be receiving updates and will be deprecated in the future.\\n\\n--------------------------------\\n\\n### Character arrays\\n\\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/arrays.classes.rst\\n\\nThe `~numpy.char.chararray` class exists for backwards compatibility with Numarray, it is not recommended for new development. Starting from numpy 1.4, if one needs arrays of strings, it is recommended to use arrays of `dtype` `object_`, `bytes_` or `str_`, and use the free functions in the `numpy.char` module for fast vectorized string operations.\\n\\n--------------------------------\\n\\n### NumPy 2.0 > Python API Migration\\n\\nSource: https://github.com/numpy/numpy/blob/main/doc/source/dev/depending_on_numpy.rst\\n\\nIf your package relies on a large portion of NumPy's Python API surface, consider adding a `numpy<2.0` requirement to your metadata until you've confirmed your code works with NumPy 2.0. NumPy 2.0 includes significant API cleanup with many aliases and deprecated objects being removed (see numpy-2-migration-guide and NEP 52). Unless your code only uses modern and recommended functions, you will likely need to make adjustments for 2.0 compatibility. Plan to release an updated version of your packages shortly after NumPy 2.0's first release candidate becomes available.\"}]}, 'jsonrpc': '2.0', 'id': '73fe8816-4ba4-44b2-b59e-6a7e4844a6d4'}"
  },
  "mcp_evidence_summary": "The function `numpy.compare_chararrays` has been removed from the main NumPy namespace in version 2.0 and relocated to `numpy.char.compare_chararrays`. This change was part of NumPy's API cleanup efforts.",
  "ai_api_fix_function": "def find_largest_equal_substring(arr1, arr2):\n    import numpy\n    max_len = 0\n    max_substring = ''\n    for i in range(len(arr1)):\n        for j in range(len(arr2)):\n            length = 0\n            while (i + length < len(arr1) and j + length < len(arr2) and \n                   numpy.char.compare_chararrays(arr1[i + length], arr2[j + length], '==', True)):\n                length += 1\n            if length > max_len:\n                max_len = length\n                max_substring = arr1[i:i + length]\n    return max_substring"
}
````

## 01.17 单个测试用例结果记录

token 从 3000 增加到 9000
时间增加到 100 秒左右
