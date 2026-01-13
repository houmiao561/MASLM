import requests
import json


def query_context7_docs(
    api_key: str,
    owner: str,
    project: str,
    topic: str,
    timeout: int = 15
) -> dict:
    base = "https://context7.com/api/v2/docs/code"
    url = f"{base}/{owner}/{project}?topic={topic}"

    print("Request URL:")
    print(url)
    print()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "*/*",
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")

    print("HTTP status:", resp.status_code)
    print("Content-Type:", content_type)
    print()

    # —— 核心修复逻辑 ——
    if "application/json" in content_type:
        data = resp.json()
        print("Parsed JSON response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return {
            "type": "json",
            "data": data,
        }
    else:
        text = resp.text
        print("Raw document response (preview):")
        print(text)
        return {
            "type": "document",
            "content_type": content_type,
            "text": text,
        }


if __name__ == "__main__":
    API_KEY = "ctx7sk-97bd7e64-9cb4-477e-a13e-51c267f58e6e"
    owner = "numpy"
    project = "numpy"
    topic = "numpy.compare_chararrays in 2.0"

    result = query_context7_docs(API_KEY, owner, project, topic)

    print("\n=== Final returned object ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))



def find_largest_equal_substring(arr1, arr2):
    import numpy
    max_len = 0
    max_substring = ''
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            length = 0
            while (
                i + length < len(arr1)
                and j + length < len(arr2)
                and numpy.char.compare_chararrays(
                    arr1[i + length],
                    arr2[j + length],
                    '==',
                    True
                )
            ):
                length += 1
            if length > max_len:
                max_len = length
                max_substring = arr1[i:i + length]
    return max_substring



Request URL:
https://context7.com/api/v2/docs/code/numpy/numpy?topic=numpy.compare_chararrays in 2.0

HTTP status: 200
Content-Type: text/plain; charset=utf-8

Raw document response (preview):
### Relocate numpy.compare_chararrays to numpy.char.compare_chararrays

Source: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst

The function `np.compare_chararrays` has been removed from the main NumPy namespace. It should now be accessed via the `np.char` submodule as `np.char.compare_chararrays`.

```python
import numpy as np

# Old (removed from main namespace) usage:
# result = np.compare_chararrays(['a', 'b'], ['a', 'c'], '==')

# New (recommended) usage:
result = np.char.compare_chararrays(['a', 'b'], ['a', 'c'], '==')
print(result)

result_ne = np.char.compare_chararrays(['hello', 'world'], ['hello', 'python'], '!=')
print(result_ne)
```

--------------------------------

### Use numpy.strings namespace for string operations

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

NumPy 2.0 introduces a new numpy.strings namespace with string operations implemented as ufuncs. The old numpy.char namespace is still available but recommended to migrate to numpy.strings for better performance.

```python
# Old way (still works)
import numpy as np
np.char.upper(['hello', 'world'])

# New recommended way
import numpy as np
np.strings.upper(['hello', 'world'])
```

--------------------------------

### Migrate np.asfarray to np.asarray with float dtype (Python)

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Instead of `np.asfarray`, use `np.

--------------------------------

### Implement __array__ method with copy parameter handling

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Add dtype=None and copy=None keywords to __array__ method signatures in array-like objects for NumPy 2.0 compatibility. This ensures proper copy behavior based on copy parameter values (True/False/None) while maintaining backward compatibility with older NumPy versions.

```python
class ArrayLike:
    def __array__(self, dtype=None, copy=None):
        # copy=True: always return a new copy
        if copy is True:
            return np.array(self.data, dtype=dtype, copy=True)
        # copy=None: create a copy if required (e.g., by dtype)
        elif copy is None:
            return np.array(self.data, dtype=dtype)
        # copy=False: never make a copy, raise if copy is needed
        elif copy is False:
            if dtype is not None and dtype != self.data.dtype:
                raise ValueError("Copy required but copy=False")
            return self.data
        else:
            return np.array(self.data, dtype=dtype)
```

--------------------------------

### Check NumPy runtime version in C-API

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Use runtime version checking to implement different behavior between NumPy 1.x and 2.0 when compiling C extension code. This macro comparison allows conditional logic based on the NumPy version at compilation time.

```c
if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
  /* NumPy 2.0 specific code */
} else {
  /* NumPy 1.x code */
}
```

--------------------------------

### Include ndarrayobject.h for dtype accessor functions

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Include ndarrayobject.h (or similar) instead of only ndarraytypes.h to access dtype flag checking functions and array item operations. This include is required when using npy_2_compat.h with NumPy 1.x and ensures proper import_array() functionality.

```C
#include "numpy/ndarrayobject.h"
/* Now PyDataType_FLAGCHK, PyDataType_REFCHK, PyArray_GETITEM, etc. are available */
```

--------------------------------

### Migrate np.alltrue to np.all (Python)

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Replace `np.alltrue` with `np.all` for checking if all elements in an array evaluate to true.

```python
import numpy as np

arr = np.array([True, True, False])

# Old way (removed in np 2.0)
# result_old = np.alltrue(arr)

# New way
result_new = np.all(arr)
print(result_new)
```

--------------------------------

### Replace np.array with np.asarray for copy-if-needed behavior

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Migrate legacy code that uses np.array(..., copy=False) to np.asarray(...) for improved compatibility and performance. The np.asarray function is now preferred as it has equivalent or better overhead compared to np.array with explicit copy parameters.

```python
# Old approach (NumPy 1.x)
result = np.array(data, copy=False)

# New approach (NumPy 1.x and 2.x compatible)
result = np.asarray(data)
```

--------------------------------

### Function PyDataType_GetArrFuncs to Fetch Legacy ArrFuncs

Source: https://github.com/numpy/numpy/blob/main/doc/source/reference/c-api/types-and-structures.rst

This C function retrieves the legacy 'PyArray_ArrFuncs' structure associated with a given 'PyArray_Descr' data type. Introduced in NumPy 2.0 for backward compatibility, it should be used instead of directly accessing the '->f' slot of 'PyArray_Descr' for future-proof code.

```c
PyArray_ArrFuncs *PyDataType_GetArrFuncs(PyArray_Descr *dtype)
```

--------------------------------

### Replace newbyteorder with ndarray.view method

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Migration from deprecated ndarray.newbyteorder() method. Use the view() method with a newbyteorder() call on the dtype instead to change byte order of array data.

```python
# Old way (deprecated)
arr.newbyteorder(order)

# New way
arr.view(arr.dtype.newbyteorder(order))
```

=== Final returned object ===
{
  "type": "document",
  "content_type": "text/plain; charset=utf-8",
  "text": "### Relocate numpy.compare_chararrays to numpy.char.compare_chararrays\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/release/2.0.0-notes.rst\n\nThe function `np.compare_chararrays` has been removed from the main NumPy namespace. It should now be accessed via the `np.char` submodule as `np.char.compare_chararrays`.\n\n```python\nimport numpy as np\n\n# Old (removed from main namespace) usage:\n# result = np.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\n\n# New (recommended) usage:\nresult = np.char.compare_chararrays(['a', 'b'], ['a', 'c'], '==')\nprint(result)\n\nresult_ne = np.char.compare_chararrays(['hello', 'world'], ['hello', 'python'], '!=')\nprint(result_ne)\n```\n\n--------------------------------\n\n### Use numpy.strings namespace for string operations\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nNumPy 2.0 introduces a new numpy.strings namespace with string operations implemented as ufuncs. The old numpy.char namespace is still available but recommended to migrate to numpy.strings for better performance.\n\n```python\n# Old way (still works)\nimport numpy as np\nnp.char.upper(['hello', 'world'])\n\n# New recommended way\nimport numpy as np\nnp.strings.upper(['hello', 'world'])\n```\n\n--------------------------------\n\n### Migrate np.asfarray to np.asarray with float dtype (Python)\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nInstead of `np.asfarray`, use `np.\n\n--------------------------------\n\n### Implement __array__ method with copy parameter handling\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nAdd dtype=None and copy=None keywords to __array__ method signatures in array-like objects for NumPy 2.0 compatibility. This ensures proper copy behavior based on copy parameter values (True/False/None) while maintaining backward compatibility with older NumPy versions.\n\n```python\nclass ArrayLike:\n    def __array__(self, dtype=None, copy=None):\n        # copy=True: always return a new copy\n        if copy is True:\n            return np.array(self.data, dtype=dtype, copy=True)\n        # copy=None: create a copy if required (e.g., by dtype)\n        elif copy is None:\n            return np.array(self.data, dtype=dtype)\n        # copy=False: never make a copy, raise if copy is needed\n        elif copy is False:\n            if dtype is not None and dtype != self.data.dtype:\n                raise ValueError(\"Copy required but copy=False\")\n            return self.data\n        else:\n            return np.array(self.data, dtype=dtype)\n```\n\n--------------------------------\n\n### Check NumPy runtime version in C-API\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nUse runtime version checking to implement different behavior between NumPy 1.x and 2.0 when compiling C extension code. This macro comparison allows conditional logic based on the NumPy version at compilation time.\n\n```c\nif (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {\n  /* NumPy 2.0 specific code */\n} else {\n  /* NumPy 1.x code */\n}\n```\n\n--------------------------------\n\n### Include ndarrayobject.h for dtype accessor functions\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nInclude ndarrayobject.h (or similar) instead of only ndarraytypes.h to access dtype flag checking functions and array item operations. This include is required when using npy_2_compat.h with NumPy 1.x and ensures proper import_array() functionality.\n\n```C\n#include \"numpy/ndarrayobject.h\"\n/* Now PyDataType_FLAGCHK, PyDataType_REFCHK, PyArray_GETITEM, etc. are available */\n```\n\n--------------------------------\n\n### Migrate np.alltrue to np.all (Python)\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nReplace `np.alltrue` with `np.all` for checking if all elements in an array evaluate to true.\n\n```python\nimport numpy as np\n\narr = np.array([True, True, False])\n\n# Old way (removed in np 2.0)\n# result_old = np.alltrue(arr)\n\n# New way\nresult_new = np.all(arr)\nprint(result_new)\n```\n\n--------------------------------\n\n### Replace np.array with np.asarray for copy-if-needed behavior\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nMigrate legacy code that uses np.array(..., copy=False) to np.asarray(...) for improved compatibility and performance. The np.asarray function is now preferred as it has equivalent or better overhead compared to np.array with explicit copy parameters.\n\n```python\n# Old approach (NumPy 1.x)\nresult = np.array(data, copy=False)\n\n# New approach (NumPy 1.x and 2.x compatible)\nresult = np.asarray(data)\n```\n\n--------------------------------\n\n### Function PyDataType_GetArrFuncs to Fetch Legacy ArrFuncs\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/reference/c-api/types-and-structures.rst\n\nThis C function retrieves the legacy 'PyArray_ArrFuncs' structure associated with a given 'PyArray_Descr' data type. Introduced in NumPy 2.0 for backward compatibility, it should be used instead of directly accessing the '->f' slot of 'PyArray_Descr' for future-proof code.\n\n```c\nPyArray_ArrFuncs *PyDataType_GetArrFuncs(PyArray_Descr *dtype)\n```\n\n--------------------------------\n\n### Replace newbyteorder with ndarray.view method\n\nSource: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst\n\nMigration from deprecated ndarray.newbyteorder() method. Use the view() method with a newbyteorder() call on the dtype instead to change byte order of array data.\n\n```python\n# Old way (deprecated)\narr.newbyteorder(order)\n\n# New way\narr.view(arr.dtype.newbyteorder(order))\n```"
}