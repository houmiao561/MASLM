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

---

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

---

### Migrate np.asfarray to np.asarray with float dtype (Python)

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Instead of `np.asfarray`, use `np.

---

### Implement **array** method with copy parameter handling

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Add dtype=None and copy=None keywords to **array** method signatures in array-like objects for NumPy 2.0 compatibility. This ensures proper copy behavior based on copy parameter values (True/False/None) while maintaining backward compatibility with older NumPy versions.

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

---

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

---

### Include ndarrayobject.h for dtype accessor functions

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Include ndarrayobject.h (or similar) instead of only ndarraytypes.h to access dtype flag checking functions and array item operations. This include is required when using npy_2_compat.h with NumPy 1.x and ensures proper import_array() functionality.

```C
#include "numpy/ndarrayobject.h"
/* Now PyDataType_FLAGCHK, PyDataType_REFCHK, PyArray_GETITEM, etc. are available */
```

---

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

---

### Replace np.array with np.asarray for copy-if-needed behavior

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Migrate legacy code that uses np.array(..., copy=False) to np.asarray(...) for improved compatibility and performance. The np.asarray function is now preferred as it has equivalent or better overhead compared to np.array with explicit copy parameters.

```python
# Old approach (NumPy 1.x)
result = np.array(data, copy=False)

# New approach (NumPy 1.x and 2.x compatible)
result = np.asarray(data)
```

---

### Function PyDataType_GetArrFuncs to Fetch Legacy ArrFuncs

Source: https://github.com/numpy/numpy/blob/main/doc/source/reference/c-api/types-and-structures.rst

This C function retrieves the legacy 'PyArray_ArrFuncs' structure associated with a given 'PyArray_Descr' data type. Introduced in NumPy 2.0 for backward compatibility, it should be used instead of directly accessing the '->f' slot of 'PyArray_Descr' for future-proof code.

```c
PyArray_ArrFuncs *PyDataType_GetArrFuncs(PyArray_Descr *dtype)
```

---

### Replace newbyteorder with ndarray.view method

Source: https://github.com/numpy/numpy/blob/main/doc/source/numpy_2_0_migration_guide.rst

Migration from deprecated ndarray.newbyteorder() method. Use the view() method with a newbyteorder() call on the dtype instead to change byte order of array data.

```python
# Old way (deprecated)
arr.newbyteorder(order)

# New way
arr.view(arr.dtype.newbyteorder(order))
```
