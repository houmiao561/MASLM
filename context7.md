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

---

---

---

---

- ## --
  ***

# NumPy - Numerical Computing Library for Python

## Introduction

NumPy is the fundamental package for scientific computing with Python, providing a powerful N-dimensional array object and a comprehensive suite of mathematical operations. It serves as the foundation for the entire scientific Python ecosystem, offering high-performance array operations, linear algebra capabilities, Fourier transforms, and random number generation. NumPy's core is implemented in C and Fortran, delivering near-native performance for numerical computations while maintaining a clean, Pythonic interface. The library requires Python 3.12 or higher and is built using the modern Meson build system.

The library is designed around the ndarray object, a homogeneous multidimensional array with sophisticated broadcasting semantics that enable vectorized operations without explicit loops. NumPy provides tools for integrating C/C++ and Fortran code through F2PY, extensive linear algebra operations via BLAS/LAPACK backends (OpenBLAS, MKL, or Accelerate on macOS), fast Fourier transforms using the PocketFFT library, and a modern random number generation API supporting multiple bit generators (PCG64, MT19937, Philox, SFC64). It supports a rich type system including integers, floats, complex numbers, datetime objects, and structured arrays, making it suitable for diverse scientific and engineering applications.

---

## APIs and Key Functions

### Array Creation from Sequences

Create NumPy arrays from Python lists, tuples, or other array-like objects with automatic type inference or explicit dtype specification.

```python
import numpy as np

# Create 1D array from list
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)
# Output: array([1, 2, 3, 4, 5])

# Create 2D array from nested lists
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Output: array([[1, 2, 3],
#                [4, 5, 6]])

# Specify dtype explicitly
arr_float = np.array([1, 2, 3], dtype=np.float64)
print(arr_float)
# Output: array([1., 2., 3.])

# Create array from tuple
arr_tuple = np.array((10, 20, 30))
print(arr_tuple)
# Output: array([10, 20, 30])

# Handle error case - dimension mismatch
try:
    irregular = np.array([[1, 2], [3, 4, 5]])  # Irregular nested list
except ValueError as e:
    print(f"Error: {e}")
```

### Array Creation with Zeros, Ones, and Empty

Generate arrays filled with zeros, ones, or uninitialized values for memory efficiency in large-scale computations.

```python
import numpy as np

# Create array of zeros
zeros_arr = np.zeros((3, 4))
print(zeros_arr)
# Output: array([[0., 0., 0., 0.],
#                [0., 0., 0., 0.],
#                [0., 0., 0., 0.]])

# Create array of ones with specific dtype
ones_arr = np.ones((2, 3), dtype=np.int32)
print(ones_arr)
# Output: array([[1, 1, 1],
#                [1, 1, 1]], dtype=int32)

# Create uninitialized array (faster, contains garbage values)
empty_arr = np.empty((2, 2))
print(empty_arr.shape)
# Output: (2, 2)

# Create array filled with specific value
full_arr = np.full((3, 3), 7.5)
print(full_arr)
# Output: array([[7.5, 7.5, 7.5],
#                [7.5, 7.5, 7.5],
#                [7.5, 7.5, 7.5]])

# Create identity matrix
identity = np.eye(4)
print(identity)
# Output: array([[1., 0., 0., 0.],
#                [0., 1., 0., 0.],
#                [0., 0., 1., 0.],
#                [0., 0., 0., 1.]])

# Create identity with offset diagonal
offset_eye = np.eye(4, k=1)  # k=1 means diagonal shifted right
print(offset_eye)
# Output: array([[0., 1., 0., 0.],
#                [0., 0., 1., 0.],
#                [0., 0., 0., 1.],
#                [0., 0., 0., 0.]])
```

### Array Creation with Ranges and Sequences

Generate evenly-spaced sequences of numbers using arange, linspace, or logspace for numerical analysis and plotting.

```python
import numpy as np

# Create sequence with step size (like Python range)
arange_arr = np.arange(0, 10, 2)
print(arange_arr)
# Output: array([0, 2, 4, 6, 8])

# Create float sequence with decimal step
float_range = np.arange(0.0, 1.0, 0.1)
print(float_range)
# Output: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# Create linearly spaced array (includes endpoint by default)
linspace_arr = np.linspace(0, 1, 5)
print(linspace_arr)
# Output: array([0.  , 0.25, 0.5 , 0.75, 1.  ])

# Exclude endpoint
linspace_no_end = np.linspace(0, 1, 5, endpoint=False)
print(linspace_no_end)
# Output: array([0. , 0.2, 0.4, 0.6, 0.8])

# Create logarithmically spaced array (base 10)
logspace_arr = np.logspace(0, 3, 4)
print(logspace_arr)
# Output: array([   1.,   10.,  100., 1000.])

# Logarithmic with different base (base 2)
logspace_base2 = np.logspace(0, 4, 5, base=2)
print(logspace_base2)
# Output: array([ 1.,  2.,  4.,  8., 16.])

# Generate mesh grid for 2D plotting
x = np.linspace(-2, 2, 5)
y = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(x, y)
print(X)
# Output: array([[-2., -1.,  0.,  1.,  2.],
#                [-2., -1.,  0.,  1.,  2.],
#                [-2., -1.,  0.,  1.,  2.]])
print(Y)
# Output: array([[-1., -1., -1., -1., -1.],
#                [ 0.,  0.,  0.,  0.,  0.],
#                [ 1.,  1.,  1.,  1.,  1.]])
```

### Array Indexing and Slicing

Access and modify array elements using integer indexing, slicing, boolean masks, and fancy indexing for flexible data manipulation.

```python
import numpy as np

# Basic indexing
arr = np.array([10, 20, 30, 40, 50])
print(arr[0])      # Output: 10
print(arr[-1])     # Output: 50
print(arr[1:4])    # Output: array([20, 30, 40])

# 2D indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[0, 1])     # Output: 2
print(arr2d[1, :])     # Output: array([4, 5, 6])
print(arr2d[:, 2])     # Output: array([3, 6, 9])

# Boolean indexing
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(arr[mask])       # Output: array([4, 5, 6])

# Boolean mask with condition
data = np.array([10, -5, 15, -3, 20, 8])
positive = data[data > 0]
print(positive)        # Output: array([10, 15, 20,  8])

# Fancy indexing with integer arrays
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
print(arr[indices])    # Output: array([10, 30, 50])

# 2D fancy indexing
arr2d = np.array([[1, 2], [3, 4], [5, 6]])
row_indices = np.array([0, 2])
col_indices = np.array([1, 0])
print(arr2d[row_indices, col_indices])  # Output: array([2, 5])

# Modify elements using indexing
arr = np.array([1, 2, 3, 4, 5])
arr[arr > 3] = 0
print(arr)             # Output: array([1, 2, 3, 0, 0])

# Error handling - out of bounds
try:
    arr = np.array([1, 2, 3])
    value = arr[10]
except IndexError as e:
    print(f"Error: {e}")
    # Output: Error: index 10 is out of bounds for axis 0 with size 3
```

### Array Shape Manipulation

Reshape, transpose, flatten, and reorganize arrays without copying data when possible for memory efficiency.

```python
import numpy as np

# Reshape array
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
print(reshaped)
# Output: array([[ 0,  1,  2,  3],
#                [ 4,  5,  6,  7],
#                [ 8,  9, 10, 11]])

# Reshape with -1 (infer dimension)
auto_reshape = arr.reshape(2, -1)
print(auto_reshape)
# Output: array([[ 0,  1,  2,  3,  4,  5],
#                [ 6,  7,  8,  9, 10, 11]])

# Flatten array to 1D
flattened = reshaped.flatten()
print(flattened)
# Output: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# Ravel (flatten without copying if possible)
raveled = reshaped.ravel()
print(raveled)
# Output: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# Transpose
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr2d.T
print(transposed)
# Output: array([[1, 4],
#                [2, 5],
#                [3, 6]])

# Add new axis
arr = np.array([1, 2, 3])
col_vector = arr[:, np.newaxis]
print(col_vector)
# Output: array([[1],
#                [2],
#                [3]])

# Squeeze - remove single-dimensional entries
arr = np.array([[[1], [2], [3]]])
print(arr.shape)        # Output: (1, 3, 1)
squeezed = np.squeeze(arr)
print(squeezed.shape)   # Output: (3,)
print(squeezed)         # Output: array([1, 2, 3])

# Concatenate arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
vstack_result = np.vstack([arr1, arr2])
print(vstack_result)
# Output: array([[1, 2],
#                [3, 4],
#                [5, 6],
#                [7, 8]])

hstack_result = np.hstack([arr1, arr2])
print(hstack_result)
# Output: array([[1, 2, 5, 6],
#                [3, 4, 7, 8]])

# Error handling - incompatible reshape
try:
    arr = np.arange(10)
    bad_reshape = arr.reshape(3, 4)  # 10 elements can't fit in 3x4
except ValueError as e:
    print(f"Error: cannot reshape")
```

### Broadcasting and Vectorized Operations

Perform element-wise operations on arrays of different shapes automatically through broadcasting rules for efficient computation.

```python
import numpy as np

# Basic broadcasting - scalar with array
arr = np.array([1, 2, 3, 4])
result = arr * 2
print(result)
# Output: array([2, 4, 6, 8])

# Broadcasting with 1D and 2D arrays
arr1d = np.array([1, 2, 3])
arr2d = np.array([[10], [20], [30]])
result = arr1d + arr2d
print(result)
# Output: array([[11, 12, 13],
#                [21, 22, 23],
#                [31, 32, 33]])

# Broadcasting with compatible shapes
a = np.array([[1, 2, 3]])      # shape (1, 3)
b = np.array([[1], [2], [3]])  # shape (3, 1)
result = a + b                  # broadcasts to (3, 3)
print(result)
# Output: array([[2, 3, 4],
#                [3, 4, 5],
#                [4, 5, 6]])

# Element-wise operations
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
print(x + y)   # Output: array([11, 22, 33, 44])
print(x * y)   # Output: array([10, 40, 90, 160])
print(x ** 2)  # Output: array([ 1,  4,  9, 16])

# Broadcasting with mathematical functions
arr = np.array([[1, 2, 3], [4, 5, 6]])
squared = arr ** 2
print(squared)
# Output: array([[ 1,  4,  9],
#                [16, 25, 36]])

# Broadcasting for normalization
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col_mean = data.mean(axis=0)
normalized = data - col_mean
print(normalized)
# Output: array([[-3., -3., -3.],
#                [ 0.,  0.,  0.],
#                [ 3.,  3.,  3.]])

# Error - incompatible shapes
try:
    a = np.array([[1, 2, 3]])      # shape (1, 3)
    b = np.array([[1, 2]])          # shape (1, 2)
    result = a + b
except ValueError as e:
    print("Error: operands could not be broadcast together")
```

### Universal Functions (ufuncs)

Apply element-wise mathematical operations using optimized C implementations for trigonometry, exponentials, logarithms, and more.

```python
import numpy as np

# Trigonometric functions
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
sin_vals = np.sin(angles)
cos_vals = np.cos(angles)
print(sin_vals)
# Output: array([0.   , 0.5  , 0.707, 0.866, 1.   ])
print(cos_vals)
# Output: array([1.   , 0.866, 0.707, 0.5  , 0.   ])

# Exponential and logarithmic
x = np.array([1, 2, 3, 4])
exp_result = np.exp(x)
print(exp_result)
# Output: array([ 2.718,  7.389, 20.086, 54.598])

log_result = np.log(x)
print(log_result)
# Output: array([0.   , 0.693, 1.099, 1.386])

log10_result = np.log10(x)
print(log10_result)
# Output: array([0.   , 0.301, 0.477, 0.602])

# Square root and power
arr = np.array([1, 4, 9, 16, 25])
sqrt_result = np.sqrt(arr)
print(sqrt_result)
# Output: array([1., 2., 3., 4., 5.])

power_result = np.power(arr, 0.5)
print(power_result)
# Output: array([1., 2., 3., 4., 5.])

# Comparison ufuncs
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 2, 2])
print(np.greater(a, b))
# Output: array([False, False,  True,  True])
print(np.equal(a, b))
# Output: array([False,  True, False, False])

# Rounding and absolute value
arr = np.array([-2.7, -1.3, 0.5, 1.8, 3.2])
print(np.floor(arr))    # Output: array([-3., -2.,  0.,  1.,  3.])
print(np.ceil(arr))     # Output: array([-2., -1.,  1.,  2.,  4.])
print(np.round(arr))    # Output: array([-3., -1.,  0.,  2.,  3.])
print(np.abs(arr))      # Output: array([2.7, 1.3, 0.5, 1.8, 3.2])

# Aggregate ufuncs
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr))           # Output: 21
print(np.sum(arr, axis=0))   # Output: array([5, 7, 9])
print(np.sum(arr, axis=1))   # Output: array([ 6, 15])
print(np.mean(arr))          # Output: 3.5
print(np.std(arr))           # Output: 1.707
print(np.max(arr))           # Output: 6
print(np.min(arr))           # Output: 1

# Error handling - domain error
try:
    result = np.sqrt(np.array([-1, -2, -3]))
    # This actually works but produces NaN with warning
    print(result)  # Output: array([nan, nan, nan])
except:
    pass
```

### Linear Algebra Operations

Solve linear systems, compute matrix decompositions, eigenvalues, and norms using optimized BLAS/LAPACK backends.

```python
import numpy as np
from numpy import linalg

# Solve linear system Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = linalg.solve(A, b)
print(x)
# Output: array([2., 3.])

# Verify solution
print(np.allclose(A @ x, b))  # Output: True

# Matrix inverse
A = np.array([[1, 2], [3, 4]])
A_inv = linalg.inv(A)
print(A_inv)
# Output: array([[-2. ,  1. ],
#                [ 1.5, -0.5]])

# Verify: A * A_inv = I
print(np.allclose(A @ A_inv, np.eye(2)))  # Output: True

# Singular Value Decomposition (SVD)
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = linalg.svd(A, full_matrices=False)
print(S)
# Output: array([9.525, 0.514])

# Reconstruct matrix from SVD
A_reconstructed = U @ np.diag(S) @ Vt
print(np.allclose(A, A_reconstructed))  # Output: True

# QR decomposition
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = linalg.qr(A)
print(Q.shape, R.shape)
# Output: (3, 2) (2, 2)

# Verify: Q is orthogonal
print(np.allclose(Q.T @ Q, np.eye(2)))  # Output: True

# Eigenvalues and eigenvectors
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = linalg.eig(A)
print(eigenvalues)
# Output: array([5., 2.])

# Matrix norms
A = np.array([[1, 2], [3, 4]])
print(linalg.norm(A))           # Frobenius norm: 5.477
print(linalg.norm(A, ord=1))    # 1-norm: 6.0
print(linalg.norm(A, ord=np.inf))  # inf-norm: 7.0

# Determinant
det = linalg.det(A)
print(det)
# Output: -2.0

# Matrix rank
A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
rank = linalg.matrix_rank(A)
print(rank)
# Output: 1

# Least squares solution (overdetermined system)
A = np.array([[1, 0], [1, 1], [1, 2]])
b = np.array([1, 2, 3])
x, residuals, rank, s = linalg.lstsq(A, b, rcond=None)
print(x)
# Output: array([0.667, 1.   ])
print(residuals)
# Output: array([0.167])

# Error handling - singular matrix
try:
    A_singular = np.array([[1, 2], [2, 4]])
    inv = linalg.inv(A_singular)
except linalg.LinAlgError as e:
    print("Error: Singular matrix")
```

### Fast Fourier Transform (FFT)

Compute discrete Fourier transforms for signal processing, frequency analysis, and spectral methods using PocketFFT backend.

```python
import numpy as np
from numpy import fft

# 1D FFT - basic usage
signal = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
freq_domain = fft.fft(signal)
print(freq_domain)
# Output: array([ 4.5+0.j, 2.081+1.652j, -1.581-0.726j, -1.581+0.726j, 2.081-1.652j])

# Inverse FFT
time_domain = fft.ifft(freq_domain)
print(time_domain.real)
# Output: array([ 1. ,  2. ,  1. , -1. ,  1.5])

# Real FFT (more efficient for real-valued signals)
signal_real = np.array([1, 2, 3, 4, 5, 6, 7, 8])
freq_real = fft.rfft(signal_real)
print(freq_real)
# Output: array([36.+0.j, -4.+9.656j, -4.+4.j, -4.+1.657j, -4.+0.j])

# Inverse real FFT
reconstructed = fft.irfft(freq_real)
print(reconstructed)
# Output: array([1., 2., 3., 4., 5., 6., 7., 8.])

# 2D FFT for image processing
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
freq_2d = fft.fft2(image)
print(freq_2d.shape)
# Output: (4, 4)

# Inverse 2D FFT
reconstructed_2d = fft.ifft2(freq_2d).real
print(np.allclose(image, reconstructed_2d))  # Output: True

# FFT frequencies
n = 8
sample_rate = 100  # Hz
frequencies = fft.fftfreq(n, d=1/sample_rate)
print(frequencies)
# Output: array([  0.,  12.5,  25. ,  37.5, -50. , -37.5, -25. , -12.5])

# FFT shift - move zero frequency to center
shifted = fft.fftshift(frequencies)
print(shifted)
# Output: array([-50. , -37.5, -25. , -12.5,   0. ,  12.5,  25. ,  37.5])

# Practical example: detect dominant frequency
sampling_rate = 1000  # Hz
duration = 1.0        # seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
frequency = 50        # Hz
signal = np.sin(2 * np.pi * frequency * t)

# Compute FFT
fft_result = fft.rfft(signal)
fft_freqs = fft.rfftfreq(len(signal), 1/sampling_rate)

# Find dominant frequency
dominant_idx = np.argmax(np.abs(fft_result[1:])) + 1  # Skip DC component
dominant_freq = fft_freqs[dominant_idx]
print(f"Dominant frequency: {dominant_freq} Hz")
# Output: Dominant frequency: 50.0 Hz

# N-dimensional FFT
data_3d = np.random.rand(4, 4, 4)
freq_3d = fft.fftn(data_3d)
reconstructed_3d = fft.ifftn(freq_3d).real
print(np.allclose(data_3d, reconstructed_3d))  # Output: True
```

### Random Number Generation

Generate random numbers from various distributions using the modern Generator API with configurable bit generators.

```python
import numpy as np

# Modern API - default random number generator
rng = np.random.default_rng(seed=42)

# Uniform distribution [0, 1)
uniform = rng.random(5)
print(uniform)
# Output: array([0.774, 0.438, 0.859, 0.697, 0.094])

# Normal (Gaussian) distribution
normal = rng.normal(loc=0, scale=1, size=5)
print(normal)
# Output: array([ 0.496,  0.138,  0.647, -0.235, -0.234])

# Integers in range
integers = rng.integers(low=1, high=10, size=5)
print(integers)
# Output: array([7, 4, 8, 5, 3])

# Choice - random sampling
choices = rng.choice(['A', 'B', 'C', 'D'], size=5, replace=True)
print(choices)
# Output: array(['C', 'A', 'D', 'B', 'A'])

# Choice without replacement
no_replace = rng.choice(10, size=5, replace=False)
print(no_replace)
# Output: array([7, 3, 8, 1, 5])

# Shuffle array in-place
arr = np.arange(10)
rng.shuffle(arr)
print(arr)
# Output: array([5, 0, 8, 9, 1, 6, 3, 2, 4, 7])

# Permutation - return shuffled copy
original = np.arange(5)
permuted = rng.permutation(original)
print(permuted)
# Output: array([3, 0, 4, 1, 2])

# Multiple distributions
exponential = rng.exponential(scale=2.0, size=5)
print(exponential)
# Output: array([0.542, 3.123, 1.876, 0.234, 2.987])

poisson = rng.poisson(lam=5, size=5)
print(poisson)
# Output: array([4, 6, 5, 3, 7])

binomial = rng.binomial(n=10, p=0.5, size=5)
print(binomial)
# Output: array([6, 5, 4, 7, 5])

# Multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 2]]
multivariate = rng.multivariate_normal(mean, cov, size=3)
print(multivariate)
# Output: array([[-0.285, -0.126],
#                [ 0.491,  0.672],
#                [-0.234,  1.123]])

# Different bit generators
from numpy.random import PCG64, MT19937, Philox

# PCG64 (default, good statistical properties)
rng_pcg = np.random.Generator(PCG64(seed=42))
print(rng_pcg.random(3))
# Output: array([0.774, 0.438, 0.859])

# Mersenne Twister
rng_mt = np.random.Generator(MT19937(seed=42))
print(rng_mt.random(3))
# Output: array([0.374, 0.950, 0.731])

# Philox (counter-based, good for parallel RNG)
rng_philox = np.random.Generator(Philox(seed=42))
print(rng_philox.random(3))
# Output: array([0.172, 0.816, 0.714])

# Save and restore RNG state
state = rng.bit_generator.state
random1 = rng.random(3)
rng.bit_generator.state = state  # Restore state
random2 = rng.random(3)
print(np.allclose(random1, random2))  # Output: True

# Legacy API (backward compatibility only)
# Use default_rng() for new code
np.random.seed(42)
legacy_random = np.random.rand(5)
print(legacy_random)
# Output: array([0.374, 0.950, 0.731, 0.598, 0.156])
```

### Array Statistics and Reductions

Compute statistical measures and reduce arrays along specified axes for data analysis and summarization.

```python
import numpy as np

# Basic statistics
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(np.mean(data))     # Output: 5.5
print(np.median(data))   # Output: 5.5
print(np.std(data))      # Output: 2.872
print(np.var(data))      # Output: 8.25
print(np.min(data))      # Output: 1
print(np.max(data))      # Output: 10
print(np.sum(data))      # Output: 55
print(np.prod(data))     # Output: 3628800

# Statistics along axes
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mean along columns (axis=0)
col_mean = np.mean(data_2d, axis=0)
print(col_mean)
# Output: array([4., 5., 6.])

# Mean along rows (axis=1)
row_mean = np.mean(data_2d, axis=1)
print(row_mean)
# Output: array([2., 5., 8.])

# Multiple statistics
print(np.std(data_2d, axis=0))   # Output: array([2.449, 2.449, 2.449])
print(np.sum(data_2d, axis=1))   # Output: array([ 6, 15, 24])

# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
cumsum = np.cumsum(arr)
print(cumsum)
# Output: array([ 1,  3,  6, 10, 15])

cumprod = np.cumprod(arr)
print(cumprod)
# Output: array([  1,   2,   6,  24, 120])

# Percentiles and quantiles
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(np.percentile(data, 25))   # Output: 3.25
print(np.percentile(data, 50))   # Output: 5.5
print(np.percentile(data, 75))   # Output: 7.75
print(np.quantile(data, [0.25, 0.5, 0.75]))
# Output: array([3.25, 5.5, 7.75])

# Argmin and argmax - find indices
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.argmin(arr))   # Output: 1
print(np.argmax(arr))   # Output: 5

# 2D argmin/argmax
arr_2d = np.array([[1, 5, 3], [8, 2, 9]])
print(np.argmin(arr_2d, axis=0))  # Output: array([0, 1, 0])
print(np.argmax(arr_2d, axis=1))  # Output: array([1, 2])

# Correlation and covariance
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
correlation = np.corrcoef(x, y)
print(correlation)
# Output: array([[1.   , 0.775],
#                [0.775, 1.   ]])

covariance = np.cov(x, y)
print(covariance)
# Output: array([[2.5  , 1.75 ],
#                [1.75 , 1.7  ]])

# NaN-aware functions
data_with_nan = np.array([1, 2, np.nan, 4, 5])
print(np.nanmean(data_with_nan))   # Output: 3.0
print(np.nanstd(data_with_nan))    # Output: 1.581
print(np.nansum(data_with_nan))    # Output: 12.0

# Error handling - empty array
try:
    empty = np.array([])
    mean = np.mean(empty)
    print(mean)  # Output: nan (with warning)
except:
    pass
```

### Sorting and Searching

Sort arrays, find unique elements, search sorted arrays, and perform set operations efficiently.

```python
import numpy as np

# Basic sorting
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_arr = np.sort(arr)
print(sorted_arr)
# Output: array([1, 1, 2, 3, 4, 5, 6, 9])

# Sort in-place
arr_inplace = np.array([3, 1, 4, 1, 5, 9, 2, 6])
arr_inplace.sort()
print(arr_inplace)
# Output: array([1, 1, 2, 3, 4, 5, 6, 9])

# Argsort - indices that would sort array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
indices = np.argsort(arr)
print(indices)
# Output: array([1, 3, 6, 0, 2, 4, 7, 5])
print(arr[indices])
# Output: array([1, 1, 2, 3, 4, 5, 6, 9])

# Sort 2D array along axis
arr_2d = np.array([[3, 2, 1], [6, 5, 4]])
sorted_cols = np.sort(arr_2d, axis=0)
print(sorted_cols)
# Output: array([[3, 2, 1],
#                [6, 5, 4]])

sorted_rows = np.sort(arr_2d, axis=1)
print(sorted_rows)
# Output: array([[1, 2, 3],
#                [4, 5, 6]])

# Unique elements
arr = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5])
unique = np.unique(arr)
print(unique)
# Output: array([1, 2, 3, 4, 5])

# Unique with counts
unique, counts = np.unique(arr, return_counts=True)
print(unique)   # Output: array([1, 2, 3, 4, 5])
print(counts)   # Output: array([1, 2, 3, 1, 2])

# Unique with indices
unique, indices = np.unique(arr, return_index=True)
print(indices)  # Output: array([0, 1, 3, 6, 7])

# Searchsorted - find insertion indices
sorted_arr = np.array([1, 2, 4, 5, 7, 9])
values = np.array([3, 6, 8])
insertion_indices = np.searchsorted(sorted_arr, values)
print(insertion_indices)
# Output: array([2, 4, 5])

# Binary search verification
print(sorted_arr)  # [1, 2, 4, 5, 7, 9]
# Insert 3 at index 2: [1, 2, 3, 4, 5, 7, 9]
# Insert 6 at index 4: [1, 2, 4, 5, 6, 7, 9]
# Insert 8 at index 5: [1, 2, 4, 5, 7, 8, 9]

# Set operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

intersection = np.intersect1d(a, b)
print(intersection)
# Output: array([3, 4, 5])

union = np.union1d(a, b)
print(union)
# Output: array([1, 2, 3, 4, 5, 6, 7])

difference = np.setdiff1d(a, b)
print(difference)
# Output: array([1, 2])

symmetric_diff = np.setxor1d(a, b)
print(symmetric_diff)
# Output: array([1, 2, 6, 7])

# Check if elements are in array
test_values = np.array([2, 5, 8])
in_array = np.isin(test_values, a)
print(in_array)
# Output: array([ True,  True, False])

# Partition - partial sort (faster than full sort)
arr = np.array([7, 2, 9, 1, 5, 3])
partitioned = np.partition(arr, kth=3)
print(partitioned)
# Output: array([1, 2, 3, 5, 7, 9]) or similar with 3 smallest first

# Argpartition
arr = np.array([7, 2, 9, 1, 5, 3])
indices = np.argpartition(arr, kth=3)
print(arr[indices])
# Output: similar to partition
```

### Masked Arrays for Missing Data

Handle arrays with missing or invalid values using masked arrays that preserve data structure while ignoring masked elements.

```python
import numpy as np
import numpy.ma as ma

# Create masked array
data = np.array([1, 2, 3, 4, 5])
mask = np.array([0, 0, 1, 0, 0])  # 1 = masked
masked_arr = ma.array(data, mask=mask)
print(masked_arr)
# Output: masked_array(data=[1, 2, --, 4, 5], mask=[False, False, True, False, False])

# Operations ignore masked values
print(masked_arr.mean())   # Output: 3.0 (average of 1,2,4,5)
print(masked_arr.sum())    # Output: 12
print(masked_arr.std())    # Output: 1.581

# Mask based on condition
data = np.array([1, -999, 3, 4, -999, 6])
masked_arr = ma.masked_equal(data, -999)
print(masked_arr)
# Output: masked_array(data=[1, --, 3, 4, --, 6], mask=[False, True, False, False, True, False])

# Alternative masking methods
data = np.array([1, 2, 3, 4, 5, 6])
masked_less = ma.masked_less(data, 3)
print(masked_less)
# Output: masked_array(data=[--, --, 3, 4, 5, 6], mask=[True, True, False, False, False, False])

masked_greater = ma.masked_greater(data, 4)
print(masked_greater)
# Output: masked_array(data=[1, 2, 3, 4, --, --], mask=[False, False, False, False, True, True])

masked_outside = ma.masked_outside(data, 2, 5)
print(masked_outside)
# Output: masked_array(data=[--, 2, 3, 4, 5, --], mask=[True, False, False, False, False, True])

# Mask invalid values (NaN, inf)
data_with_nan = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
masked_invalid = ma.masked_invalid(data_with_nan)
print(masked_invalid)
# Output: masked_array(data=[1.0, --, 3.0, --, 5.0], mask=[False, True, False, True, False])

# Arithmetic preserves masks
a = ma.array([1, 2, 3, 4], mask=[0, 0, 1, 0])
b = ma.array([10, 20, 30, 40], mask=[0, 1, 0, 0])
result = a + b
print(result)
# Output: masked_array(data=[11, --, --, 44], mask=[False, True, True, False])

# Fill masked values
filled = masked_arr.filled(fill_value=-1)
print(filled)
# Output: array([ 1,  2, -1,  4,  5])

# Get unmasked data
compressed = masked_arr.compressed()
print(compressed)
# Output: array([1, 2, 4, 5])

# 2D masked arrays
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_2d = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
masked_2d = ma.array(data_2d, mask=mask_2d)
print(masked_2d)
# Output: masked_array(
#   data=[[1, 2, --],
#         [4, --, 6],
#         [--, 8, 9]],
#   mask=[[False, False, True],
#         [False, True, False],
#         [True, False, False]])

# Column-wise mean (ignoring masks)
col_means = masked_2d.mean(axis=0)
print(col_means)
# Output: masked_array(data=[2.5, 5.0, 7.5], mask=[False, False, False])

# Detect masked elements
print(ma.is_masked(masked_arr))   # Output: True
print(ma.is_masked(np.array([1, 2, 3])))  # Output: False
```

### File I/O Operations

Save and load NumPy arrays in native binary format, text files, or compressed archives for data persistence.

```python
import numpy as np
import tempfile
import os

# Create temporary directory for examples
tmpdir = tempfile.mkdtemp()

# Save single array to binary file (.npy)
arr = np.array([[1, 2, 3], [4, 5, 6]])
npy_file = os.path.join(tmpdir, 'array.npy')
np.save(npy_file, arr)

# Load from binary file
loaded = np.load(npy_file)
print(loaded)
# Output: array([[1, 2, 3],
#                [4, 5, 6]])
print(np.array_equal(arr, loaded))  # Output: True

# Save multiple arrays to archive (.npz)
arr1 = np.array([1, 2, 3])
arr2 = np.array([[4, 5], [6, 7]])
npz_file = os.path.join(tmpdir, 'arrays.npz')
np.savez(npz_file, first=arr1, second=arr2)

# Load from archive
data = np.load(npz_file)
print(data['first'])
# Output: array([1, 2, 3])
print(data['second'])
# Output: array([[4, 5],
#                [6, 7]])
data.close()

# Save compressed archive
npz_compressed = os.path.join(tmpdir, 'arrays_compressed.npz')
np.savez_compressed(npz_compressed, a=arr1, b=arr2)

# Save to text file
txt_file = os.path.join(tmpdir, 'array.txt')
arr_text = np.array([[1.5, 2.7, 3.2], [4.1, 5.9, 6.3]])
np.savetxt(txt_file, arr_text, fmt='%.2f', delimiter=',')

# Load from text file
loaded_text = np.loadtxt(txt_file, delimiter=',')
print(loaded_text)
# Output: array([[1.5, 2.7, 3.2],
#                [4.1, 5.9, 6.3]])

# Save with custom format
txt_custom = os.path.join(tmpdir, 'custom.txt')
np.savetxt(txt_custom, arr_text, fmt='%d', delimiter=' | ', header='Col1 | Col2 | Col3')

# Load with skip header
loaded_custom = np.loadtxt(txt_custom, delimiter=' | ', skiprows=1)
print(loaded_custom)

# genfromtxt - more flexible text loading
csv_file = os.path.join(tmpdir, 'data.csv')
with open(csv_file, 'w') as f:
    f.write('# Comment line\n')
    f.write('1,2,3\n')
    f.write('4,5,6\n')
    f.write('7,NA,9\n')  # Missing value

# Load with genfromtxt
data_csv = np.genfromtxt(csv_file, delimiter=',', skip_header=1,
                         missing_values='NA', filling_values=0)
print(data_csv)
# Output: array([[1., 2., 3.],
#                [4., 5., 6.],
#                [7., 0., 9.]])

# Binary files - tofile/fromfile
bin_file = os.path.join(tmpdir, 'binary.dat')
arr_bin = np.array([1, 2, 3, 4, 5], dtype=np.int32)
arr_bin.tofile(bin_file)

loaded_bin = np.fromfile(bin_file, dtype=np.int32)
print(loaded_bin)
# Output: array([1, 2, 3, 4, 5])

# Memory-mapped files (for large datasets)
mmap_file = os.path.join(tmpdir, 'memmap.dat')
mmap_arr = np.memmap(mmap_file, dtype='float32', mode='w+', shape=(3, 4))
mmap_arr[:] = np.arange(12).reshape(3, 4)
mmap_arr.flush()

# Load memory-mapped file
mmap_loaded = np.memmap(mmap_file, dtype='float32', mode='r', shape=(3, 4))
print(mmap_loaded)
# Output: array([[ 0.,  1.,  2.,  3.],
#                [ 4.,  5.,  6.,  7.],
#                [ 8.,  9., 10., 11.]])

# Error handling - file not found
try:
    missing = np.load('nonexistent.npy')
except FileNotFoundError as e:
    print("Error: File not found")

# Clean up
import shutil
shutil.rmtree(tmpdir)
```

### Advanced Indexing with where and select

Conditionally select elements, replace values, or apply multiple conditions using where, select, and choose functions.

```python
import numpy as np

# np.where - element-wise conditional
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.where(arr > 3, arr * 2, arr)
print(result)
# Output: array([ 1,  2,  3,  8, 10, 12])

# where with condition only (returns indices)
indices = np.where(arr > 3)
print(indices)
# Output: (array([3, 4, 5]),)
print(arr[indices])
# Output: array([4, 5, 6])

# 2D where
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices, col_indices = np.where(arr_2d > 5)
print(row_indices)    # Output: array([1, 2, 2, 2])
print(col_indices)    # Output: array([2, 0, 1, 2])
print(arr_2d[row_indices, col_indices])
# Output: array([6, 7, 8, 9])

# np.select - multiple conditions
x = np.arange(10)
conditions = [
    x < 3,
    (x >= 3) & (x < 7),
    x >= 7
]
choices = [
    x * 0,      # < 3: set to 0
    x * 10,     # 3-6: multiply by 10
    x * 100     # >= 7: multiply by 100
]
result = np.select(conditions, choices)
print(result)
# Output: array([  0,   0,   0,  30,  40,  50,  60, 700, 800, 900])

# np.choose - select from multiple arrays
choices = [
    np.array([10, 20, 30]),
    np.array([40, 50, 60]),
    np.array([70, 80, 90])
]
indices = np.array([0, 1, 2])
result = np.choose(indices, choices)
print(result)
# Output: array([10, 50, 90])

# Practical example: clip values
data = np.array([-5, 0, 5, 10, 15, 20, 25])
clipped = np.where(data < 0, 0, np.where(data > 20, 20, data))
print(clipped)
# Output: array([ 0,  0,  5, 10, 15, 20, 20])

# Alternative using clip function
clipped_alt = np.clip(data, 0, 20)
print(clipped_alt)
# Output: array([ 0,  0,  5, 10, 15, 20, 20])

# Replace NaN values conditionally
data_with_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
result = np.where(np.isnan(data_with_nan), 0.0, data_with_nan)
print(result)
# Output: array([1., 0., 3., 0., 5.])

# Complex conditional logic
temperature = np.array([15, 20, 25, 30, 35, 40])
conditions = [
    temperature < 20,
    (temperature >= 20) & (temperature < 30),
    temperature >= 30
]
labels = ['Cold', 'Moderate', 'Hot']
result = np.select(conditions, labels)
print(result)
# Output: array(['Cold', 'Moderate', 'Moderate', 'Hot', 'Hot', 'Hot'])

# Piecewise function
x = np.linspace(-2, 2, 9)
conditions = [x < 0, x >= 0]
choices = [x**2, x**3]
result = np.select(conditions, choices)
print(result)
# Output: array([4., 1., 0., 0., 0., 0., 1., 8.])
```

### Polynomial Operations

Create, evaluate, fit, and manipulate polynomials using multiple orthogonal bases including power, Chebyshev, and Legendre.

```python
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre

# Create polynomial: 1 + 2x + 3x^2
p = Polynomial([1, 2, 3])
print(p)
# Output: 1.0 + 2.0·x + 3.0·x²

# Evaluate polynomial
x = np.array([0, 1, 2, 3])
values = p(x)
print(values)
# Output: array([ 1.,  6., 17., 34.])

# Polynomial arithmetic
p1 = Polynomial([1, 2])      # 1 + 2x
p2 = Polynomial([3, 4])      # 3 + 4x

print(p1 + p2)  # Output: 4.0 + 6.0·x
print(p1 * p2)  # Output: 3.0 + 10.0·x + 8.0·x²
print(p1 ** 2)  # Output: 1.0 + 4.0·x + 4.0·x²

# Polynomial derivative
p = Polynomial([1, 2, 3, 4])  # 1 + 2x + 3x² + 4x³
dp = p.deriv()
print(dp)
# Output: 2.0 + 6.0·x + 12.0·x²

# Polynomial integral
integral = p.integ()
print(integral)
# Output: 0.0 + 1.0·x + 1.0·x² + 1.0·x³ + 1.0·x⁴

# Find roots
p = Polynomial([1, -5, 6])   # 1 - 5x + 6x² = (1-2x)(1-3x)
roots = p.roots()
print(roots)
# Output: array([0.333, 0.5])

# Polynomial fitting
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1, 3, 7, 13, 21])  # Roughly y = 1 + 2x
p_fit = Polynomial.fit(x_data, y_data, deg=2)
print(p_fit)
# Output: approximately 1.0 + 2.0·x + 0.5·x²

# Evaluate fitted polynomial
x_test = np.array([0.5, 1.5, 2.5])
y_pred = p_fit(x_test)
print(y_pred)
# Output: array([ 2.125,  5.125, 10.125])

# Chebyshev polynomials (better numerical properties)
c = Chebyshev([1, 2, 3])
print(c)
# Output: 1.0 + 2.0·T₁(x) + 3.0·T₂(x)

# Evaluate Chebyshev polynomial
x_cheb = np.linspace(-1, 1, 5)
values_cheb = c(x_cheb)
print(values_cheb)
# Output: array([2., 3.5, 1., 3.5, 2.])

# Convert between bases
p = Polynomial([1, 2, 3])
c = p.convert(kind=Chebyshev)
print(c)
# Output: Chebyshev coefficients

# Legendre polynomials
leg = Legendre([1, 2, 3])
print(leg)
# Output: 1.0 + 2.0·P₁(x) + 3.0·P₂(x)

# Polynomial interpolation
x_points = np.array([0, 1, 2, 3])
y_points = np.array([1, 2, 0, 3])
p_interp = Polynomial.fit(x_points, y_points, deg=len(x_points)-1)

# Verify interpolation
print(np.allclose(p_interp(x_points), y_points))  # Output: True

# Chebyshev interpolation (more stable)
x_cheb_points = np.cos(np.pi * np.arange(5) / 4)  # Chebyshev nodes
y_cheb_points = np.exp(x_cheb_points)
c_interp = Chebyshev.fit(x_cheb_points, y_cheb_points, deg=4)

# Legacy polynomial interface (numpy.poly1d)
p_legacy = np.poly1d([3, 2, 1])  # 3x² + 2x + 1
print(p_legacy)
# Output: 3 x^2 + 2 x + 1
print(p_legacy(2))  # Output: 17
```

### Einstein Summation Convention

Perform complex tensor operations efficiently using Einstein summation notation for matrix multiplication, traces, and more.

```python
import numpy as np

# Matrix multiplication using einsum
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Equivalent to A @ B or np.dot(A, B)
C = np.einsum('ij,jk->ik', A, B)
print(C)
# Output: array([[19, 22],
#                [43, 50]])

# Element-wise multiplication
result = np.einsum('ij,ij->ij', A, B)
print(result)
# Output: array([[ 5, 12],
#                [21, 32]])

# Matrix trace (sum of diagonal)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
trace = np.einsum('ii->', A)
print(trace)
# Output: 15 (1 + 5 + 9)

# Transpose
A = np.array([[1, 2, 3], [4, 5, 6]])
transposed = np.einsum('ij->ji', A)
print(transposed)
# Output: array([[1, 4],
#                [2, 5],
#                [3, 6]])

# Sum along axis
A = np.array([[1, 2, 3], [4, 5, 6]])
row_sum = np.einsum('ij->i', A)
print(row_sum)
# Output: array([ 6, 15])

col_sum = np.einsum('ij->j', A)
print(col_sum)
# Output: array([5, 7, 9])

# Batch matrix multiplication
# A: (batch, n, k), B: (batch, k, m) -> C: (batch, n, m)
A_batch = np.random.rand(3, 2, 4)
B_batch = np.random.rand(3, 4, 2)
C_batch = np.einsum('bnk,bkm->bnm', A_batch, B_batch)
print(C_batch.shape)
# Output: (3, 2, 2)

# Outer product
a = np.array([1, 2, 3])
b = np.array([4, 5])
outer = np.einsum('i,j->ij', a, b)
print(outer)
# Output: array([[ 4,  5],
#                [ 8, 10],
#                [12, 15]])

# Inner product (dot product)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner = np.einsum('i,i->', a, b)
print(inner)
# Output: 32 (1*4 + 2*5 + 3*6)

# Batch dot product
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 0], [0, 1], [1, 1]])
batch_dot = np.einsum('ij,ij->i', A, B)
print(batch_dot)
# Output: array([ 1,  7, 11])

# Frobenius inner product
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
frobenius = np.einsum('ij,ij->', A, B)
print(frobenius)
# Output: 70

# Tensor contraction
# Contract 3D tensors along last dimension
A = np.random.rand(2, 3, 4)
B = np.random.rand(2, 4, 5)
result = np.einsum('ijk,ikl->ijl', A, B)
print(result.shape)
# Output: (2, 3, 5)

# Diagonal extraction
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
diag = np.einsum('ii->i', A)
print(diag)
# Output: array([1, 5, 9])

# Hadamard product with broadcasting
A = np.random.rand(3, 1, 4)
B = np.random.rand(1, 2, 4)
result = np.einsum('ijk,ljk->iljk', A, B)
print(result.shape)
# Output: (3, 1, 2, 4)
```

### Data Type Management

Work with NumPy's rich type system including integers, floats, complex numbers, datetimes, and structured arrays.

```python
import numpy as np

# Basic data types
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)
arr_bool = np.array([True, False, True], dtype=np.bool_)

print(arr_int.dtype)      # Output: int32
print(arr_float.dtype)    # Output: float64
print(arr_complex.dtype)  # Output: complex128

# Type information
int_info = np.iinfo(np.int32)
print(f"int32 range: {int_info.min} to {int_info.max}")
# Output: int32 range: -2147483648 to 2147483647

float_info = np.finfo(np.float64)
print(f"float64 precision: {float_info.precision} decimal digits")
# Output: float64 precision: 15 decimal digits

# Type conversion
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = arr_int.astype(np.float64)
print(arr_float)
# Output: array([1., 2., 3.])

# Automatic type promotion
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
result = int_arr + float_arr
print(result.dtype)
# Output: float64

# Structured arrays (record arrays)
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
data = np.array([
    ('Alice', 25, 55.5),
    ('Bob', 30, 75.0),
    ('Charlie', 35, 80.2)
], dtype=dt)

print(data['name'])
# Output: array(['Alice', 'Bob', 'Charlie'])
print(data['age'])
# Output: array([25, 30, 35])

# Access individual record
print(data[0])
# Output: ('Alice', 25, 55.5)
print(data[0]['name'])
# Output: 'Alice'

# Datetime arrays
dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64')
print(dates)
# Output: array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')

# Datetime arithmetic
dates_shifted = dates + np.timedelta64(7, 'D')
print(dates_shifted)
# Output: array(['2024-01-08', '2024-01-09', '2024-01-10'], dtype='datetime64[D]')

# Time differences
start = np.datetime64('2024-01-01')
end = np.datetime64('2024-12-31')
duration = end - start
print(duration)
# Output: 365 days

# Datetime ranges
date_range = np.arange('2024-01', '2024-04', dtype='datetime64[M]')
print(date_range)
# Output: array(['2024-01', '2024-02', '2024-03'], dtype='datetime64[M]')

# String arrays
str_arr = np.array(['hello', 'world', 'numpy'], dtype='U10')
print(str_arr.dtype)
# Output: <U10 (Unicode string, max 10 chars)

# Object arrays (Python objects)
obj_arr = np.array([{'a': 1}, {'b': 2}, [1, 2, 3]], dtype=object)
print(obj_arr)
# Output: array([{'a': 1}, {'b': 2}, [1, 2, 3]], dtype=object)

# Byte order
arr = np.array([1, 2, 3], dtype='>i4')  # Big-endian int32
print(arr.dtype.byteorder)
# Output: > (big-endian)

arr_little = arr.astype('<i4')  # Convert to little-endian
print(arr_little.dtype.byteorder)
# Output: < (little-endian)

# Custom structured dtype with nested fields
dt_complex = np.dtype([
    ('id', 'i4'),
    ('position', [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
    ('velocity', [('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')])
])

particles = np.zeros(3, dtype=dt_complex)
particles[0]['id'] = 1
particles[0]['position']['x'] = 1.0
particles[0]['position']['y'] = 2.0
print(particles[0])
# Output: (1, (1., 2., 0.), (0., 0., 0.))
```

---

## Summary

NumPy serves as the cornerstone of scientific computing in Python, providing high-performance multidimensional arrays and a comprehensive mathematical function library. Its primary use cases include numerical analysis, data preprocessing for machine learning pipelines, scientific simulations, signal processing with FFT operations, statistical analysis, and linear algebra computations in computer graphics and physics simulations. The library excels at handling large datasets efficiently through vectorized operations that eliminate Python loops, broadcasting semantics that enable intuitive array arithmetic, and memory-efficient views that avoid unnecessary copying.

NumPy integrates seamlessly into the broader scientific Python ecosystem, serving as the foundation for pandas (data analysis), scikit-learn (machine learning), SciPy (scientific computing), matplotlib (visualization), and TensorFlow/PyTorch (deep learning). Common integration patterns include using NumPy arrays as the universal data interchange format between libraries, leveraging C/Fortran extensions via F2PY for performance-critical code, accessing optimized BLAS/LAPACK implementations through configuration, and utilizing memory-mapped files for out-of-core computation on datasets larger than RAM. The library's consistent API, comprehensive documentation, and battle-tested stability make it an essential tool for any Python-based numerical computing workflow.
