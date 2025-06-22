# NumPy for Data Processing

## Table of Contents
1. [Introduction to NumPy](#introduction-to-numpy)
2. [Array Creation and Basic Operations](#array-creation-and-basic-operations)
3. [Array Indexing and Slicing](#array-indexing-and-slicing)
4. [Array Manipulation](#array-manipulation)
5. [Mathematical Operations](#mathematical-operations)
6. [Broadcasting](#broadcasting)
7. [Performance Considerations](#performance-considerations)

## Introduction to NumPy
NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides:
- A powerful N-dimensional array object
- Sophisticated (broadcasting) functions
- Tools for integrating C/C++ and Fortran code
- Useful linear algebra, Fourier transform, and random number capabilities

### Why NumPy?
- **Performance**: NumPy arrays are more efficient than Python lists for numerical operations
- **Functionality**: Rich set of mathematical functions
- **Interoperability**: Works well with other scientific libraries

## Array Creation and Basic Operations

### Creating Arrays
```python
import numpy as np

# From Python lists
arr1 = np.array([1, 2, 3, 4, 5])

# Arrays of zeros and ones
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3, 4))  # 2x3x4 array of ones

# Identity matrix
identity = np.eye(3)  # 3x3 identity matrix

# Arange (similar to range but produces arrays)
arr = np.arange(10)  # array([0, 1, 2, ..., 9])

# Linspace (evenly spaced numbers over a range)
arr = np.linspace(0, 1, 5)  # array([0., 0.25, 0.5, 0.75, 1.])
```

### Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Array shape:", arr.shape)  # (2, 3)
print("Number of dimensions:", arr.ndim)  # 2
print("Number of elements:", arr.size)  # 6
print("Data type:", arr.dtype)  # int64
print("Item size:", arr.itemsize, "bytes")  # 8
print("Total size:", arr.nbytes, "bytes")  # 48
```

## Array Indexing and Slicing

### Basic Indexing
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element
print(arr[0, 1])  # 2

# Slicing
print(arr[0:2, 1:3])  # Rows 0-1, columns 1-2
# array([[2, 3],
#        [5, 6]])

```

### Boolean Indexing
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Boolean indexing
print(arr[arr > 5])  # [6 7 8 9]

# Multiple conditions
condition = (arr > 2) & (arr < 8)
print(arr[condition])  # [3 4 5 6 7]
```

## Array Manipulation

### Reshaping
```python
arr = np.arange(12)

# Reshape to 3x4
reshaped = arr.reshape(3, 4)


# Flatten array
flattened = reshaped.flatten()
```

### Stacking Arrays
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack
print(np.vstack((a, b)))
# array([[1, 2, 3],
#        [4, 5, 6]])

# Horizontal stack
print(np.hstack((a, b)))  # array([1, 2, 3, 4, 5, 6])
```

## Mathematical Operations

### Basic Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print(a + b)  # [5 7 9]
print(a * 2)  # [2 4 6]
print(np.sin(a))  # Element-wise sine
```

### Matrix Operations
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print(np.dot(a, b))
# array([[19, 22],
#        [43, 50]])

# Alternative matrix multiplication
print(a @ b)
```

## Broadcasting

Broadcasting allows operations on arrays of different shapes.

```python
# Adding a scalar to an array
a = np.array([1, 2, 3])
print(a + 5)  # [6 7 8]

# Adding arrays of different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 0, 1])
print(a + b)
# array([[2, 2, 4],
#        [5, 5, 7]])
```

## Performance Considerations

### Vectorized Operations
```python
# Slow: Using Python loops
def slow_sum(arr):
    result = 0
    for num in arr:
        result += num
    return result

# Fast: Using NumPy's sum
fast_sum = np.sum(arr)
```

### Memory Efficiency
```python
# Create a view (no copy)
a = np.array([1, 2, 3, 4])
b = a[1:3]  # View, not a copy

# Create a copy explicitly
c = a[1:3].copy()
```

## Practice Exercises
1. Create a 5x5 array with random values and find the minimum and maximum values.
2. Multiply a 3x3 matrix by its transpose.
3. Normalize a 1D array to have zero mean and unit variance.
4. Create a function that computes the Euclidean distance between two points.
5. Implement matrix multiplication without using `np.dot()` or `@`.

---
Next: [Data Cleaning and Preprocessing](./04_data_cleaning.md)
