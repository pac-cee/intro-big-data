# Mastering Python Functions and Modules

## Table of Contents
1. [Defining Functions](#defining-functions)
   - [Basic Function Syntax](#basic-function-syntax)
   - [Function Documentation](#function-documentation)
   - [Type Hints and Annotations](#type-hints-and-annotations)

2. [Function Arguments and Parameters](#function-arguments-and-parameters)
   - [Positional and Keyword Arguments](#positional-and-keyword-arguments)
   - [Default Parameter Values](#default-parameter-values)
   - [Variable-Length Arguments](#variable-length-arguments)
   - [Keyword-Only and Positional-Only Parameters](#keyword-only-and-positional-only-parameters)
   - [Function Parameter Best Practices](#function-parameter-best-practices)

3. [Return Values and Multiple Returns](#return-values-and-multiple-returns)
   - [Returning Multiple Values](#returning-multiple-values)
   - [Returning Functions (Closures)](#returning-functions-closures)
   - [Generator Functions](#generator-functions)

4. [Advanced Function Concepts](#advanced-function-concepts)
   - [First-Class Functions](#first-class-functions)
   - [Lambda Functions](#lambda-functions)
   - [Function Decorators](#function-decorators)
   - [Function Caching with `functools.lru_cache`](#function-caching)

5. [Variable Scope and Namespaces](#variable-scope-and-namespaces)
   - [LEGB Rule](#legb-rule)
   - [Global and Nonlocal Keywords](#global-and-nonlocal-keywords)
   - [Closures and Factory Functions](#closures-and-factory-functions)

6. [Modules and Packages](#modules-and-packages)
   - [Creating and Importing Modules](#creating-and-importing-modules)
   - [The Module Search Path](#the-module-search-path)
   - [Package Structure and `__init__.py`](#package-structure)
   - [Relative vs Absolute Imports](#relative-vs-absolute-imports)
   - [Installing Third-Party Packages](#installing-third-party-packages)

7. [The Python Standard Library](#the-python-standard-library)
   - [Commonly Used Standard Library Modules](#commonly-used-standard-library-modules)
   - [Working with Dates and Times](#working-with-dates-and-times)
   - [File and Directory Operations](#file-and-directory-operations)
   - [Working with JSON and CSV](#working-with-json-and-csv)

8. [The `__name__` Variable](#the-__name__-variable)
   - [Script vs Module Execution](#script-vs-module-execution)
   - [Common Patterns](#common-patterns)

9. [Virtual Environments and Dependency Management](#virtual-environments-and-dependency-management)
   - [Creating and Managing Virtual Environments](#creating-and-managing-virtual-environments)
   - [requirements.txt and pip-tools](#requirements-management)
   - [Dependency Resolution](#dependency-resolution)

10. [Best Practices](#best-practices)
    - [Function Design Principles](#function-design-principles)
    - [Module Organization](#module-organization)
    - [Documentation and Type Hints](#documentation-and-type-hints)
    - [Testing and Debugging](#testing-and-debugging)

11. [Practice Exercises](#practice-exercises)
12. [Additional Resources](#additional-resources)

# Defining Functions

## Basic Function Syntax

### Function Definition
```python
def function_name(parameters):
    """Docstring (optional but highly recommended)"""
    # Function body
    # ...
    return value  # Optional
```

### Simple Example
```python
def greet(name: str) -> None:
    """
    Greet a person by name.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        None: This function only prints a greeting
    """
    print(f"Hello, {name}!")

greet("Alice")  # Output: Hello, Alice!
```

## Function Documentation

### Docstring Formats
Python supports several docstring formats. The most common are:

1. **Google Style** (recommended for most projects):
```python
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
        
    Returns:
        The area of the rectangle (length * width)
        
    Raises:
        ValueError: If either length or width is negative
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions cannot be negative")
    return length * width
```

2. **NumPy/SciPy Style**:
```python
def calculate_area(length, width):
    """Calculate the area of a rectangle.
    
    Parameters
    ----------
    length : float
        The length of the rectangle
    width : float
        The width of the rectangle
        
    Returns
    -------
    float
        The area of the rectangle (length * width)
        
    Raises
    ------
    ValueError
        If either length or width is negative
    """
    if length < 0 or width < 0:
        raise ValueError("Dimensions cannot be negative")
    return length * width
```

## Type Hints and Annotations

Python 3.5+ supports type hints through the typing module:

```python
from typing import List, Dict, Tuple, Optional, Union, Callable

def process_data(
    data: List[Dict[str, Union[int, float]]],
    callback: Optional[Callable[[Dict], bool]] = None
) -> Tuple[List[Dict], int]:
    """Process a list of data dictionaries with an optional filter callback.
    
    Args:
        data: List of dictionaries containing numeric data
        callback: Optional function to filter data. Should return True to keep item.
        
    Returns:
        A tuple containing:
        - Filtered list of dictionaries
        - Number of items processed
    """
    if callback is None:
        return data, len(data)
        
    filtered = [item for item in data if callback(item)]
    return filtered, len(data)

# Using the function
result = process_data([{"value": 1}, {"value": 2.5}], lambda x: x["value"] > 1)
print(result)  # ([{'value': 2.5}], 2)
```

### Type Aliases
For complex types, you can create aliases:

```python
from typing import Dict, List, Tuple

# Type aliases
UserId = int
UserData = Dict[str, str]
UserList = List[Tuple[UserId, UserData]]

def process_users(users: UserList) -> None:
    for user_id, data in users:
        print(f"Processing user {user_id}: {data}")
```

### Using `typing` Module

```python
from typing import (
    List, Dict, Set, Tuple, Optional, Union, Any,
    Callable, TypeVar, Generic, Iterable, Iterator
)

# Commonly used types:
# - List[str] - List of strings
# - Dict[str, int] - Dictionary with string keys and integer values
# - Optional[str] - Either str or None
# - Union[str, int] - Either string or integer
# - Callable[[int, str], bool] - Function taking (int, str) and returning bool
# - Iterable[float] - Any iterable containing floats
# - TypeVar - For generic types

T = TypeVar('T')

def first_item(items: List[T]) -> Optional[T]:
    """Return the first item or None if the list is empty."""
    return items[0] if items else None
```

## Function Arguments and Parameters

### Positional and Keyword Arguments

Python functions can be called using positional or keyword arguments:

```python
def describe_pet(pet_name: str, animal_type: str = 'dog', age: Optional[int] = None) -> str:
    """Return a formatted string describing a pet."""
    description = f"{pet_name.title()} is a"
    if age is not None:
        description += f" {age}-year-old"
    description += f" {animal_type}."
    return description

# Positional arguments (order matters)
print(describe_pet('willie', 'hamster', 2))
# Output: 'Willie is a 2-year-old hamster.'

# Keyword arguments (order doesn't matter)
print(describe_pet(animal_type='dog', pet_name='buddy', age=4))
# Output: 'Buddy is a 4-year-old dog.'

# Mixed (positional must come before keyword arguments)
print(describe_pet('whiskers', age=3, animal_type='cat'))
# Output: 'Whiskers is a 3-year-old cat.'

# Using default values
print(describe_pet('fido'))  # Uses default animal_type='dog'
# Output: 'Fido is a dog.'
```

### Default Parameter Values

Parameters can have default values, making them optional. Important considerations:

```python
# Good practice: Use immutable default values or None
def add_item(item: Any, items: Optional[list] = None) -> list:
    """Add an item to a list, creating a new list if none provided."""
    if items is None:
        items = []
    items.append(item)
    return items

# Bad practice: Mutable default argument (anti-pattern)
def add_to_cart(item: Any, cart: list = []):  # Don't do this!
    cart.append(item)
    return cart

# Why it's bad:
carts = []
for item in [1, 2, 3]:
    carts.append(add_to_cart(item))  # All carts reference the same list!
print(carts)  # [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

# Good:
carts = []
for item in [1, 2, 3]:
    carts.append(add_item(item))  # Each cart is a new list
print(carts)  # [[1], [2], [3]]
```

### Variable-Length Arguments

Python allows functions to accept any number of positional (`*args`) and keyword (`**kwargs`) arguments:

```python
from datetime import datetime
from typing import Any, Dict, List, Optional

def log_message(level: str, *args: Any, **kwargs: Any) -> None:
    """Log a message with variable arguments and metadata.
    
    Args:
        level: Log level (e.g., 'info', 'warning', 'error')
        *args: Message parts to be joined with spaces
        **kwargs: Additional metadata to include in the log
    """
    timestamp = kwargs.pop('timestamp', None) or datetime.now().isoformat()
    message = ' '.join(str(arg) for arg in args)
    
    print(f"[{timestamp}] {level.upper()}: {message}")
    
    # Log additional metadata if provided
    if kwargs:
        print("Additional info:")
        for key, value in kwargs.items():
            print(f"- {key}: {value}")

# Usage
log_message("info", "System started")
log_message("warning", "Disk space low", "on server", 
           timestamp="2023-05-15T10:00:00",
           disk_usage="95%", 
           server="web01")
```

### Keyword-Only and Positional-Only Parameters

Python 3.0+ allows specifying parameters that must be passed as keyword arguments or positional arguments:

```python
def process_data(
    data: list[float],  # Regular parameter (can be positional or keyword)
    /,                  # Parameters before / are positional-only
    *,                  # Parameters after * are keyword-only
    threshold: float = 0.0,
    normalize: bool = True,
    **options: Any      # Collects remaining keyword arguments
) -> list[float]:
    """Process data with positional-only and keyword-only parameters.
    
    Args:
        data: List of numerical values to process
        threshold: Minimum value to include (default: 0.0)
        normalize: Whether to normalize the data (default: True)
        **options: Additional processing options
            - max_value: Optional maximum value for clipping
            - fill_na: Value to use for None/NaN values
    
    Returns:
        Processed list of floats
    """
    # Handle None/NaN values
    fill_value = options.get('fill_na', 0.0)
    processed = [fill_value if x is None else x for x in data]
    
    # Apply threshold
    processed = [x for x in processed if x > threshold]
    
    # Apply max value if specified
    if 'max_value' in options:
        processed = [min(x, options['max_value']) for x in processed]
    
    # Normalize if requested
    if normalize and processed:
        max_val = max(processed)
        if max_val > 0:
            processed = [x / max_val for x in processed]
    
    return processed

# Valid calls
result1 = process_data([1, 2, 3, None], threshold=0.5, normalize=True)
result2 = process_data([1, 2, 3], normalize=False, max_value=2.5)

# Invalid calls
# process_data(data=[1, 2, 3])  # Error: data is positional-only
# process_data([1, 2, 3], 0.5)  # Error: threshold must be keyword argument
```

### Parameter Passing Techniques

Python uses different ways to pass arguments to functions:

1. **Pass by Assignment**: Python's model where arguments are passed by object reference
2. **Mutable vs Immutable Arguments**: How different types behave when modified inside functions
3. **Argument Unpacking**: Using `*` and `**` to unpack sequences and dictionaries

```python
def demonstrate_passing(a: int, b: list, c: dict) -> None:
    """Demonstrate how different types are passed to functions."""
    a = 10  # Creates a new local variable, doesn't affect the original
    b.append(4)  # Modifies the original list
    c['new_key'] = 'value'  # Modifies the original dict
    c = {'new': 'dict'}  # Creates a new local variable

# Immutable (int, str, tuple, etc.)
x = 5
# Mutable (list, dict, set, etc.)
my_list = [1, 2, 3]
my_dict = {'key': 'value'}

demonstrate_passing(x, my_list, my_dict)
print(f"x: {x}")           # x: 5 (unchanged)
print(f"my_list: {my_list}") # my_list: [1, 2, 3, 4] (modified)
print(f"my_dict: {my_dict}") # my_dict: {'key': 'value', 'new_key': 'value'} (modified)

# Argument unpacking
def print_coordinates(x: int, y: int, z: int) -> None:
    print(f"Coordinates: ({x}, {y}, {z})")

point = (10, 20, 30)
print_coordinates(*point)  # Unpack tuple

params = {'x': 1, 'y': 2, 'z': 3}
print_coordinates(**params)  # Unpack dictionary
```

## Return Values and Multiple Returns

### Returning Single Values

Functions in Python can return any type of object. When no return statement is specified, the function returns `None`.

```python
def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

result = add(3, 5)  # Returns 8
print(result)  # Output: 8

# Functions without a return statement return None
def no_return():
    print("This function returns None")
    
value = no_return()
print(value)  # Output: None
```

### Returning Multiple Values

Python functions can return multiple values as a tuple, which can be unpacked into multiple variables:

```python
def get_statistics(numbers: list[float]) -> tuple[float, float, float, int]:
    """Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        A tuple containing (mean, median, standard_deviation, count)
    """
    if not numbers:
        return 0.0, 0.0, 0.0, 0
        
    count = len(numbers)
    mean = sum(numbers) / count
    
    # Calculate median
    sorted_nums = sorted(numbers)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    else:
        median = sorted_nums[mid]
    
    # Calculate standard deviation
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    std_dev = (sum(squared_diffs) / count) ** 0.5
    
    return mean, median, std_dev, count

# Unpacking returned values
data = [1.2, 2.3, 3.4, 4.5, 5.6]
mean, median, std_dev, count = get_statistics(data)
print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}, Count: {count}")

# Or capture as a single tuple
stats = get_statistics(data)
print(f"Statistics: {stats}")
```

### Returning Functions (Closures)

Functions can return other functions, creating closures that remember values from their enclosing scope:

```python
def make_multiplier(factor: float) -> callable:
    """Return a function that multiplies its input by a factor."""
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier

# Create specialized functions
double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# More complex example with state
from typing import Callable, List

def create_counter() -> tuple[Callable[[], int], Callable[[], None]]:
    """Create a counter with increment and reset functionality.
    
    Returns:
        A tuple of (get_count, increment) functions
    """
    count = 0  # This variable is part of the closure
    
    def get_count() -> int:
        return count
        
    def increment() -> None:
        nonlocal count
        count += 1
    
    return get_count, increment

# Usage
counter, increment = create_counter()
print(counter())  # 0
increment()
increment()
print(counter())  # 2
```

### Generator Functions

Functions that use `yield` return a generator iterator, allowing for lazy evaluation:

```python
def fibonacci_sequence(n: int):
    """Generate the first n numbers in the Fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using the generator
for num in fibonacci_sequence(10):
    print(num, end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34

# Generator expressions (similar to list comprehensions)
squares = (x**2 for x in range(10))
print(sum(squares))  # 285

# Chaining generators
def square(nums):
    for num in nums:
        yield num ** 2

def even(nums):
    for num in nums:
        if num % 2 == 0:
            yield num

# Process numbers through multiple generators
numbers = range(10)
result = sum(square(even(numbers)))
print(result)  # 120 (0 + 4 + 16 + 36 + 64)
```

### Advanced Return Patterns

#### Returning Multiple Types with Union
```python
from typing import Union, Tuple

def parse_number(input_str: str) -> Union[float, Tuple[None, str]]:
    """Parse a string to a float or return an error message."""
    try:
        return float(input_str)
    except ValueError:
        return None, f"Could not convert '{input_str}' to a number"

# Usage
result = parse_number("3.14")
if isinstance(result, tuple):
    print(f"Error: {result[1]}")
else:
    print(f"Success: {result}")
```

#### Using Dataclasses for Complex Returns
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class OperationResult:
    success: bool
    value: Optional[float] = None
    error: Optional[str] = None

def divide(a: float, b: float) -> OperationResult:
    """Divide two numbers with proper error handling."""
    if b == 0:
        return OperationResult(False, error="Cannot divide by zero")
    return OperationResult(True, value=a / b)

# Usage
result = divide(10, 2)
if result.success:
    print(f"Result: {result.value}")
else:
    print(f"Error: {result.error}")
```

#### Context Managers with `__enter__` and `__exit__`
```python
from typing import IO, Optional

class FileHandler:
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file: Optional[IO] = None
    
    def __enter__(self) -> IO:
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileHandler('example.txt', 'w') as f:
    f.write('Hello, context manager!')
```

### Best Practices for Return Values

1. **Be Consistent**: Return the same type of value in all code paths of a function
2. **Use Exceptions for Errors**: Raise exceptions for error conditions rather than returning error codes
3. **Consider Returning `None`**: For functions that perform actions rather than calculations
4. **Use Type Hints**: Clearly document the return type using type annotations
5. **Keep Return Types Simple**: Avoid returning complex nested structures when possible
6. **Document Return Values**: Clearly document what the function returns, especially for complex types
7. **Consider Using Data Classes**: For functions that return multiple related values, consider using a data class or named tuple
```

### Returning a Dictionary
```python
def build_person(first_name, last_name, age=None):
    person = {'first': first_name, 'last': last_name}
    if age:
        person['age'] = age
    return person

musician = build_person('jimi', 'hendrix', age=27)
print(musician)
```

## Variable Scope

### Global vs Local Variables
```python
def test_scope():
    x = "local x"
    print(x)  # local x

x = "global x"
test_scope()
print(x)  # global x

# Using global keyword
def change_global():
    global x
    x = "changed global x"

change_global()
print(x)  # changed global x

# Nonlocal Variables
def outer():
    x = "outer x"
    
    def inner():
        nonlocal x
        x = "inner x"
        print(f"Inner: {x}")
    
    inner()
    print(f"Outer: {x}")

outer()

### Variable Scope and Namespaces

Understanding variable scope is crucial for writing maintainable and bug-free Python code. Python uses the LEGB rule to determine the order in which it looks up variable names.

### The LEGB Rule

Python looks for variables in the following order:
1. **L**ocal - Names assigned within a function
2. **E**nclosing - Names in the local scope of any enclosing functions
3. **G**lobal - Names assigned at the top-level of a module
4. **B**uilt-in - Names preassigned in the built-in names module

```python
# Built-in: print, len, etc. are in the built-in scope

global_var = "I'm global"

def outer():
    enclosing_var = "I'm in the enclosing scope"
    
    def inner():
        local_var = "I'm local"
        print(local_var)        # Local
        print(enclosing_var)    # Enclosing
        print(global_var)       # Global
        print(len)              # Built-in (len function)
    
    inner()

outer()

### Global and Nonlocal Keywords

#### The `global` Keyword
Used to modify a global variable from within a function:

```python
count = 0  # Global variable

def increment():
    global count  # Declare we want to use the global variable
    count += 1
    print(f"Count is now {count}")

increment()  # Count is now 1
increment()  # Count is now 2
print(f"Final count: {count}")  # Final count: 2

#### The `nonlocal` Keyword
Used to modify a variable in the nearest enclosing scope (but not global):

```python
def counter():
    count = 0  # Enclosing scope
    
    def increment():
        nonlocal count  # Refers to the count in the enclosing scope
        count += 1
        return count
    
    return increment

# Create a counter
c = counter()
print(c())  # 1
print(c())  # 2
print(c())  # 3

# Create another independent counter
c2 = counter()
print(c2())  # 1 (new instance, new count)
print(c())   # 4 (continues from previous counter)

### Closures and Factory Functions

A closure is a function object that remembers values in the enclosing scope even if they are not present in memory.

```python
def make_multiplier(factor):
    """Factory function that creates multiplier functions."""
    def multiplier(x):
        return x * factor
    return multiplier

# Create specialized functions
double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# The factor value is remembered by the inner function
print(double.__closure__)  # Contains the factor value
print(double.__closure__[0].cell_contents)  # 2

### Practical Example: Function Caching with Closures

```python
def cache(func):
    """Decorator that caches function results."""
    cached_results = {}
    
    def wrapper(*args):
        if args in cached_results:
            print(f"Cache hit for {args}")
            return cached_results[args]
        print(f"Calculating result for {args}")
        result = func(*args)
        cached_results[args] = result
        return result
    
    return wrapper

@cache
def expensive_operation(x, y):
    """Simulate an expensive computation."""
    return x ** y

# First call - not in cache
print(expensive_operation(2, 10))  # Calculates and caches
# Second call with same args - uses cache
print(expensive_operation(2, 10))  # Uses cached result

### Common Pitfalls and Best Practices

1. **Avoid modifying global variables** - Prefer passing values as parameters and returning results
2. **Use `nonlocal` carefully** - It can make code harder to understand if overused
3. **Closures maintain state** - Be aware that closures keep references to variables in the enclosing scope
4. **Default arguments are evaluated once** - Be careful with mutable default arguments
5. **Use `global` and `nonlocal` sparingly** - They can make code harder to reason about

```python
# Bad practice: Modifying globals without declaring them
total = 0

def add_to_total(amount):
    # This would raise an UnboundLocalError
    # total += amount  # Uncomment to see the error
    pass

# Good practice: Be explicit about using globals
def add_to_total_good(amount):
    global total
    total += amount

# Better practice: Avoid globals when possible
def add_to_total_better(current_total, amount):
    return current_total + amount

total = add_to_total_better(total, 10)

### Scoping in List Comprehensions and Generator Expressions

In Python 3, list comprehensions, set comprehensions, and generator expressions have their own scope:

```python
x = 10
# In Python 3, list comp has its own scope
squares = [x**2 for x in range(5)]
print(x)  # Still 10, not 4

# But in Python 2, x would be 4 here

# This is equivalent to:
def make_squares():
    squares = []
    for x in range(5):
        squares.append(x**2)
    return squares

squares = make_squares()
print(x)  # Still 10

### The `__closure__` Attribute

You can inspect a function's closure to see what variables it's capturing:

```python
def outer(x):
    def inner(y):
        return x + y
    return inner

closure_func = outer(10)
print(closure_func(5))  # 15

# Inspect the closure
print(closure_func.__closure__)  # Contains the cell for x
print(closure_func.__closure__[0].cell_contents)  # 10

## Modules and Packages

### Creating and Using Modules

A Python module is a file containing Python definitions and statements. The file name is the module name with the suffix `.py`.

#### Basic Module Example

```python
# File: calculator.py
"""A simple calculator module with basic arithmetic operations."""

def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Return the difference between two numbers."""
    return a - b

# Module-level constant
PI = 3.14159265359

# This code runs when the module is executed directly
if __name__ == "__main__":
    print("Running calculator module directly")
    print(f"2 + 2 = {add(2, 2)}")
```

#### Importing Modules

```python
# Import the entire module
import calculator
result = calculator.add(5, 3)  # 8.0

# Import specific functions/classes
from calculator import add, subtract
result = add(10, 20)  # 30.0

# Import with alias
import calculator as calc
from calculator import add as addition

# Import all names (not recommended in production code)
from calculator import *
result = subtract(10, 5)  # 5.0
```

### Creating and Using Packages

A package is a way to organize related modules into a directory hierarchy. A package is a directory that contains a special `__init__.py` file.

#### Package Structure

```
my_package/
├── __init__.py           # Package initialization
├── module1.py            # Module 1
├── module2.py            # Module 2
├── subpackage1/          # Subpackage 1
│   ├── __init__.py
│   ├── submodule1.py
│   └── submodule2.py
└── tests/                # Tests directory
    ├── __init__.py
    └── test_module1.py
```

#### `__init__.py` File

The `__init__.py` file can be empty or can contain initialization code for the package. It's executed when the package is imported.

```python
# my_package/__init__.py
"""
My Package - A collection of useful modules.

This package provides various utilities for common programming tasks.
"""

__version__ = '1.0.0'
__author__ = 'Your Name <your.email@example.com>'

# Import key functions/classes to make them available at package level
from .module1 import important_function
from .module2 import ImportantClass

# Define what gets imported with 'from my_package import *'
__all__ = ['important_function', 'ImportantClass']

# Package initialization code
print(f"Initializing {__name__} package")
```

#### Relative vs Absolute Imports

```python
# Absolute import (recommended)
from my_package.subpackage1.submodule1 import some_function

# Relative import (within the same package)
from .module1 import helper_function  # Same directory
from ..module2 import another_function  # Parent directory
```

### The Module Search Path

When you import a module, Python searches for it in the following order:
1. The current directory
2. The list of directories in the `PYTHONPATH` environment variable
3. The installation-dependent default directory

```python
import sys
print(sys.path)  # List of directories Python searches for modules
```

### Installing Third-Party Packages

```bash
# Install a package
pip install requests

# Install a specific version
pip install django==4.0.0

# Install from requirements.txt
pip install -r requirements.txt

# Install in development mode (editable)
pip install -e .
```

### Best Practices for Modules and Packages

1. **Keep modules focused**: Each module should have a single, well-defined purpose
2. **Use meaningful names**: Module names should be short, lowercase, and descriptive
3. **Document your modules**: Include docstrings at the top of each module
4. **Organize imports properly**: Group imports in the following order:
   - Standard library imports
   - Third-party imports
   - Local application/library specific imports
5. **Use `if __name__ == '__main__'`**: For code that should only run when the module is executed directly
6. **Be careful with `from module import *`**: It can make code harder to understand and maintain
7. **Handle package dependencies**: Clearly specify dependencies in `setup.py` or `requirements.txt`
8. **Use virtual environments**: To manage package dependencies for different projects

## The `__name__` Variable

In Python, `__name__` is a special built-in variable that evaluates to the name of the current module. It's particularly useful for determining whether a Python file is being run directly or being imported as a module.

### Basic Usage

```python
# File: mymodule.py
def main():
    print("This is the main function")

# This code runs only when the module is executed directly
if __name__ == "__main__":
    print("Running as main program")
    main()
else:
    print(f"Imported as a module: {__name__}")
```

### How It Works

- When a Python file is run directly, `__name__` is set to `"__main__"`
- When a module is imported, `__name__` is set to the module's name

### Common Use Cases

#### 1. Making a Python File Both Importable and Executable

```python
# File: calculator.py

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def main():
    """Run when the script is executed directly."""
    print("Running calculator in script mode")
    result = add(5, 3)
    print(f"5 + 3 = {result}")

if __name__ == "__main__":
    main()
```

#### 2. Module Testing

```python
# File: mymodule.py

def complex_calculation(x: int) -> int:
    """A complex calculation that needs testing."""
    return x * x + 2 * x + 1

def test():
    """Run tests for this module."""
    assert complex_calculation(0) == 1
    assert complex_calculation(1) == 4
    assert complex_calculation(2) == 9
    print("All tests passed!")

if __name__ == "__main__":
    test()
```

#### 3. Command-Line Interface (CLI)

```python
# File: cli_app.py
import argparse

def process_file(filename: str) -> None:
    """Process the input file."""
    print(f"Processing {filename}...")
    # Processing logic here

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('files', nargs='+', help='Files to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    if args.verbose:
        print("Verbose mode enabled")
    
    for filename in args.files:
        process_file(filename)

if __name__ == "__main__":
    main()
```

### Advanced Patterns

#### 1. Package Initialization

```python
# File: my_package/__main__.py
"""
This module is executed when the package is run with -m:
    python -m my_package
"""

def main():
    print("Running package as main script")
    # Package execution code here

if __name__ == "__main__":
    main()
```

#### 2. Conditional Imports

```python
# File: mymodule.py

def get_platform():
    """Return the current platform."""
    import sys
    return sys.platform

# Only import platform-specific modules when needed
if get_platform() == "win32":
    import msvcrt
    def getch():
        """Get a single character on Windows."""
        return msvcrt.getch().decode('utf-8')
else:
    import sys, tty, termios
    def getch():
        """Get a single character on Unix-like systems."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def main():
    print("Press any key...")
    char = getch()
    print(f"You pressed: {char}")

if __name__ == "__main__":
    main()
```

### Best Practices

1. **Always use `if __name__ == "__main__":`** for code that should only run when the script is executed directly
2. **Keep the main block minimal** - Move most code into functions or classes
3. **Use argument parsing** for command-line tools (e.g., `argparse`)
4. **Document your module** with docstrings at the top of the file
5. **Include example usage** in the docstring or as a `main()` function
6. **Consider using `__main__.py`** for packages that can be run with `python -m package`
7. **Test your module** - Include test functions that run when the module is executed directly

### Common Pitfalls

1. **Forgetting the quotes** in `"__main__"`
2. **Putting too much code** at the module level
3. **Not handling imports properly** when the module is imported
4. **Assuming the current working directory** will be the same as the script location
5. **Not using proper error handling** in the main block

## Virtual Environments and Dependency Management

Virtual environments are essential tools for Python development that allow you to create isolated Python environments for different projects. This prevents package conflicts and ensures reproducibility across different systems.

### Creating and Managing Virtual Environments

#### Using `venv` (Built-in to Python 3.3+)

```bash
# Create a virtual environment
python -m venv myenv

# Activate (Unix/macOS)
source myenv/bin/activate

# Activate (Windows)
# Command Prompt:
myenv\Scripts\activate.bat
# PowerShell:
myenv\Scripts\Activate.ps1  # Might need to set execution policy first

# Deactivate (any platform)
deactivate

# Check which Python interpreter is being used
which python  # Unix/macOS
where python  # Windows
```

#### Using `virtualenv` (Third-party, works with Python 2 and 3)

```bash
# Install virtualenv
pip install --user virtualenv

# Create a virtual environment
virtualenv myenv

# Create with a specific Python version
virtualenv -p python3.9 myenv
```

### Managing Dependencies

#### requirements.txt

```bash
# Generate requirements.txt from installed packages
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Install in development mode (editable)
pip install -e .
```

#### pip-tools for Better Dependency Management

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in with your direct dependencies
# requirements.in
django>=4.0.0
requests>=2.25.0

# Compile requirements.txt
pip-compile requirements.in

# Sync virtual environment with requirements.txt
pip-sync
```

### Advanced Virtual Environment Management

#### Using `.env` Files with python-dotenv

```bash
# Install python-dotenv
pip install python-dotenv
```

```python
# .env
DATABASE_URL=postgres://user:password@localhost:5432/dbname
DEBUG=True
SECRET_KEY=your-secret-key-here

# settings.py
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
```

#### Virtual Environment in Jupyter Notebooks

```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"

# List available kernels
jupyter kernelspec list

# Remove a kernel
jupyter kernelspec uninstall myenv
```

### Best Practices for Virtual Environments

1. **Never commit your virtual environment** to version control
   - Add the virtual environment directory to `.gitignore`:
     ```
     # .gitignore
     venv/
     env/
     .env
     .venv/
     *.pyc
     __pycache__/
     ```

2. **Use different environments** for different projects to avoid conflicts

3. **Document dependencies** clearly in `requirements.txt` or `setup.py`

4. **Pin your dependencies** for production:
   ```
   # requirements.txt
   Django==4.0.3
   requests==2.26.0
   ```

5. **Use `python -m pip`** instead of just `pip` to avoid ambiguity

6. **Consider using higher-level tools** for complex projects:
   - `poetry` for dependency management and packaging
   - `pipenv` for combining pip and virtualenv
   - `conda` for data science projects with non-Python dependencies

### Example: Setting Up a New Project

```bash
# Create project directory
mkdir my_project
cd my_project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install django requests python-dotenv

# Create requirements.txt
pip freeze > requirements.txt

# Create .gitignore
echo "venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore

# Initialize git repository
git init
git add .
git commit -m "Initial commit"
```

### Troubleshooting Virtual Environments

1. **Command not found: activate**
   - On Windows, use backslashes: `venv\Scripts\activate`
   - On PowerShell, you might need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

2. **Python version mismatch**
   - Specify the Python version when creating the environment: `python3.9 -m venv myenv`

3. **Broken environment**
   - Sometimes it's easier to delete and recreate the environment
   ```bash
   deactivate
   rm -rf venv/
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Path too long on Windows**
   - Enable long paths in Windows or create the environment in a shorter path
   - Or use: `python -m venv C:\short\path\venv`

## Practice Exercises

1. **Basic Function**
   Write a function that takes a temperature in Celsius and converts it to Fahrenheit.

2. **Default Arguments**
   Create a function that builds a dictionary with user information. Make some parameters required and others optional.

3. **Variable Arguments**
   Write a function that can take any number of numbers and returns their sum.

4. **Modules**
   Create a module called `calculator.py` with functions for basic arithmetic operations. Import and use it in another file.

5. **Scope**
   Create a function that has a local variable with the same name as a global variable. Show how to access both inside the function.

6. **Package**
   Create a Python package with at least two modules. Import and use functions from both modules.

7. **Main Function**
   Write a script that has a main function that's only executed when the script is run directly.

8. **Docstrings**
   Write a well-documented function that calculates the factorial of a number.

9. **Lambda Functions**
   Create a function that takes a function and a list and applies the function to each element of the list.

10. **Advanced**
    Create a decorator that logs the execution time of a function.

---
Next: [Object-Oriented Programming in Python](./04_oop_python.md)
