# Error Handling in Python

## Table of Contents
1. [Introduction to Exceptions](#introduction-to-exceptions)
2. [Handling Exceptions](#handling-exceptions)
3. [Built-in Exceptions](#built-in-exceptions)
4. [Raising Exceptions](#raising-exceptions)
5. [Creating Custom Exceptions](#creating-custom-exceptions)
6. [The `else` and `finally` Clauses](#the-else-and-finally-clauses)
7. [Context Managers Revisited](#context-managers-revisited)
8. [Debugging Techniques](#debugging-techniques)
9. [Practice Exercises](#practice-exercises)

## Introduction to Exceptions

Exceptions are events that occur during program execution that disrupt the normal flow of the program.

### Common Exceptions
- `ZeroDivisionError`: Division by zero
- `TypeError`: Operation on inappropriate type
- `ValueError`: Function receives correct type but inappropriate value
- `FileNotFoundError`: File doesn't exist
- `IndexError`: Sequence index out of range
- `KeyError`: Dictionary key not found
- `ImportError`: Module import fails

## Handling Exceptions

### Basic `try`-`except` Block
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

### Handling Multiple Exceptions
```python
try:
    # Code that might raise an exception
    value = int(input("Enter a number: "))
    result = 10 / value
except (ValueError, ZeroDivisionError) as e:
    print(f"Error: {e}")
```

### Catching All Exceptions (Not Recommended)
```python
try:
    # Risky code
    pass
except Exception as e:
    print(f"An error occurred: {e}")
```

## Built-in Exceptions

### Common Exception Classes
- `BaseException`: Base class for all built-in exceptions
  - `Exception`: Base class for all non-system-exiting exceptions
    - `ArithmeticError`: Base class for arithmetic errors
      - `ZeroDivisionError`
      - `OverflowError`
    - `LookupError`: Base class for lookup errors
      - `IndexError`
      - `KeyError`
    - `OSError`: Operating system errors
      - `FileNotFoundError`
      - `PermissionError`
    - `TypeError`
    - `ValueError`

### The Exception Hierarchy
```python
import builtins

# Print the exception hierarchy
def print_exception_hierarchy(exception_class, indent=0):
    print('  ' * indent + exception_class.__name__)
    for subclass in exception_class.__subclasses__():
        print_exception_hierarchy(subclass, indent + 1)

print_exception_hierarchy(builtins.BaseException)
```

## Raising Exceptions

### The `raise` Statement
```python
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

try:
    result = divide(10, 0)
except ZeroDivisionError as e:
    print(f"Error: {e}")
```

### Re-raising Exceptions
```python
try:
    # Some code that might fail
    x = 1 / 0
except ZeroDivisionError as e:
    print("Logging the error")
    raise  # Re-raise the current exception
```

### Exception Chaining
```python
try:
    # Some code that might fail
    x = 1 / 0
except Exception as e:
    raise RuntimeError("An error occurred") from e
```

## Creating Custom Exceptions

### Basic Custom Exception
```python
class InvalidEmailError(Exception):
    """Raised when the email format is invalid"""
    pass

def validate_email(email):
    if '@' not in email:
        raise InvalidEmailError("Invalid email format")
    return True

try:
    validate_email("invalid-email")
except InvalidEmailError as e:
    print(f"Validation error: {e}")
```

### Advanced Custom Exception
```python
class APIError(Exception):
    """Base class for API errors"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
    
    def to_dict(self):
        rv = dict(self.payload or {})
        rv['message'] = self.message
        rv['status_code'] = self.status_code
        return rn

class NotFoundError(APIError):
    """Raised when a resource is not found"""
    def __init__(self, message="Resource not found", payload=None):
        super().__init__(message, 404, payload)

# Usage
try:
    # Code that might fail
    raise NotFoundError("User not found")
except APIError as e:
    print(e.to_dict())
```

## The `else` and `finally` Clauses

### The `else` Clause
```python
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero")
else:
    print(f"Result: {result}")  # Executes only if no exception occurred
```

### The `finally` Clause
```python
file = None
try:
    file = open('example.txt', 'r')
    content = file.read()
    # Some processing
    result = 10 / 0
except (FileNotFoundError, ZeroDivisionError) as e:
    print(f"Error: {e}")
finally:
    if file is not None:
        file.close()  # Always executed
    print("Cleanup complete")
```

## Context Managers Revisited

### Using `contextlib` for Error Handling
```python
from contextlib import contextmanager

@contextmanager
def open_file(filename, mode):
    try:
        file = open(filename, mode)
        yield file
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        file.close()

# Usage
try:
    with open_file('nonexistent.txt', 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("File not found")
```

## Debugging Techniques

### Using `assert`
```python
def calculate_average(numbers):
    assert len(numbers) > 0, "List cannot be empty"
    return sum(numbers) / len(numbers)

# Test with -O flag to disable assertions
# python -O script.py
```

### Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='w'
)

def divide(a, b):
    try:
        result = a / b
        logging.info(f"Division successful: {a} / {b} = {result}")
        return result
    except ZeroDivisionError:
        logging.error("Attempted to divide by zero")
        raise
    except Exception as e:
        logging.exception("An unexpected error occurred")
        raise
```

### Using `pdb` (Python Debugger)
```python
import pdb

def buggy_function():
    x = 1
    y = 0
    pdb.set_trace()  # Execution will pause here
    return x / y

# Commands in pdb:
# n: next line
# c: continue
# s: step into function
# q: quit
# p <var>: print variable
# l: list source code
# h: help
```

## Practice Exercises

1. **Basic Exception Handling**
   Write a function that takes two numbers and returns their division. Handle division by zero.

2. **File Operations**
   Write a function that reads a file and handles `FileNotFoundError`.

3. **Custom Exception**
   Create a custom exception for invalid age (e.g., negative age).

4. **Input Validation**
   Write a function that asks for a number between 1-10 and handles invalid input.

5. **Context Manager**
   Create a context manager that measures and prints the execution time of a code block.

6. **Logging**
   Modify a function to log errors to a file instead of printing them.

7. **Recovery**
   Write a function that tries to open a file, and if it doesn't exist, creates it.

8. **Exception Chaining**
   Write code that catches an exception and raises a custom exception with the original as the cause.

9. **Multiple Exception Types**
   Write a function that handles different types of exceptions differently.

10. **Debugging**
    Use `pdb` to debug a function that's not working as expected.

---
Next: [List and Dictionary Comprehensions](./07_comprehensions.md)
