# Python Coding Best Practices

## Table of Contents
1. [Code Style and Readability](#code-style-and-readability)
2. [Naming Conventions](#naming-conventions)
3. [Documentation and Comments](#documentation-and-comments)
4. [Error Handling](#error-handling)
5. [Performance Considerations](#performance-considerations)
6. [Code Organization](#code-organization)
7. [Testing and Quality Assurance](#testing-and-quality-assurance)
8. [Security Best Practices](#security-best-practices)
9. [Version Control Best Practices](#version-control-best-practices)
10. [Pythonic Idioms](#pythonic-idioms)

## Code Style and Readability

### Follow PEP 8
- Use 4 spaces per indentation level
- Limit lines to 79 characters (72 for docstrings)
- Use blank lines to separate functions and classes
- Use spaces around operators and after commas

```python
# Good
result = calculate(a, b, c=1)

# Bad
result=calculate(a,b,c=1)
```

### Use Consistent Quotes
```python
# Good
text = 'single quotes'
text = "double quotes when needed for the string's content"

# Be consistent within a project
```

### Line Continuation
```python
# Use parentheses for implicit line continuation
total = (first_variable + second_variable
         - third_variable)

# For long strings, use parentheses or triple quotes
message = ("This is a very long string that "
           "spans multiple lines.")
```

### Imports
```python
# Standard library imports
import os
import sys
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
from . import my_module
from .my_package import my_function

# Group imports in this order with blank lines between groups
```

## Naming Conventions

### General Naming
- `snake_case` for variables, functions, and methods
- `PascalCase` for class names
- `UPPER_CASE` for constants
- `_single_leading_underscore` for "private" implementation details
- `__double_leading_underscore` for name mangling
- `single_trailing_underscore_` to avoid conflicts with Python keywords

### Descriptive Names
```python
# Good
def calculate_total_price(items):
    return sum(item.price for item in items)

# Bad
def calc(x):
    return sum(i.p for i in x)
```

### Avoid Using Built-in Names
```python
# Bad
list = [1, 2, 3]  # Overrides built-in list()
dict = {}         # Overrides built-in dict()
```

## Documentation and Comments

### Docstrings
```python
"""Module-level docstring describing the module's purpose."""

def calculate_total(items, discount=0.0):
    """Calculate the total price of items with an optional discount.
    
    Args:
        items (list): List of items with 'price' attribute
        discount (float, optional): Discount percentage (0-1). Defaults to 0.0.
        
    Returns:
        float: Total price after applying discount
        
    Raises:
        ValueError: If discount is not between 0 and 1
    """
    if not 0 <= discount <= 1:
        raise ValueError("Discount must be between 0 and 1")
    subtotal = sum(item.price for item in items)
    return subtotal * (1 - discount)
```

### Comments
```python
# Use comments to explain why, not what
# Bad: Increment i by 1
i += 1

# Good: Adjust index to account for header row
i += 1
```

### Type Hints (Python 3.5+)
```python
from typing import List, Dict, Optional, Union, Tuple

def process_data(
    data: List[Dict[str, Union[str, int]]],
    threshold: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """Process input data and return success status and optional message."""
    # Function implementation
    return True, None
```

## Error Handling

### Use Specific Exceptions
```python
# Good
try:
    value = my_dict[key]
except KeyError as e:
    logger.error(f"Key {key} not found: {e}")
    raise

# Bad
try:
    value = my_dict[key]
except:  # Too broad
    pass
```

### Create Custom Exceptions
```python
class ValidationError(Exception):
    """Raised when validation fails."""
    pass

def validate_user(user):
    if not user.name:
        raise ValidationError("User must have a name")
```

### Use Context Managers for Resources
```python
# Good
with open('file.txt') as f:
    content = f.read()

# Bad
f = open('file.txt')
try:
    content = f.read()
finally:
    f.close()
```

## Performance Considerations

### Use Built-in Functions
```python
# Good
names = [item.name for item in items]

# Bad
names = []
for item in items:
    names.append(item.name)
```

### Avoid Global Variables
```python
# Bad
global_var = 42

def process():
    global global_var
    global_var += 1

# Good
def process(counter):
    return counter + 1
```

### Use Generator Expressions for Large Datasets
```python
# Good - Lazy evaluation
sum_squares = sum(x**2 for x in range(1000000))

# Bad - Creates entire list in memory
sum_squares = sum([x**2 for x in range(1000000)])
```

## Code Organization

### Functions and Methods
- Keep functions small and focused (Single Responsibility Principle)
- Limit the number of parameters (preferably 3 or fewer)
- Use keyword arguments for clarity
- Return early to reduce nesting

### Classes
```python
class User:
    """A class representing a user."""
    
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self._private_var = None  # Internal use only
    
    @property
    def email_domain(self) -> str:
        """Return the domain part of the email address."""
        return self.email.split('@')[-1]
    
    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"
```

### Modules and Packages
- Keep related functionality together
- Use `__init__.py` to define package interfaces
- Follow the principle of least surprise

## Testing and Quality Assurance

### Write Unit Tests
```python
# test_calculator.py
import unittest
from calculator import add

class TestCalculator(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)
    
    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
```

### Use Linters and Formatters
- `black`: Uncompromising code formatter
- `flake8`: Linter that checks for style and programming errors
- `mypy`: Static type checker
- `pylint`: Source code analyzer

### Continuous Integration
Set up CI/CD pipelines to run tests, linters, and type checkers on every commit.

## Security Best Practices

### Input Validation
```python
import re

def is_valid_email(email: str) -> bool:
    """Check if the email address is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

### Avoid `eval()` and `exec()`
```python
# Dangerous
user_input = "os.system('rm -rf /')"
eval(user_input)  # Never do this!
```

### Use `secrets` for Cryptography
```python
import secrets

# Generate a secure token
token = secrets.token_hex(32)

# Generate a secure random password
import string
alphabet = string.ascii_letters + string.digits
password = ''.join(secrets.choice(alphabet) for _ in range(16))
```

## Version Control Best Practices

### Meaningful Commit Messages
```
feat: add user authentication

- Add login/logout functionality
- Implement JWT token generation
- Add password hashing with bcrypt

Resolves: #123
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
```

## Pythonic Idioms

### List Comprehensions
```python
# Good
squares = [x**2 for x in range(10)]

# Better with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Dictionary Comprehensions
```python
# Good
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Unpacking
```python
# Multiple assignment
a, b = 1, 2

# Swap variables
a, b = b, a

# Extended unpacking
first, *rest = [1, 2, 3, 4]  # first=1, rest=[2, 3, 4]
```

### Context Managers
```python
# Custom context manager
from contextlib import contextmanager

@contextmanager
temporary_change(obj, attr, value):
    """Temporarily change an object's attribute."""
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old_value)
```

### The `with` Statement
```python
# Good
with open('file.txt') as f:
    content = f.read()

# Even better with multiple files
with open('file1.txt') as f1, open('file2.txt') as f2:
    content1 = f1.read()
    content2 = f2.read()
```

### Use `collections` Module
```python
from collections import defaultdict, Counter, namedtuple

# Count occurrences
counts = Counter('abracadabra')  # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# Default dictionary
d = defaultdict(list)
d['key'].append('value')

# Named tuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
```

### Use `enumerate` for Index-Value Pairs
```python
# Good
for index, value in enumerate(['a', 'b', 'c']):
    print(f"{index}: {value}")
```

### Use `zip` to Iterate Over Multiple Sequences
```python
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

### Use `any()` and `all()`
```python
# Check if any value is True
if any(x > 0 for x in values):
    print("At least one positive value")

# Check if all values are True
if all(x > 0 for x in values):
    print("All values are positive")
```

### Use `f-strings` (Python 3.6+)
```python
name = "Alice"
age = 30
print(f"{name} is {age} years old")  # Alice is 30 years old

# With expressions
print(f"Next year, {name} will be {age + 1}")

# With formatting
pi = 3.14159
print(f"π is approximately {pi:.2f}")  # π is approximately 3.14
```

### Use `pathlib` for File Paths
```python
from pathlib import Path

# Create a path object
path = Path('directory') / 'file.txt'

# Check if file exists
if path.exists():
    content = path.read_text()

# Create directory if it doesn't exist
path.parent.mkdir(parents=True, exist_ok=True)

# Write to file
path.write_text('Hello, World!')
```

### Use `dataclasses` (Python 3.7+)
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0  # Default value

@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self) -> float:
        return abs(self.bottom_right.x - self.top_left.x)
    
    @property
    def height(self) -> float:
        return abs(self.bottom_right.y - self.top_left.y)
    
    def area(self) -> float:
        return self.width * self.height
```

### Use Type Hints (Python 3.5+)
```python
from typing import List, Dict, Optional, Union, Tuple, Callable

def process_data(
    data: List[Dict[str, Union[str, int]]],
    callback: Optional[Callable[[int], None]] = None
) -> Tuple[bool, str]:
    """Process data and return success status and message."""
    # Function implementation
    if callback:
        callback(len(data))
    return True, "Success"
```

By following these best practices, you'll write Python code that is more maintainable, efficient, and less prone to errors. Remember that while these are guidelines, always consider the specific needs of your project and team.
