# Python Exam Preparation Guide

## Table of Contents
1. [Exam Overview](#exam-overview)
2. [Functions in Depth](#functions-in-depth)
3. [Text Manipulation](#text-manipulation)
4. [Data Structures](#data-structures)
5. [Study Strategies](#study-strategies)
6. [Practice Problems](#practice-problems)
7. [Common Pitfalls](#common-pitfalls)

## Exam Overview

The exam will test your understanding of Python programming with a focus on:
- Writing and using functions
- Text manipulation techniques
- Working with different data structures (lists, tuples, sets, dictionaries)
- Problem-solving skills

The key to success is understanding concepts rather than memorizing code. Expect questions that test your ability to apply Python concepts to solve problems.

## Functions in Depth

### Function Basics
```python
def function_name(parameter1, parameter2=default_value):
    """
    Docstring: A multi-line string that documents the function's purpose,
    parameters, and return value.
    
    Args:
        parameter1: Description of the first parameter
        parameter2: Description of the second parameter (optional)
        
    Returns:
        Description of the return value
    """
    # Function body
    result = parameter1 + parameter2
    return result
```

### Function Components Explained

1. **Parameters vs Arguments**
   - Parameters are variables in the function definition
   - Arguments are the actual values passed to the function
   ```python
   def greet(name):  # name is a parameter
       return f"Hello, {name}!"
   
   greet("Alice")  # "Alice" is an argument
   ```

2. **Return Values**
   - `return` exits the function and sends back a value
   - Functions return `None` if no return statement is reached
   - Multiple values can be returned as a tuple
   ```python
   def min_max(numbers):
       return min(numbers), max(numbers)  # Returns a tuple
   ```

3. **Variable Scope**
   - Local variables: Defined inside a function
   - Global variables: Defined outside any function
   - `nonlocal` keyword for nested functions
   ```python
   x = 10  # Global
   
   def outer():
       y = 20  # Enclosing
       
       def inner():
           nonlocal y  # Refers to y in the enclosing scope
           global x    # Refers to x in the global scope
           z = 30     # Local
   ```

4. **Default Parameters**
   - Parameters with default values must come after parameters without defaults
   - Mutable default arguments are evaluated only once
   ```python
   def power(base, exponent=2):
       return base ** exponent
   ```

5. **Keyword Arguments**
   - Pass arguments by parameter name
   - Order doesn't matter when using keyword arguments
   ```python
   def describe_pet(pet_name, animal_type='dog'):
       print(f"I have a {animal_type} named {pet_name}.")
   
   describe_pet(animal_type='hamster', pet_name='Harry')
   ```

6. **Variable-length Arguments**
   - `*args`: Variable number of positional arguments (stored as tuple)
   - `**kwargs`: Variable number of keyword arguments (stored as dictionary)
   ```python
   def print_info(*args, **kwargs):
       print("Positional arguments:", args)
       print("Keyword arguments:", kwargs)
   
   print_info(1, 2, 3, name="Alice", age=25)
   ```

7. **Lambda Functions**
   - Anonymous functions defined with `lambda`
   - Can only contain a single expression
   ```python
   square = lambda x: x ** 2
   sorted_by_second = lambda items: sorted(items, key=lambda x: x[1])
   ```

8. **Recursion**
   - Functions that call themselves
   - Must have a base case to prevent infinite recursion
   ```python
   def factorial(n):
       return 1 if n <= 1 else n * factorial(n-1)
   ```

### Advanced Function Concepts

1. **Function Annotations**
   ```python
   def greet(name: str, age: int) -> str:
       return f"{name} is {age} years old"
   ```

2. **Closures**
   ```python
   def make_multiplier(factor):
       def multiply(x):
           return x * factor
       return multiply
   
   double = make_multiplier(2)
   print(double(5))  # 10
   ```

3. **Decorators**
   ```python
   def my_decorator(func):
       def wrapper(*args, **kwargs):
           print("Something before the function is called.")
           result = func(*args, **kwargs)
           print("Something after the function is called.")
           return result
       return wrapper
   
   @my_decorator
   def say_hello(name):
       print(f"Hello {name}!")
   ```

4. **Generators**
   ```python
   def count_up_to(max):
       count = 1
       while count <= max:
           yield count
           count += 1
   
   counter = count_up_to(5)
   print(next(counter))  # 1
   print(next(counter))  # 2
   ```

### Example Problems

1. **Basic Function**
   ```python
   def square_even_numbers(numbers):
       """Return a new list with even numbers squared."""
       return [x**2 for x in numbers if x % 2 == 0]
   ```

2. **Advanced Function**
   ```python
   def process_data(data, *, reverse=False, key=None):
       """
       Process a list of items with optional sorting.
       
       Args:
           data: List of items to process
           reverse: If True, sort in descending order
           key: Function of one argument to extract a comparison key
       """
       if key:
           return sorted(data, key=key, reverse=reverse)
       return sorted(data, reverse=reverse)
   ```

## Text Manipulation

### String Fundamentals

**Real-world Applications:**
- Data cleaning and preprocessing
- Log file analysis
- Web scraping and parsing
- Natural Language Processing (NLP) tasks
- Form validation
- Report generation

### Common String Operations with Real-world Examples

1. **Basic String Operations**
```python
# User input sanitization
def sanitize_username(username):
    """Remove special characters and convert to lowercase."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', username).lower()

# File path manipulation
file_path = "/user/documents/reports/2023/summary.pdf"
filename = file_path.split('/')[-1]  # 'summary.pdf'
extension = filename.split('.')[-1]  # 'pdf'
```

2. **String Formatting**
```python
# Dynamic email templates
def send_welcome_email(user):
    subject = f"Welcome to Our Service, {user['name']}!"
    body = """
    Dear {name},
    
    Thank you for joining on {join_date:%B %d, %Y}.
    Your username is: {username}
    
    Best regards,
    The Team
    """.format(
        name=user['name'],
        join_date=user['join_date'],
        username=user['username']
    )
    return subject, body
```

3. **Advanced Text Processing**
```python
# Log file analysis
def parse_log_line(line):
    """Parse a log line into its components."""
    # Example: "2023-04-01 12:34:56 [ERROR] Failed to connect to database"
    import datetime
    from collections import namedtuple
    
    LogEntry = namedtuple('LogEntry', ['timestamp', 'level', 'message'])
    
    try:
        timestamp_str = line[:19]
        timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        level_end = line.find(']', 20)
        level = line[21:level_end]
        message = line[level_end + 2:].strip()
        return LogEntry(timestamp, level, message)
    except Exception as e:
        return None
```

4. **Regular Expressions**
```python
# Data extraction from text
def extract_emails(text):
    """Extract all email addresses from a string."""
    import re
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)

# URL validation
def is_valid_url(url):
    """Check if a string is a valid URL."""
    import re
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(pattern.match(url))
```

5. **Text Cleaning and Normalization**
```python
def clean_text(text):
    """Clean and normalize text for NLP processing."""
    import re
    import string
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords (example with common English stopwords)
    stopwords = {'the', 'and', 'to', 'of', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'you', 'it', 'as', 'from'}
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    return text
```

6. **Working with CSV/TSV Data**
```python
def process_csv_file(file_path):
    """Process a CSV file and extract specific information."""
    import csv
    
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Process each row (example: filter and transform data)
            if float(row['price']) > 100:  # Example condition
                results.append({
                    'id': row['id'],
                    'name': row['name'].title(),  # Capitalize first letters
                    'price': f"${float(row['price']):.2f}",
                    'category': row['category'].lower()
                })
    return results
```

### String Performance Considerations
- Use `join()` instead of `+` for concatenating multiple strings in a loop
- Use f-strings (Python 3.6+) for better readability and performance
- For large text processing, consider using generators and streaming
- Be cautious with regex complexity for very large texts
- Use built-in string methods when possible as they're optimized

1. **String Creation**
   ```python
   single = 'single quotes'
   double = "double quotes"
   triple = """multi-line
   string"""
   raw = r"C:\\path\\to\\file"  # Raw string
   ```

2. **String Indexing and Slicing**
   ```python
   text = "Python"
   print(text[0])     # 'P' (first character)
   print(text[-1])    # 'n' (last character)
   print(text[1:4])   # 'yth' (slice from index 1 to 3)
   print(text[::2])   # 'Pto' (every second character)
   print(text[::-1])  # 'nohtyP' (reverse string)
   ```

### Essential String Methods

1. **Case Conversion**
   ```python
   text = "Hello, World!"
   print(text.lower())       # 'hello, world!'
   print(text.upper())       # 'HELLO, WORLD!'
   print(text.title())       # 'Hello, World!'
   print(text.capitalize())  # 'Hello, world!'
   print(text.swapcase())    # 'hELLO, wORLD!'
   ```

2. **Searching and Replacing**
   ```python
   text = "Hello, World!"
   
   # Finding substrings
   print(text.find('World'))     # 7 (returns -1 if not found)
   print(text.index('World'))    # 7 (raises ValueError if not found)
   print(text.count('l'))        # 3
   
   # Checking content
   print(text.startswith('Hello'))  # True
   print(text.endswith('!'))        # True
   print('123'.isdigit())          # True
   print('abc'.isalpha())          # True
   
   # Replacing
   print(text.replace('World', 'Python'))  # 'Hello, Python!'
   ```

3. **Splitting and Joining**
   ```python
   csv = "apple,banana,cherry"
   fruits = csv.split(',')
   print(fruits)  # ['apple', 'banana', 'cherry']
   
   # Joining with a separator
   print('-'.join(['2023', '12', '01']))  # '2023-12-01'
   
   # Splitting with maxsplit
   print('one two three'.split(' ', 1))  # ['one', 'two three']
   ```

4. **Whitespace Management**
   ```python
   text = "   Hello, World!   "
   print(text.strip())   # 'Hello, World!'
   print(text.lstrip())  # 'Hello, World!   '
   print(text.rstrip())  # '   Hello, World!'
   ```

### String Formatting

1. **f-strings (Python 3.6+)**
   ```python
   name = "Alice"
   age = 25
   print(f"{name} is {age} years old.")
   print(f"{name.upper()} is {age + 5} years from now.")
   print(f"{name:>10}")  # Right-aligned in 10 characters
   print(f"{3.14159:.2f}")  # '3.14'
   ```

2. **str.format()**
   ```python
   print("{} is {} years old.".format(name, age))
   print("{1} is {0} years old.".format(age, name))  # Positional
   print("{name} is {age} years old.".format(name="Alice", age=25))
   ```

### Advanced String Operations

1. **String Translation**
   ```python
   translation = str.maketrans('aeiou', '12345')
   print('hello'.translate(translation))  # 'h2ll4'
   ```

2. **Partitioning**
   ```python
   print('hello-world'.partition('-'))  # ('hello', '-', 'world')
   print('hello-world'.rpartition('-')) # ('hello', '-', 'world')
   ```

3. **String Validation**
   ```python
   print('abc123'.isalnum())  # True
   print('ABC'.isupper())     # True
   print('   '.isspace())     # True
   print('hello'.islower())   # True
   ```

### Regular Expressions (re module)

```python
import re

# Basic matching
match = re.search(r'\d+', 'abc123def')
if match:
    print(match.group())  # '123'

# Finding all matches
numbers = re.findall(r'\d+', 'a1b22c333d')
print(numbers)  # ['1', '22', '333']

# Substitution
result = re.sub(r'\s+', '-', 'hello   world')
print(result)  # 'hello-world'
```

### Practical Applications

1. **Email Validation**
   ```python
   import re
   
   def is_valid_email(email):
       pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
       return bool(re.match(pattern, email))
   ```

2. **Extracting Data**
   ```python
   log = """
   [2023-01-01 12:00:00] INFO: User logged in
   [2023-01-01 12:05:23] ERROR: Connection failed
   """
   
   for match in re.finditer(r'\[(.*?)\] (\w+): (.*)', log):
       timestamp, level, message = match.groups()
       print(f"{timestamp} - {level}: {message}")
   ```

## Data Structures

### Lists
**Real-world Applications:**
- Storing and processing sequences of data (e.g., sensor readings, time series data)
- Maintaining ordered collections that need frequent modifications
- Implementing stacks (LIFO) and queues (FIFO) using `append()`/`pop()`

**When to use:**
- When you need an ordered, mutable collection
- When you need to store duplicates
- When you need index-based access to elements
- When you need to modify the collection (add/remove items)

**Example Scenario:**
```python
# E-commerce shopping cart
shopping_cart = [
    {'product': 'Laptop', 'price': 999.99, 'quantity': 1},
    {'product': 'Mouse', 'price': 24.99, 'quantity': 2}
]

def add_to_cart(product, price, quantity=1):
    shopping_cart.append({'product': product, 'price': price, 'quantity': quantity})

def calculate_total():
    return sum(item['price'] * item['quantity'] for item in shopping_cart)
```

### Tuples
**Real-world Applications:**
- Storing related data that shouldn't change (e.g., coordinates, RGB colors)
- Dictionary keys (since they're hashable)
- Function return values for multiple values

**When to use:**
- When you need an immutable sequence
- When you need to ensure data integrity
- When working with dictionary keys
- For better performance with large collections that won't change

**Example Scenario:**
```python
# Geographic coordinates (latitude, longitude, elevation)
location = (40.7128, -74.0060, 10)  # New York City with elevation

def get_weather(coordinates):
    lat, lon, _ = coordinates  # Unpacking tuple
    # API call to get weather data
    return {'temperature': 22, 'conditions': 'Sunny'}
```

### Dictionaries
**Real-world Applications:**
- JSON data handling
- Caching/memoization
- Counting occurrences of items
- Fast lookups by key

**When to use:**
- When you need key-value pairs
- For fast lookups (O(1) average case)
- When you need to count or group items
- When working with JSON data

**Example Scenario:**
```python
# User authentication system
users = {
    'alice': {'password': 'hashed_pass123', 'role': 'admin'},
    'bob': {'password': 'hashed_pass456', 'role': 'user'}
}

def authenticate(username, password):
    user = users.get(username)
    if user and user['password'] == hash_password(password):
        return {'status': 'success', 'role': user['role']}
    return {'status': 'failure', 'message': 'Invalid credentials'}
```

### Sets
**Real-world Applications:**
- Removing duplicates from a list
- Membership testing
- Mathematical operations (union, intersection, difference)
- Finding unique elements in a dataset

**When to use:**
- When you need to store unique elements
- For fast membership testing
- When you need set operations
- When order doesn't matter

**Example Scenario:**
```python
# Social media followers analysis
followers_alice = {'bob', 'charlie', 'dave'}
followers_bob = {'charlie', 'dave', 'eve'}

# Find mutual followers
mutual_followers = followers_alice & followers_bob  # {'charlie', 'dave'}

# Find users who follow only one of them
exclusive_followers = followers_alice ^ followers_bob  # {'bob', 'eve'}
```

### Choosing the Right Data Structure
1. **Need key-value pairs?**
   - Yes → Dictionary
   - No → Continue

2. **Need to maintain order?**
   - Yes → List or Tuple (if immutable)
   - No → Continue

3. **Need to store unique elements?**
   - Yes → Set
   - No → Continue

4. **Need to modify the collection?**
   - Yes → List or Dictionary
   - No → Tuple (immutable)

### 1. Lists

**Characteristics:**
- Ordered, mutable, allows duplicates
- Zero-based indexing
- Can contain mixed data types

**Common Operations:**
```python
# Creation
numbers = [1, 2, 3, 4, 5]
mixed = [1, 'two', 3.0, [4, 5]]

# Adding elements
numbers.append(6)           # [1, 2, 3, 4, 5, 6]
numbers.insert(1, 1.5)      # [1, 1.5, 2, 3, 4, 5, 6]
numbers.extend([7, 8])      # [1, 1.5, 2, 3, 4, 5, 6, 7, 8]

# Removing elements
popped = numbers.pop()      # 8, numbers is [1, 1.5, 2, 3, 4, 5, 6, 7]
removed = numbers.pop(1)    # 1.5, numbers is [1, 2, 3, 4, 5, 6, 7]
numbers.remove(3)           # Removes first occurrence of 3

# Slicing
print(numbers[1:4])         # [2, 3, 4]
print(numbers[::2])         # [1, 3, 5, 7]
print(numbers[::-1])        # [7, 6, 5, 4, 3, 2, 1]

# List comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]
```

### 2. Tuples

**Characteristics:**
- Ordered, immutable, allows duplicates
- Faster than lists
- Can be used as dictionary keys

**Common Operations:**
```python
# Creation
point = (10, 20)
colors = 'red', 'green', 'blue'  # Parentheses are optional
single = (1,)                   # Note the comma for single-item tuples

# Unpacking
x, y = point                    # x=10, y=20
first, *rest = colors           # first='red', rest=['green', 'blue']

# Named tuples (from collections import namedtuple)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)                 # 10 20
```

### 3. Sets

**Characteristics:**
- Unordered, mutable, no duplicates
- Very fast membership testing
- Mathematical set operations

**Common Operations:**
```python
# Creation
primes = {2, 3, 5, 7}
even = set([2, 4, 6, 8])

# Set operations
print(primes | even)       # Union: {2, 3, 4, 5, 6, 7, 8}
print(primes & even)       # Intersection: {2}
print(primes - even)       # Difference: {3, 5, 7}
print(primes ^ even)       # Symmetric difference: {3, 4, 5, 6, 7, 8}

# Set comprehensions
squares = {x**2 for x in range(10)}
```

### 4. Dictionaries

**Characteristics:**
- Key-value pairs (keys must be hashable)
- Mutable, ordered (as of Python 3.7)
- Very fast lookups

**Common Operations:**
```python
# Creation
person = {'name': 'Alice', 'age': 25}
empty_dict = {}

# Accessing
print(person['name'])           # 'Alice'
print(person.get('age', 0))     # 25 (0 is default if key doesn't exist)

# Adding/Updating
person['city'] = 'New York'     # Add new key-value pair
person.update({'age': 26, 'job': 'Engineer'})  # Multiple updates

# Dictionary comprehensions
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Dictionary views
print(person.keys())     # dict_keys(['name', 'age', 'city', 'job'])
print(person.values())   # dict_values(['Alice', 26, 'New York', 'Engineer'])
print(person.items())    # dict_items([('name', 'Alice'), ...])
```

### 5. Collections Module

**defaultdict**
```python
from collections import defaultdict

# Automatically creates new list when key doesn't exist
word_counts = defaultdict(int)
for word in ['a', 'b', 'a', 'c', 'b', 'a']:
    word_counts[word] += 1
# word_counts = {'a': 3, 'b': 2, 'c': 1}
```

**Counter**
```python
from collections import Counter

# Count occurrences of elements
counts = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counts)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(counts.most_common(1))  # [('a', 3)]
```

**deque**
```python
from collections import deque

# Double-ended queue
d = deque([1, 2, 3])
d.appendleft(0)  # [0, 1, 2, 3]
d.pop()          # Removes and returns 3
```

### 6. Choosing the Right Data Structure

| Operation | List | Tuple | Set | Dictionary |
|-----------|------|-------|-----|------------|
| Indexing | O(1) | O(1) | N/A | O(1) |
| Append | O(1) | N/A | N/A | N/A |
| Insert | O(n) | N/A | N/A | N/A |
| Delete | O(n) | N/A | O(1) | O(1) |
| Search | O(n) | O(n) | O(1) | O(1) |
| Membership | O(n) | O(n) | O(1) | O(1) |

### 7. Practical Examples

1. **Counting Word Frequencies**
   ```python
   from collections import Counter
   
   def word_frequency(text):
       words = text.lower().split()
       return Counter(words)
   ```

2. **Grouping Data**
   ```python
   from collections import defaultdict
   
   def group_by_key(pairs):
       result = defaultdict(list)
       for key, value in pairs:
           result[key].append(value)
       return dict(result)
   ```

3. **Removing Duplicates (Preserving Order)**
   ```python
   def remove_duplicates(items):
       seen = set()
       return [x for x in items if not (x in seen or seen.add(x))]
   ```

## Study Strategies

1. **Understand, Don't Memorize**: Focus on understanding why things work the way they do.
2. **Practice Coding**: Write code from scratch without looking at references.
3. **Explain Concepts**: Try to explain Python concepts out loud as if teaching someone else.
4. **Work in Groups**: Discuss problems and solutions with classmates.
5. **Review Error Messages**: Understand common errors and how to fix them.
6. **Time Yourself**: Practice solving problems under time constraints.

## Practice Problems

### Problem 1: Word Frequency Counter
Write a function that takes a string and returns a dictionary with words as keys and their frequency as values.

### Problem 2: List Operations
Given two lists, write a function that returns a new list containing only the common elements (without duplicates).

### Problem 3: Text Analysis
Write a function that analyzes a text file and returns:
- Number of words
- Number of unique words
- Most common word
- Average word length

## Common Pitfalls

1. **Mutable Default Arguments**: Default arguments are evaluated only once when the function is defined.
   ```python
   # Bad
   def add_item(item, items=[]):
       items.append(item)
       return items
   
   # Good
   def add_item(item, items=None):
       if items is None:
           items = []
       items.append(item)
       return items
   ```

2. **Modifying a List While Iterating**: This can lead to unexpected behavior.
   ```python
   # Bad
   numbers = [1, 2, 3, 4]
   for num in numbers:
       if num % 2 == 0:
           numbers.remove(num)
   
   # Good
   numbers = [num for num in numbers if num % 2 != 0]
   ```

3. **Confusing `==` and `is`**:
   - `==` checks for equality of values
   - `is` checks if two variables refer to the same object

## Final Tips

- Read questions carefully and understand what is being asked
- Plan your solution before writing code
- Test your code with different inputs
- Check for edge cases (empty inputs, single item, etc.)
- Manage your time during the exam
- Don't leave any questions blank - partial credit is better than no credit

Good luck with your exam preparation! Remember that understanding the concepts is more important than memorizing code. Practice regularly and don't hesitate to ask for help when needed.
