# Python Intermediate Concepts: Taking Your Skills Further

## Table of Contents
1. [Advanced Data Structures](#advanced-data-structures)
   - [Dictionaries Deep Dive](#dictionaries-deep-dive)
   - [Sets and Their Operations](#sets-and-their-operations)
   - [Collections Module](#collections-module)
2. [List and Dictionary Comprehensions](#list-and-dictionary-comprehensions)
   - [Basic to Advanced Examples](#basic-to-advanced-examples)
   - [Nested Comprehensions](#nested-comprehensions)
   - [Performance Considerations](#performance-considerations)
3. [Lambda Functions and Functional Programming](#lambda-functions-and-functional-programming)
   - [Understanding Lambda](#understanding-lambda)
   - [Built-in Functions: map, filter, reduce](#built-in-functions)
   - [Functional Programming Concepts](#functional-programming-concepts)
4. [Working with Dates and Times](#working-with-dates-and-times)
   - [datetime Module](#datetime-module)
   - [Time Zones and Localization](#time-zones-and-localization)
   - [Parsing and Formatting](#parsing-and-formatting)
5. [Iterators and Generators](#iterators-and-generators)
   - [Iterator Protocol](#iterator-protocol)
   - [Generator Functions](#generator-functions)
   - [Generator Expressions](#generator-expressions)
6. [Practice Exercises](#practice-exercises)
7. [Additional Resources](#additional-resources)

## Advanced Data Structures

### Dictionaries Deep Dive

Dictionaries are one of Python's most powerful data structures, providing efficient key-value storage and retrieval.

#### Core Dictionary Operations

```python
# Creating dictionaries
person = {"name": "Alice", "age": 30, "city": "New York"}
empty_dict = {}

# Accessing elements
name = person["name"]  # Direct access (raises KeyError if key doesn't exist)
age = person.get("age")  # Returns None if key doesn't exist
age = person.get("age", 0)  # Returns 0 if key doesn't exist

# Dictionary views (dynamic)
keys = person.keys()     # dict_keys(['name', 'age', 'city'])
values = person.values() # dict_values(['Alice', 30, 'New York'])
items = person.items()   # dict_items([('name', 'Alice'), ('age', 30), ('city', 'New York')])

# Modifying dictionaries
person["age"] = 31          # Update existing key
person["email"] = "alice@example.com"  # Add new key-value pair
person.update({"age": 32, "country": "USA"})  # Update multiple items

# Removing elements
age = person.pop("age")     # Remove and return value
person.popitem()             # Remove and return last inserted item (Python 3.7+)
del person["city"]          # Remove key
person.clear()              # Remove all items
```

#### Dictionary Methods

```python
# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Dictionary from two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
mapping = dict(zip(keys, values))  # {'a': 1, 'b': 2, 'c': 3}

# Dictionary merging (Python 3.5+)
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = {**dict1, **dict2}  # {'a': 1, 'b': 3, 'c': 4}

# Python 3.9+ merge operator
merged = dict1 | dict2  # {'a': 1, 'b': 3, 'c': 4}
```

### Sets and Their Operations

Sets are unordered collections of unique elements, optimized for membership testing and eliminating duplicates.

#### Basic Set Operations

```python
# Creating sets
primes = {2, 3, 5, 7, 11}
even = {2, 4, 6, 8, 10}
empty_set = set()  # {} creates an empty dictionary, not a set

# Set operations
union = primes | even           # {2, 3, 4, 5, 6, 7, 8, 10, 11}
intersection = primes & even    # {2}
difference = primes - even      # {3, 5, 7, 11}
sym_diff = primes ^ even        # {3, 4, 5, 6, 7, 8, 10, 11}

# Set methods
primes.add(13)                  # Add single element
primes.update([17, 19, 23])     # Add multiple elements
primes.remove(2)                # Remove element (raises KeyError if not found)
primes.discard(2)               # Remove element if exists (no error)
popped = primes.pop()          # Remove and return an arbitrary element

# Set membership
is_prime = 7 in primes          # True
is_even = 4 in primes           # False

# Frozenset (immutable set)
immutable_primes = frozenset([2, 3, 5, 7])
```

#### Set Comprehensions

```python
# Set comprehension
unique_lengths = {len(word) for word in ["apple", "banana", "cherry"]}  # {5, 6}

# Filtering with set comprehension
vowels = {'a', 'e', 'i', 'o', 'u'}
word = "hello"
unique_vowels = {char for char in word if char in vowels}  # {'e', 'o'}
```

### Collections Module

The `collections` module provides specialized container datatypes that can be more efficient than the general-purpose ones.

#### Counter

```python
from collections import Counter

# Count occurrences of elements
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_counts = Counter(words)  # {'apple': 3, 'banana': 2, 'orange': 1}

# Most common elements
print(word_counts.most_common(2))  # [('apple', 3), ('banana', 2)]

# Update counter
word_counts.update(["apple", "orange"])

# Arithmetic operations
counter1 = Counter(a=3, b=1)
counter2 = Counter(a=1, b=2)
print(counter1 + counter2)  # Counter({'a': 4, 'b': 3})
print(counter1 - counter2)  # Counter({'a': 2})
```

#### defaultdict

```python
from collections import defaultdict

# Default dictionary with list as default factory
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
# d = {'yellow': [1, 3], 'blue': [2, 4], 'red': [1]}

# Default dictionary with int as default factory
word_counts = defaultdict(int)
for word in ['apple', 'banana', 'apple']:
    word_counts[word] += 1
# word_counts = {'apple': 2, 'banana': 1}
```

#### namedtuple

```python
from collections import namedtuple

# Create a named tuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, y=20)  # Instantiate with positional or keyword arguments
print(p.x, p.y)      # Access fields by name
print(p[0], p[1])     # Access fields by index

# _make() and _asdict()
point_list = [30, 40]
p2 = Point._make(point_list)  # Create from iterable
print(p2._asdict())           # Convert to OrderedDict
```

## List and Dictionary Comprehensions

Comprehensions provide a concise way to create lists, dictionaries, and sets in Python. They are more readable and often faster than traditional loops.

### List Comprehensions

#### Basic Syntax
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]

# With if-else
categorized = ["even" if x % 2 == 0 else "odd" for x in range(5)]  # ['even', 'odd', 'even', 'odd', 'even']
```

#### Nested List Comprehensions

```python
# Flattening a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Transposing a matrix
transposed = [[row[i] for row in matrix] for i in range(3)]
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# Flattening a 3D list
matrix_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
flattened_3d = [num for matrix_2d in matrix_3d 
                for row in matrix_2d 
                for num in row]  # [1, 2, 3, 4, 5, 6, 7, 8]
```

### Dictionary Comprehensions

```python
# Basic dictionary comprehension
square_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Dictionary comprehension with condition
filtered_dict = {k: v for k, v in [('a', 1), ('b', 2), ('c', 3)] if v > 1}  # {'b': 2, 'c': 3}

# Swapping keys and values
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}  # {1: 'a', 2: 'b', 3: 'c'}

# Creating a dictionary from two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
mapped = {k: v for k, v in zip(keys, values)}  # {'a': 1, 'b': 2, 'c': 3}
```

### Set Comprehensions

```python
# Basic set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "python"]}  # {5, 6}

# Filtering with set comprehension
vowels = {'a', 'e', 'i', 'o', 'u'}
word = "hello"
unique_vowels = {char for char in word if char in vowels}  # {'e', 'o'}

# Creating a set of unique first letters
names = ["Alice", "Bob", "Charlie", "David"]
first_letters = {name[0] for name in names}  # {'A', 'B', 'C', 'D'}
```

### Performance Considerations

```python
# List comprehension vs map+filter
# Using list comprehension (generally more readable)
squares = [x**2 for x in range(1000) if x % 2 == 0]

# Using map and filter (might be slightly faster for very large datasets)
squares = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(1000))))

# Generator expressions for memory efficiency
# This doesn't create the entire list in memory at once
sum_of_squares = sum(x**2 for x in range(1000) if x % 2 == 0)

# Files are iterators - process line by line without loading entire file
with open('large_file.txt') as f:
    total_chars = sum(len(line) for line in f)
```

### Nested Comprehensions with Multiple Conditions

```python
# Flatten a list of lists with filtering
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_even = [num for row in matrix 
                 for num in row 
                 if num % 2 == 0]  # [2, 4, 6, 8]

# Dictionary comprehension with nested loops
coordinates = {(x, y): x*y for x in range(3) 
                             for y in range(3) 
                             if x != y}
# {(0, 1): 0, (0, 2): 0, (1, 0): 0, (1, 2): 2, (2, 0): 0, (2, 1): 2}
```

### When to Use Comprehensions

- **Use comprehensions when**:
  - The operation is simple and fits on one line
  - The primary purpose is to transform or filter data
  - Readability is improved over a traditional loop
  
- **Avoid comprehensions when**:
  - The logic is complex and requires multiple lines
  - You need to handle exceptions
  - The operation has side effects
  - The comprehension becomes hard to read

## Lambda Functions and Functional Programming

Lambda functions, also known as anonymous functions, are small, one-line functions defined with the `lambda` keyword. They are particularly useful for short, simple operations that are used only once.

### Understanding Lambda Functions

#### Basic Syntax
```python
# Basic lambda function
add = lambda x, y: x + y
print(add(3, 5))  # 8

# Equivalent to
def add(x, y):
    return x + y
```

#### Key Characteristics
- Can take any number of arguments but only one expression
- The expression is evaluated and returned
- Can be assigned to a variable or used inline
- Often used with `filter()`, `map()`, and `sorted()`

### Common Use Cases

#### 1. Sorting with `sorted()`
```python
# Sort by length
names = ["Alice", "Bob", "Charlie", "David"]
sorted_names = sorted(names, key=lambda x: len(x))  # ['Bob', 'Alice', 'David', 'Charlie']

# Sort by last character
sorted_last_char = sorted(names, key=lambda x: x[-1])  # ['Alice', 'Charlie', 'David', 'Bob']

# Sort list of dictionaries
people = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
sorted_people = sorted(people, key=lambda x: x["age"])  # Sort by age
```

#### 2. Filtering with `filter()`
```python
# Filter even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6, 8, 10]

# Filter strings containing 'a'
words = ["apple", "banana", "cherry", "date"]
a_words = list(filter(lambda x: 'a' in x, words))  # ['apple', 'banana', 'date']
```

#### 3. Transforming with `map()`
```python
# Double each number
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))  # [2, 4, 6, 8, 10]

# Get lengths of strings
words = ["apple", "banana", "cherry"]
lengths = list(map(lambda x: len(x), words))  # [5, 6, 6]

# Convert strings to title case
names = ["alice", "bob", "charlie"]
title_names = list(map(lambda x: x.title(), names))  # ['Alice', 'Bob', 'Charlie']
```

### Advanced Lambda Usage

#### 4. With `reduce()`
```python
from functools import reduce

# Calculate product of numbers
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)  # 120

# Find maximum number
max_num = reduce(lambda x, y: x if x > y else y, numbers)  # 5

# Concatenate strings
words = ["Hello", " ", "World", "!"]
sentence = reduce(lambda x, y: x + y, words)  # 'Hello World!'
```

#### 5. With `sort()`
```python
# Sort list of tuples by second element
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
pairs.sort(key=lambda pair: pair[1])  # [(1, 'one'), (3, 'three'), (2, 'two')]

# Case-insensitive sort
fruits = ["banana", "Apple", "cherry", "Date"]
fruits.sort(key=lambda x: x.lower())  # ['Apple', 'banana', 'cherry', 'Date']
```

### Limitations of Lambda Functions

1. **Single Expression Only**: Can only contain expressions, not statements.
   ```python
   # Invalid: Can't use statements like if, for, while in lambda
   invalid = lambda x: if x > 0: return x else return -x  # SyntaxError
   ```

2. **No Annotations**: Can't use type hints in Python 3.
   ```python
   # Invalid: Type hints not allowed in lambda
   add = lambda x: int, y: int -> int: x + y  # SyntaxError
   ```

3. **No Documentation Strings**: Can't include docstrings.
   ```python
   # No way to add docstring to lambda
   ```

### When to Use Lambda Functions

- **Use lambda when**:
  - The function is a simple expression (fits in one line)
  - The function is used only once or a few times
  - The function is passed as an argument to higher-order functions
  - The function doesn't need a name

- **Avoid lambda when**:
  - The function is complex (use `def` instead)
  - The function needs a docstring
  - The function needs to be called recursively
  - The function needs to be tested or debugged separately

### Alternatives to Lambda

For more complex operations, consider using:
1. **Named Functions**: More readable and maintainable
   ```python
   def is_even(x):
       return x % 2 == 0
   
   evens = list(filter(is_even, numbers))
   ```

2. **List/Dict/Set Comprehensions**: More readable for simple transformations
   ```python
   # Instead of map
   doubled = [x * 2 for x in numbers]
   
   # Instead of filter
   evens = [x for x in numbers if x % 2 == 0]
   ```

3. **operator Module**: For common operations
   ```python
   from operator import add, itemgetter
   
   # Instead of lambda x, y: x + y
   result = reduce(add, [1, 2, 3, 4])  # 10
   
   # Instead of lambda x: x[1]
   sorted(pairs, key=itemgetter(1))
   ```

## Working with Dates and Times

Python's `datetime` module provides classes for manipulating dates and times in both simple and complex ways. This section covers the essential functionality for working with dates, times, and time intervals.

### The datetime Module

#### Basic Date and Time Objects

```python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()  # datetime.datetime(2023, 4, 1, 15, 30, 45, 123456)


# Date object
today = date.today()  # datetime.date(2023, 4, 1)


# Time object
current_time = time(15, 30, 45)  # datetime.time(15, 30, 45)


# Create datetime from components
dt = datetime(2023, 4, 1, 15, 30, 45)  # April 1, 2023 15:30:45

# Combine date and time
combined = datetime.combine(today, current_time)
```

#### Formatting and Parsing

```python
# Format datetime as string
formatted = now.strftime("%Y-%m-%d %H:%M:%S")  # '2023-04-01 15:30:45'
formatted_date = now.strftime("%A, %B %d, %Y")  # 'Saturday, April 01, 2023'

# Parse string to datetime
parsed = datetime.strptime("2023-04-01", "%Y-%m-%d")
parsed_with_time = datetime.strptime("01/04/2023 15:30", "%d/%m/%Y %H:%M")

# Common format codes:
# %Y - Year with century (2023)
# %m - Month as zero-padded number (01-12)
# %d - Day of month (01-31)
# %H - Hour (00-23)
# %M - Minute (00-59)
# %S - Second (00-59)
# %A - Weekday name (Monday, Tuesday, etc.)
# %B - Month name (January, February, etc.)
```

#### Time Deltas and Date Arithmetic

```python
# Time differences
diff = datetime(2023, 4, 2) - datetime(2023, 4, 1)  # 1 day
print(diff.days)  # 1
print(diff.seconds)  # 0

# Adding/subtracting time
tomorrow = now + timedelta(days=1)
next_week = now + timedelta(weeks=1)
three_hours_later = now + timedelta(hours=3)

# Calculate days until a future date
future_date = datetime(2023, 12, 31)
days_remaining = (future_date - now).days

# Business days calculation (excluding weekends)
def business_days(start, end):
    day = timedelta(days=1)
    count = 0
    while start <= end:
        if start.weekday() < 5:  # Monday = 0, Sunday = 6
            count += 1
        start += day
    return count
from datetime import timezone
utc_now = datetime.now(timezone.utc)

## Iterators and Generators

Iterators and generators are powerful Python features for working with sequences of data, especially when dealing with large datasets or infinite sequences. They enable lazy evaluation, which means values are generated on-the-fly rather than stored in memory.

### Understanding Iterators

An iterator is an object that implements the iterator protocol, which consists of the `__iter__()` and `__next__()` methods. Let's explore this in detail.

#### The Iterator Protocol

```python
# A simple iterator class
class CountDown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        # Returns the iterator object itself
        return self

    def __next__(self):
        # Returns the next value or raises StopIteration
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Using the iterator
countdown = CountDown(3)
for num in countdown:
    print(num)  # Output: 3, 2, 1

# Under the hood, the for loop does:
countdown = CountDown(3)
iterator = iter(countdown)  # Calls __iter__
try:
    while True:
        num = next(iterator)  # Calls __next__
        print(num)
except StopIteration:
    pass
```

#### Built-in Iterables and Iterators

```python
# Common iterables
numbers = [1, 2, 3]        # List is iterable
iterator = iter(numbers)    # Get iterator
print(next(iterator))       # 1
print(next(iterator))       # 2

# File objects are iterators
with open('example.txt') as f:
    for line in f:         # Files implement iterator protocol
        print(line.strip())

# Dictionary iteration
person = {"name": "Alice", "age": 30}
for key in person:          # Iterate over keys
    print(key)
for value in person.values():  # Iterate over values
    print(value)
for key, value in person.items():  # Iterate over key-value pairs
    print(f"{key}: {value}")
```

### Understanding Generators

Generators are a simple way to create iterators using functions. They use the `yield` keyword to return values one at a time.

#### Generator Functions

```python
def count_down(start):
    """Generator that counts down from start to 1."""
    current = start
    while current > 0:
        # Yield the current value and pause execution
        yield current
        current -= 1

# Using the generator
for num in count_down(3):
    print(num)  # 3, 2, 1

# Under the hood
gen = count_down(3)  # Returns a generator object
print(next(gen))     # 3
print(next(gen))     # 2
print(next(gen))     # 1
# print(next(gen))   # Raises StopIteration
```

#### Generator Expressions

```python
# Similar to list comprehensions but with parentheses
squares = (x**2 for x in range(5))
print(list(squares))  # [0, 1, 4, 9, 16]

# Memory efficient for large datasets
# This doesn't create the entire list in memory
sum_of_squares = sum(x**2 for x in range(1000000))

# Filtering with generator expressions
even_squares = (x**2 for x in range(10) if x % 2 == 0)
print(list(even_squares))  # [0, 4, 16, 36, 64]
```

### Advanced Generator Features

#### Sending Values to Generators

```python
def running_averager():
    total = 0
    count = 0
    while True:
        value = yield  # Receives the value sent to the generator
        if value is None:
            break
        total += value
        count += 1
        yield total / count  # Yields the running average

# Using the generator
avg = running_averager()
next(avg)  # Start the generator (runs until first yield)

avg.send(10)  # Send 10 to the generator
print(avg.send(None))  # Get the average: 10.0

avg.send(20)  # Send 20 to the generator
print(avg.send(None))  # Get the average: 15.0

avg.send(30)  # Send 30 to the generator
print(avg.send(None))  # Get the average: 20.0
```

#### yield from (Python 3.3+)

```python
def chain(*iterables):
    for it in iterables:
        yield from it  # Delegates to another generator

result = list(chain('ABC', 'DEF'))
print(result)  # ['A', 'B', 'C', 'D', 'E', 'F']

# Equivalent to:
def chain_equivalent(*iterables):
    for it in iterables:
        for i in it:
            yield i
```

### Real-world Use Cases

#### 1. Processing Large Files

```python
def read_large_file(file_path):
    """Read a large file line by line without loading it entirely into memory."""
    with open(file_path, 'r') as f:
        for line in f:
            # Process each line
            processed_line = line.strip().upper()
            yield processed_line

# Usage
# for line in read_large_file('huge_file.txt'):
#     print(line)
```

#### 2. Infinite Sequences

```python
def fibonacci():
    """Generate an infinite sequence of Fibonacci numbers."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get first 10 Fibonacci numbers
fib = fibonacci()
first_ten = [next(fib) for _ in range(10)]
print(first_ten)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

#### 3. Pipelining Generators

```python
def parse_logs(log_file):
    """Parse log file and yield structured log entries."""
    with open(log_file) as f:
        for line in f:
            # Parse each line into a dictionary
            # This is a simplified example
            parts = line.strip().split()
            if len(parts) >= 3:
                yield {
                    'timestamp': parts[0],
                    'level': parts[1],
                    'message': ' '.join(parts[2:])
                }

def filter_logs(logs, level):
    """Filter logs by log level."""
    return (log for log in logs if log['level'] == level)

# Create a processing pipeline
# logs = parse_logs('app.log')
# error_logs = filter_logs(logs, 'ERROR')
# for log in error_logs:
#     print(f"{log['timestamp']} - {log['message']}")
```

### Generator Best Practices

1. **Use generators for large datasets**: They're memory efficient as they generate items one at a time.

2. **Be careful with multiple iterations**: Generator objects are consumed after iteration.
   ```python
   squares = (x**2 for x in range(5))
   print(list(squares))  # [0, 1, 4, 9, 16]
   print(list(squares))  # [] (already consumed)
   ```

3. **Use `itertools` for advanced patterns**: The `itertools` module provides many useful generator functions.
   ```python
   from itertools import count, cycle, repeat, islice, chain
   
   # Count from 10 to infinity
   for i in islice(count(10), 5):
       print(i)  # 10, 11, 12, 13, 14
   
   # Cycle through a sequence
   for i, item in enumerate(cycle('ABC')):
       if i > 5: break
       print(item)  # A, B, C, A, B, C, A
   ```

4. **Consider memory vs. speed**: While generators save memory, they might be slower than lists for small datasets due to the overhead of function calls.

5. **Use `yield from` for cleaner code**: It simplifies the code when you need to delegate to another generator.

6. **Handle resources properly**: If a generator acquires resources, make sure to clean them up using `try/finally` or context managers.

### When to Use Generators

- **Use generators when**:
  - Working with large datasets that don't fit in memory
  - Creating infinite sequences
  - Implementing data processing pipelines
  - You only need to process items one at a time

- **Avoid generators when**:
  - You need random access to elements
  - You need to iterate over the data multiple times
  - The dataset is small and memory isn't a concern
  - You need list methods like `sort()` or `reverse()`squares from 0 to 81

## Practice Exercises

### Exercise 1: Advanced Data Structures

#### 1.1 Dictionary Deep Dive
```python
# Given a list of dictionaries representing products, create a function that:
# 1. Groups products by category
# 2. Calculates the average price per category
# 3. Returns a dictionary with categories as keys and average price as values

def average_price_by_category(products):
    """
    Calculate average price of products by category.
    
    Args:
        products: List of dicts with 'name', 'category', and 'price' keys
        
    Returns:
        dict: {category: average_price}
    """
    from collections import defaultdict
    
    # Your code here
    pass

# Test case
products = [
    {"name": "Laptop", "category": "Electronics", "price": 999.99},
    {"name": "Smartphone", "category": "Electronics", "price": 699.99},
    {"name": "Desk Chair", "category": "Furniture", "price": 149.99},
    {"name": "Coffee Mug", "category": "Kitchen", "price": 12.99},
    {"name": "Monitor", "category": "Electronics", "price": 249.99},
    {"name": "Dining Table", "category": "Furniture", "price": 599.99}
]

# Expected output: {'Electronics': 649.99, 'Furniture': 374.99, 'Kitchen': 12.99}
```

#### 1.2 Set Operations
```python
# Write a function that finds the Jaccard similarity between two sets
# Jaccard similarity = size of intersection / size of union
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    # Your code here
    pass

# Test case
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}
# Expected output: 0.25 (2/8)
```

### Exercise 2: List and Dictionary Comprehensions

#### 2.1 Advanced List Comprehension
```python
# Flatten a 3D list and filter out odd numbers, then square the remaining numbers
def process_3d_list(matrix_3d):
    """
    Process a 3D list by flattening, filtering odds, and squaring evens.
    
    Args:
        matrix_3d: A 3D list of integers
        
    Returns:
        list: Processed 1D list
    """
    # Your code here (one line with nested comprehensions)
    pass

# Test case
matrix = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]
# Expected output: [4, 16, 36, 64, 100, 144]
```

### Exercise 3: Lambda and Functional Programming

#### 3.1 Higher-Order Functions
```python
# Implement a function that takes a list of functions and a value,
# and returns a list of results from applying each function to the value
def apply_functions(functions, value):
    """
    Apply a list of functions to a value.
    
    Args:
        functions: List of functions
        value: Value to apply functions to
        
    Returns:
        list: Results of applying each function to the value
    """
    # Your code here (one line with map and lambda)
    pass

# Test case
functions = [
    lambda x: x * 2,
    lambda x: x ** 2,
    lambda x: x + 10,
    lambda x: x - 5
]
# apply_functions(functions, 5) should return [10, 25, 15, 0]
```

## Solutions

### 1.1 Dictionary Deep Dive Solution
```python
def average_price_by_category(products):
    from collections import defaultdict
    
    category_totals = defaultdict(float)
    category_counts = defaultdict(int)
    
    for product in products:
        category = product['category']
        price = product['price']
        category_totals[category] += price
        category_counts[category] += 1
    
    return {
        category: round(total / category_counts[category], 2)
        for category, total in category_totals.items()
    }
```

### 1.2 Set Operations Solution
```python
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0
```

### 2.1 Advanced List Comprehension Solution
```python
def process_3d_list(matrix_3d):
    return [num ** 2
            for matrix_2d in matrix_3d
            for row in matrix_2d
            for num in row
            if num % 2 == 0]
```

### 3.1 Higher-Order Functions Solution
```python
def apply_functions(functions, value):
    return list(map(lambda f: f(value), functions))
```

## Additional Resources

### Official Documentation
- [Python Standard Library](https://docs.python.org/3/library/)
- [datetime module](https://docs.python.org/3/library/datetime.html)
- [itertools module](https://docs.python.org/3/library/itertools.html)
- [functools module](https://docs.python.org/3/library/functools.html)

### Recommended Books
- "Fluent Python" by Luciano Ramalho
- "Python Cookbook" by David Beazley and Brian K. Jones
- "Effective Python" by Brett Slatkin

### Online Courses
- [Advanced Python on Real Python](https://realpython.com/)
- [Python Data Structures and Algorithms on Coursera](https://www.coursera.org/)
- [Advanced Python Programming on Udemy](https://www.udemy.com/)

### Practice Platforms
- [LeetCode](https://leetcode.com/)
- [HackerRank](https://www.hackerrank.com/)
- [Exercism](https://exercism.io/)

---
Next: [Functions and Modules](./03_functions_and_modules.md)
