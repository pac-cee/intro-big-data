# Python Intermediate Concepts

## Table of Contents
1. [Advanced Data Structures](#advanced-data-structures)
2. [List Comprehensions](#list-comprehensions)
3. [Lambda Functions](#lambda-functions)
4. [Working with Dates and Times](#working-with-dates-and-times)
5. [Iterators and Generators](#iterators-and-generators)
6. [Practice Exercises](#practice-exercises)

## Advanced Data Structures

### Dictionaries

```python
# Dictionary methods
person = {"name": "Alice", "age": 30, "city": "New York"}

# Get all keys
keys = person.keys()

# Get all values
values = person.values()

# Get key-value pairs
items = person.items()

# Get with default
age = person.get("age", 0)  # Returns 0 if key doesn't exist

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

### Sets

```python
# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union
union = set1 | set2  # or set1.union(set2)


# Intersection
intersection = set1 & set2  # or set1.intersection(set2)


# Difference
diff = set1 - set2  # or set1.difference(set2)


# Symmetric difference
sym_diff = set1 ^ set2  # or set1.symmetric_difference(set2)
```

## List Comprehensions

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]


# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "python"]}
```

## Lambda Functions

```python
# Basic lambda
add = lambda x, y: x + y
print(add(3, 5))  # Output: 8

# With sorted
names = ["Alice", "Bob", "Charlie", "David"]
sorted_names = sorted(names, key=lambda x: len(x))

# With filter
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

# With map
doubled = list(map(lambda x: x * 2, numbers))
```

## Working with Dates and Times

```python
from datetime import datetime, timedelta

# Current date and time
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Create a specific date
date = datetime(2023, 5, 15)

# Date arithmetic
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)

# Parsing date from string
date_str = "2023-05-15"
parsed_date = datetime.strptime(date_str, "%Y-%m-%d")

# Timezones
from datetime import timezone
utc_now = datetime.now(timezone.utc)
```

## Iterators and Generators

### Iterators

```python
# Create an iterator
class CountDown:
    def __init__(self, start):
        self.current = start
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        else:
            self.current -= 1
            return self.current + 1

# Using the iterator
for num in CountDown(5):
    print(num)  # Prints 5, 4, 3, 2, 1
```

### Generators

```python
# Generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Using the generator
for num in countdown(5):
    print(num)  # Prints 5, 4, 3, 2, 1

# Generator expression
squares = (x**2 for x in range(10))
print(sum(squares))  # Sum of squares from 0 to 81
```

## Practice Exercises

1. **List Comprehension**
   Create a list comprehension that generates all prime numbers up to 100.

2. **Dictionary Manipulation**
   Given a list of words, create a dictionary where keys are words and values are their lengths. Then, create a new dictionary with only words longer than 3 characters.

3. **Date Handling**
   Write a function that takes a date string in the format "YYYY-MM-DD" and returns the day of the week.

4. **Generator**
   Create a generator that yields Fibonacci numbers up to a given limit.

5. **Lambda Functions**
   Use `filter()` and `lambda` to extract all even numbers from a list. Then use `map()` to square each number.

6. **Set Operations**
   Given two lists of numbers, find the numbers that are in one list but not the other.

7. **Advanced**
   Create a generator that takes an iterable and yields tuples of (index, value) for each element.

---
Next: [Functions and Modules](./03_functions_and_modules.md)
