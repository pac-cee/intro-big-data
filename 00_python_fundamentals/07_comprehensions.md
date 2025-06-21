# List and Dictionary Comprehensions in Python

## Table of Contents
1. [List Comprehensions](#list-comprehensions)
2. [Dictionary Comprehensions](#dictionary-comprehensions)
3. [Set Comprehensions](#set-comprehensions)
4. [Nested Comprehensions](#nested-comprehensions)
5. [Generator Expressions](#generator-expressions)
6. [Performance Considerations](#performance-considerations)
7. [Best Practices](#best-practices)
8. [Practice Exercises](#practice-exercises)

## List Comprehensions

### Basic Syntax
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### With Condition
```python
# Only even numbers
evens = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]

# With if-else
numbers = [1, 2, 3, 4, 5]
result = [x if x % 2 == 0 else 'odd' for x in numbers]
# ['odd', 2, 'odd', 4, 'odd']
```

### Multiple Iterables
```python
# Cartesian product
colors = ['red', 'green', 'blue']
sizes = ['S', 'M', 'L']
products = [(color, size) for color in colors for size in sizes]
# [('red', 'S'), ('red', 'M'), ('red', 'L'), 
#  ('green', 'S'), ('green', 'M'), ('green', 'L'),
#  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]
```

### Flattening a List
```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Dictionary Comprehensions

### Basic Syntax
```python
# Create a dictionary of squares
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### With Condition
```python
# Only even squares
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}
```

### Dictionary Mapping
```python
# Convert list to dictionary with indices
fruits = ['apple', 'banana', 'cherry']
fruit_dict = {i: fruit for i, fruit in enumerate(fruits, 1)}
# {1: 'apple', 2: 'banana', 3: 'cherry'}
```

### Dictionary Filtering
```python
# Filter dictionary items
prices = {'apple': 1.0, 'banana': 0.5, 'orange': 1.5, 'kiwi': 2.0}
expensive = {k: v for k, v in prices.items() if v > 1.0}
# {'orange': 1.5, 'kiwi': 2.0}
```

## Set Comprehensions

### Basic Syntax
```python
# Create a set of unique squares
squares = {x**2 for x in [-2, -1, 0, 1, 2]}
# {0, 1, 4}
```

### With Condition
```python
# Only even squares
even_squares = {x**2 for x in range(10) if x % 2 == 0}
# {0, 4, 16, 64, 36}
```

### Finding Unique Letters
```python
sentence = "the quick brown fox jumps over the lazy dog"
unique_letters = {char for char in sentence if char != ' '}
# {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
#  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
```

## Nested Comprehensions

### Nested List Comprehension
```python
# Transpose a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(3)]
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

### Flattening a 2D List
```python
matrix = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Dictionary of Lists to List of Dictionaries
```python
# Input
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Paris', 'London']
}

# Convert to list of dictionaries
result = [
    {key: data[key][i] for key in data}
    for i in range(len(data['name']))
]
# [
#   {'name': 'Alice', 'age': 25, 'city': 'New York'},
#   {'name': 'Bob', 'age': 30, 'city': 'Paris'},
#   {'name': 'Charlie', 'age': 35, 'city': 'London'}
# ]
```

## Generator Expressions

### Basic Syntax
```python
# Generator expression
squares = (x**2 for x in range(10))

# Consume the generator
for num in squares:
    print(num)  # 0, 1, 4, 9, 16, 25, 36, 49, 64, 81

# Can only be consumed once
print(sum(squares))  # 0 (already consumed)
```

### With Functions
```python
# Sum of squares
sum_sq = sum(x**2 for x in range(10))  # 285

# Maximum with key
max_sq = max((x**2 for x in range(10)), key=lambda x: x % 7)  # 9 (9 % 7 = 2)
```

## Performance Considerations

### Memory Efficiency
```python
# List comprehension (creates full list in memory)
squares_list = [x**2 for x in range(1000000)]  # Uses more memory

# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(1000000))  # Memory efficient
```

### Timing Comparison
```python
import timeit

# List comprehension
time_list = timeit.timeit('[x**2 for x in range(1000)]', number=1000)

# Generator expression
time_gen = timeit.timeit('(x**2 for x in range(1000))', number=1000)

print(f"List: {time_list:.6f} seconds")
print(f"Generator: {time_gen:.6f} seconds")
```

## Best Practices

### When to Use Comprehensions
- For simple transformations and filtering
- When the logic is clear and concise
- When you need a new list/dict/set

### When to Avoid Comprehensions
- When the logic is complex or requires multiple steps
- When you need to use the result multiple times (for generators)
- When it reduces readability

### Readability Tips
- Keep it simple and readable
- Break complex comprehensions into multiple lines
- Use meaningful variable names
- Consider using traditional loops for complex logic

## Practice Exercises

1. **Basic List Comprehension**
   Create a list of the first 10 square numbers using a list comprehension.

2. **Filtering with List Comprehension**
   Given a list of numbers, create a new list with only the even numbers.

3. **Dictionary Comprehension**
   Create a dictionary where the keys are numbers from 1 to 10 and the values are their squares.

4. **Set Comprehension**
   Given a string, create a set of all the vowels in the string.

5. **Nested Comprehension**
   Flatten a 2D list into a 1D list using a nested list comprehension.

6. **Conditional Logic**
   Create a list that contains "Even" for even numbers and "Odd" for odd numbers from 1 to 10.

7. **Dictionary Filtering**
   Given a dictionary of items and their prices, create a new dictionary with only items that cost more than $10.

8. **Matrix Operations**
   Given a 3x3 matrix, create a new matrix where each element is the square of the original element.

9. **String Manipulation**
   Given a list of words, create a new list with the lengths of each word, but only for words with more than 3 characters.

10. **Advanced**
    Convert a list of dictionaries into a dictionary of lists.

---
Next: [Decorators in Python](./08_decorators.md)
