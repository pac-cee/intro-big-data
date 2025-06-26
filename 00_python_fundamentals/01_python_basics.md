# Python Fundamentals: The Complete Beginner's Guide

## Table of Contents
1. [Introduction to Python](#introduction-to-python)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Variables and Data Types](#variables-and-data-types)
4. [Operators and Expressions](#operators-and-expressions)
5. [Control Flow](#control-flow)
6. [Functions](#functions)
7. [Working with Collections](#working-with-collections)
8. [Input and Output](#input-and-output)
9. [Practice Exercises](#practice-exercises)
10. [Additional Resources](#additional-resources)

## Introduction to Python

Python is a high-level, interpreted, and general-purpose programming language that emphasizes code readability through its clean and straightforward syntax. Created by Guido van Rossum and first released in 1991, Python has grown to become one of the most popular programming languages worldwide.

### Key Features of Python
- **Readable and Maintainable Code**: Python's syntax is designed to be readable with significant whitespace and a clear, English-like structure.
- **Interpreted Language**: Python code is executed line by line, making debugging easier.
- **Dynamically Typed**: No need to declare variable types explicitly.
- **Cross-platform**: Python code can run on various operating systems without modification.
- **Large Standard Library**: Comes with an extensive collection of modules and packages.
- **Supports Multiple Paradigms**: Object-oriented, procedural, and functional programming styles.

### Why Python for Big Data and Beyond?
- **Data Science Powerhouse**: Libraries like NumPy, Pandas, and Matplotlib make data manipulation and visualization efficient.
- **Machine Learning**: Frameworks like TensorFlow, PyTorch, and scikit-learn are built with Python.
- **Web Development**: Frameworks like Django and Flask enable rapid web application development.
- **Automation**: Excellent for writing scripts to automate repetitive tasks.
- **Community and Support**: One of the largest and most active programming communities.
- **Extensibility**: Can be integrated with C/C++ and other languages.

## Setting Up Your Environment

### Installing Python
1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download the latest stable version.
2. **Verify Installation**: Open a terminal/command prompt and type:
   ```bash
   python --version
   ```
3. **Package Management**: Python comes with `pip`, a package manager. Check its version:
   ```bash
   pip --version
   ```

### Choosing an IDE/Code Editor
- **VS Code**: Lightweight, powerful, and highly customizable with Python extensions.
- **PyCharm**: Full-featured Python IDE with excellent debugging capabilities.
- **Jupyter Notebook**: Great for data science and interactive computing.
- **IDLE**: Comes bundled with Python, good for beginners.

## Variables and Data Types

### Understanding Variables
In Python, variables are created when you assign a value to them. No explicit declaration is needed.

```python
# Variable assignment
name = "Alice"
age = 25
height = 5.9
is_student = True
```

### Basic Data Types

#### 1. Numeric Types
- **Integers (`int`)**: Whole numbers, positive or negative, without decimals.
  ```python
  x = 10
  y = -5
  large_number = 1_000_000  # Underscores for better readability
  ```

- **Floating-Point (`float`)**: Numbers with decimal points or in exponential form.
  ```python
  pi = 3.14159
  scientific = 2.5e3  # 2500.0
  ```

- **Complex Numbers (`complex`)**: Numbers with real and imaginary parts.
  ```python
  z = 3 + 4j
  ```

#### 2. Boolean (`bool`)
Represents `True` or `False` values (must be capitalized in Python).

```python
is_active = True
has_permission = False
```

#### 3. Text Type (`str`)
Strings are sequences of Unicode characters, defined with single, double, or triple quotes.

```python
name = "Alice"
greeting = 'Hello, World!'
multiline = """This is a
multi-line
string"""
```

### Type Conversion
Python provides built-in functions to convert between types:

```python
# Converting to integer
num_str = "123"
num_int = int(num_str)  # 123

# Converting to float
num_float = float("3.14")  # 3.14

# Converting to string
text = str(42)  # "42"

# Converting to boolean
bool(1)  # True
bool(0)  # False
bool("")  # False (empty string is falsy)
```

## Operators and Expressions

### Arithmetic Operators
```python
# Basic arithmetic
5 + 3   # Addition (8)
5 - 3   # Subtraction (2)
5 * 3   # Multiplication (15)
5 / 3   # Division (1.666...)
5 // 3  # Floor division (1)
5 % 3   # Modulus (remainder) (2)
5 ** 3  # Exponentiation (125)
```

### Comparison Operators
```python
x = 5
y = 3
x == y  # Equal to (False)
x != y  # Not equal to (True)
x > y   # Greater than (True)
x < y   # Less than (False)
x >= y  # Greater than or equal to (True)
x <= y  # Less than or equal to (False)
```

### Logical Operators
```python
True and False  # AND (False)
True or False   # OR (True)
not True        # NOT (False)
```

### Membership Operators
```python
names = ["Alice", "Bob", "Charlie"]
"Alice" in names     # True
"David" not in names # True
```

### Identity Operators
```python
x = [1, 2, 3]
y = x
z = [1, 2, 3]

x is y     # True (same object)
x is z     # False (different objects with same value)
x == z     # True (same value)
```

## Control Flow

### Conditional Statements
Python uses indentation to define code blocks.

```python
# Basic if-elif-else
age = 18

if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"
```

### Loops

#### For Loops
```python
# Iterating over a sequence
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# Using range()
for i in range(5):        # 0 to 4
    print(i)

for i in range(1, 6):     # 1 to 5
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    print(i)
```

#### While Loops
```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1  # Equivalent to count = count + 1

# Using break and continue
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input.lower() == 'quit':
        break
    elif user_input == '':
        continue
    print(f"You entered: {user_input}")
```

## Functions

### Defining and Calling Functions
```python
def greet(name, greeting="Hello"):
    """
    Greet a person with an optional custom greeting.
    
    Args:
        name (str): The name of the person to greet
        greeting (str, optional): The greeting to use. Defaults to "Hello".
    
    Returns:
        str: A greeting message
    """
    return f"{greeting}, {name}!"

# Function calls
print(greet("Alice"))                # "Hello, Alice!"
print(greet("Bob", "Good morning"))  # "Good morning, Bob!"
```

### Function Arguments

#### Positional and Keyword Arguments
```python
def describe_pet(pet_name, animal_type='dog'):
    print(f"I have a {animal_type} named {pet_name}.")

# Positional arguments
describe_pet("Willie", "hamster")  # Order matters

# Keyword arguments
describe_pet(animal_type="hamster", pet_name="Willie")  # Order doesn't matter

# Default parameter value
describe_pet("Rex")  # Uses default animal_type='dog'
```

#### Arbitrary Arguments
```python
def make_pizza(*toppings):
    """Print the list of toppings that have been requested."""
    print("\nMaking a pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")

make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')
```

#### Keyword Arbitrary Arguments
```python
def build_profile(first, last, **user_info):
    """Build a dictionary containing everything we know about a user."""
    user_info['first_name'] = first
    user_info['last_name'] = last
    return user_info

user_profile = build_profile('albert', 'einstein',
                           location='princeton',
                           field='physics')
print(user_profile)
```

## Working with Collections

### Lists
```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
fruits = ['apple', 'banana', 'cherry']
mixed = [1, 'hello', 3.14, True]

# List operations
fruits.append('orange')      # Add to end
fruits.insert(1, 'mango')    # Insert at position
fruits.remove('banana')      # Remove by value
popped = fruits.pop(0)       # Remove and return item at index

# List slicing
numbers[1:3]    # [2, 3]
numbers[::2]    # Every second item
numbers[::-1]   # Reversed list

# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Tuples
```python
# Creating tuples
coordinates = (10, 20)
colors = 'red', 'green', 'blue'  # Parentheses are optional

# Unpacking
x, y = coordinates
print(f"X: {x}, Y: {y}")

# Single-element tuple
single = (42,)  # Note the comma
```

### Dictionaries
```python
# Creating dictionaries
person = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}

# Accessing values
print(person['name'])  # Alice
print(person.get('age'))  # 30 (safer with .get())

# Adding/updating
person['email'] = 'alice@example.com'
person['age'] = 31

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Sets
```python
# Creating sets
fruits = {'apple', 'banana', 'cherry'}
numbers = set([1, 2, 3, 4, 5])

# Set operations
fruits.add('orange')
fruits.remove('banana')
'apple' in fruits  # True

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union = set1 | set2        # {1, 2, 3, 4, 5}
intersection = set1 & set2 # {3}
difference = set1 - set2   # {1, 2}
```

## Input and Output

### Getting User Input
```python
name = input("What's your name? ")
print(f"Hello, {name}!")

# Converting input to number
age = int(input("How old are you? "))
print(f"Next year you'll be {age + 1} years old.")
```

### File Handling
```python
# Writing to a file
with open('example.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a second line.")

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

# Reading line by line
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes the newline character
```

## Practice Exercises

### Exercise 1: Factorial Calculator
Write a function that calculates the factorial of a non-negative integer.

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


# Test cases
print(factorial(5))  # 120
print(factorial(0))  # 1
```

### Exercise 2: Fibonacci Sequence
Create a program that prints the Fibonacci sequence up to n numbers.

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b
    print()  # For newline

# Test case
fibonacci(10)  # 0 1 1 2 3 5 8 13 21 34
```

### Exercise 3: Palindrome Checker
Write a function that checks if a string is a palindrome.

```python
def is_palindrome(s):
    # Convert to lowercase and remove non-alphanumeric characters
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

# Test cases
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

### Exercise 4: Prime Number Checker
Write a function to check if a number is prime.

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Test cases
print(is_prime(17))  # True
print(is_prime(15))  # False
```

### Exercise 5: Word Frequency Counter
Write a function that counts the frequency of each word in a string.

```python
def word_frequency(text):
    words = text.lower().split()
    frequency = {}
    for word in words:
        # Remove punctuation
        word = word.strip('.,!?;:"(){}[]')
        if word:
            frequency[word] = frequency.get(word, 0) + 1
    return frequency

# Test case
text = "Hello, world! Hello, Python. Python is awesome!"
print(word_frequency(text))
# Output: {'hello': 2, 'world': 1, 'python': 2, 'is': 1, 'awesome': 1}
```

## Additional Resources

### Official Documentation
- [Python Official Documentation](https://docs.python.org/3/)
- [Python Standard Library](https://docs.python.org/3/library/)

### Online Learning Platforms
- [Real Python](https://realpython.com/)
- [Python.org Tutorial](https://docs.python.org/3/tutorial/index.html)
- [W3Schools Python Tutorial](https://www.w3schools.com/python/)
- [Codecademy Python Course](https://www.codecademy.com/learn/learn-python-3)

### Recommended Books
- "Python Crash Course" by Eric Matthes
- "Automate the Boring Stuff with Python" by Al Sweigart
- "Fluent Python" by Luciano Ramalho
- "Python Cookbook" by David Beazley and Brian K. Jones

### Practice Platforms
- [LeetCode](https://leetcode.com/)
- [HackerRank](https://www.hackerrank.com/domains/tutorials/10-days-of-python)
- [Exercism](https://exercism.io/tracks/python)
- [Codewars](https://www.codewars.com/)

---
Next: [Python Intermediate Concepts](./02_python_intermediate.md)
