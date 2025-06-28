def greet(name):
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"

def add(a, b):
    """Return the sum of two numbers."""
    return a + b

def factorial(n):
    """Return the factorial of a non-negative integer n."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def reverse_string(s):
    """Return the reversed version of a string."""
    return s[::-1]

def fibonacci(n):
    """Return the nth Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def count_vowels(s):
    """Count the number of vowels in a string."""
    return sum(1 for char in s.lower() if char in 'aeiou')

def unique_elements(lst):
    """Return a list of unique elements from the input list."""
    return list(set(lst))

def sort_numbers(nums):
    """Return a sorted copy of the list of numbers."""
    return sorted(nums)

def square_list(lst):
    """Return a list with the squares of the input numbers."""
    return [x ** 2 for x in lst]

def merge_dicts(d1, d2):
    """Merge two dictionaries and return the result."""
    return {**d1, **d2}

def read_file(filename):
    """Read and return the contents of a file."""
    with open(filename, 'r') as f:
        return f.read()

def write_file(filename, content):
    """Write content to a file."""
    with open(filename, 'w') as f:
        f.write(content)

def get_even_numbers(lst):
    """Return a list of even numbers from the input list."""
    return [x for x in lst if x % 2 == 0]

def get_odd_numbers(lst):
    """Return a list of odd numbers from the input list."""
    return [x for x in lst if x % 2 != 0]

def flatten_list(nested_list):
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]

def capitalize_words(s):
    """Capitalize the first letter of each word in a string."""
    return ' '.join(word.capitalize() for word in s.split())

def get_max(lst):
    """Return the maximum value in a list."""
    return max(lst)

def get_min(lst):
    """Return the minimum value in a list."""
    return min(lst)

def sum_list(lst):
    """Return the sum of all elements in a list."""
    return sum(lst)