# Decorators in Python

## Table of Contents
1. [Introduction to Decorators](#introduction-to-decorators)
2. [Function Decorators](#function-decorators)
3. [Class Decorators](#class-decorators)
4. [Decorators with Arguments](#decorators-with-arguments)
5. [Chaining Decorators](#chaining-decorators)
6. [Built-in Decorators](#built-in-decorators)
7. [Decorator Use Cases](#decorator-use-cases)
8. [Practice Exercises](#practice-exercises)

## Introduction to Decorators

Decorators are a powerful feature in Python that allow you to modify the behavior of functions or classes without changing their source code.

### Basic Concepts
- **Decorator**: A function that takes another function and extends its behavior
- **Wrapping**: The process of adding functionality to an existing function
- **Syntactic Sugar**: The `@decorator` syntax makes it easy to apply decorators

### Simple Decorator Example
```python
def my_decorator(func):
    def wrapper():
        print("Something before the function is called.")
        func()
        print("Something after the function is called.")
    return wrapper

def say_hello():
    print("Hello!")

# Using the decorator
decorated_hello = my_decorator(say_hello)
decorated_hello()

# Using the @syntax
@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

## Function Decorators

### Decorating Functions with Arguments
```python
def greet_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        result = func(*args, **kwargs)
        print("After calling the function")
        return result
    return wrapper

@greet_decorator
greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

### Preserving Function Metadata
```python
import functools

def debug(func):
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@debug
def add(a, b):
    """Add two numbers."""
    return a + b

print(add.__name__)  # 'add' (without @functools.wraps it would be 'wrapper')
print(add.__doc__)   # 'Add two numbers.'
```

## Class Decorators

### Basic Class Decorator
```python
def add_method(cls):
    def method(self):
        return "This is an added method"
    
    cls.new_method = method
    return cls

@add_method
class MyClass:
    pass

obj = MyClass()
print(obj.new_method())  # "This is an added method"
```

### Class as Decorator
```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.num_calls = 0
    
    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print(f"Call {self.num_calls} of {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()
say_hello()
print(f"Called {say_hello.num_calls} times")
```

## Decorators with Arguments

### Decorator Factory
```python
def repeat(num_times):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_repeat

@repeat(num_times=3)
def greet(name):
    print(f"Hello {name}")

greet("Alice")  # Prints "Hello Alice" 3 times
```

### Decorator with Optional Arguments
```python
def log_activity(log_file=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_message = f"{func.__name__} was called"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(log_message + '\n')
            else:
                print(log_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log_activity(log_file='activity.log')
def process_data(data):
    return data.upper()

@log_activity()
def say_hello():
    print("Hello!")
```

## Chaining Decorators

### Multiple Decorators
```python
def bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

def italic(func):
    def wrapper():
        return f"<i>{func()}</i>"
    return wrapper

@bold
@italic
def hello():
    return "Hello, World!"

print(hello())  # <b><i>Hello, World!</i></b>
```

### Order of Execution
```python
def decorator1(func):
    print("decorator1")
    def wrapper():
        print("wrapper1")
        return func()
    return wrapper

def decorator2(func):
    print("decorator2")
    def wrapper():
        print("wrapper2")
        return func()
    return wrapper

@decorator1
@decorator2
def say_hello():
    print("Hello!")

# Output when module is imported:
# decorator2
# decorator1

say_hello()
# Output:
# wrapper1
# wrapper2
# Hello!
```

## Built-in Decorators

### @property, @setter, @deleter
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def diameter(self):
        return 2 * self._radius
    
    @diameter.setter
    def diameter(self, value):
        self.radius = value / 2

circle = Circle(5)
print(circle.radius)    # 5
print(circle.diameter)  # 10
circle.diameter = 14
print(circle.radius)    # 7.0
```

### @classmethod and @staticmethod
```python
class MyClass:
    class_var = "class variable"
    
    def __init__(self, value):
        self.instance_var = value
    
    def instance_method(self):
        return f"instance method: {self.instance_var}"
    
    @classmethod
    def class_method(cls):
        return f"class method: {cls.class_var}"
    
    @staticmethod
    def static_method():
        return "static method"

# Instance method requires an instance
obj = MyClass("value")
print(obj.instance_method())

# Class method can be called on the class or instance
print(MyClass.class_method())
print(obj.class_method())

# Static method can be called on the class or instance
print(MyClass.static_method())
print(obj.static_method())
```

## Decorator Use Cases

### Timing Function Execution
```python
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return result
    return wrapper

@timer
def waste_time(n):
    for _ in range(n):
        sum([i**2 for i in range(1000)])

waste_time(1000)
```

### Caching (Memoization)
```python
def memoize(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Without memoization, this would be very slow
print(fibonacci(50))
```

### Access Control
```python
def admin_required(func):
    @functools.wraps(func)
    def wrapper(user, *args, **kwargs):
        if user.get('is_admin'):
            return func(user, *args, **kwargs)
        else:
            raise PermissionError("Admin access required")
    return wrapper

@admin_required
def delete_user(user, user_id):
    return f"User {user_id} deleted by {user['name']}"

admin = {'name': 'Alice', 'is_admin': True}
regular = {'name': 'Bob', 'is_admin': False}

print(delete_user(admin, 1))  # Works
print(delete_user(regular, 1))  # Raises PermissionError
```

## Practice Exercises

1. **Basic Decorator**
   Create a decorator that prints "Function started" before the function runs and "Function completed" after it finishes.

2. **Timing Decorator**
   Write a decorator that measures and prints the execution time of a function.

3. **Retry Decorator**
   Create a decorator that retries a function a specified number of times if it raises an exception.

4. **Validation Decorator**
   Write a decorator that validates function arguments (e.g., check if arguments are positive).

5. **Logging Decorator**
   Create a decorator that logs function calls with their arguments and return values to a file.

6. **Rate Limiter**
   Implement a decorator that limits how often a function can be called (e.g., max 5 calls per minute).

7. **Class Decorator**
   Create a class decorator that adds a `to_dict` method to any class, converting its attributes to a dictionary.

8. **Memoization**
   Implement a memoization decorator that caches function results based on its arguments.

9. **Authentication**
   Write a decorator that checks if a user is authenticated before allowing access to a function.

10. **Deprecation Warning**
    Create a decorator that issues a deprecation warning when a function is used.

---
Next: [Context Managers in Python](./09_context_managers.md)
