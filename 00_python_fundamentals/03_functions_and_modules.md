# Functions and Modules in Python

## Table of Contents
1. [Defining Functions](#defining-functions)
2. [Function Arguments](#function-arguments)
3. [Return Values](#return-values)
4. [Variable Scope](#variable-scope)
5. [Modules and Packages](#modules-and-packages)
6. [The `__name__` Variable](#the-__name__-variable)
7. [Virtual Environments](#virtual-environments)
8. [Practice Exercises](#practice-exercises)

## Defining Functions

### Basic Function
```python
def greet(name):
    """
    Greet a person by name.
    
    Args:
        name (str): The name of the person to greet
    """
    print(f"Hello, {name}!")

greet("Alice")  # Output: Hello, Alice!
```

### Default Arguments
```python
def power(base, exponent=2):
    return base ** exponent

print(power(3))      # 9 (uses default exponent=2)
print(power(3, 3))   # 27
```

### Docstrings
```python
def add(a, b):
    """
    Add two numbers together.
    
    Args:
        a (int or float): First number
        b (int or float): Second number
        
    Returns:
        int or float: The sum of a and b
    """
    return a + b
```

## Function Arguments

### Positional and Keyword Arguments
```python
def describe_pet(pet_name, animal_type='dog'):
    print(f"I have a {animal_type} named {pet_name}.")

# Positional arguments
describe_pet('Willie', 'hamster')

# Keyword arguments
describe_pet(pet_name='Willie', animal_type='hamster')

# Mixing positional and keyword
describe_pet('Willie', animal_type='hamster')
```

### Arbitrary Arguments
```python
# *args for variable number of arguments
def make_pizza(*toppings):
    print("Making a pizza with:")
    for topping in toppings:
        print(f"- {topping}")

make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')

# **kwargs for variable keyword arguments
def build_profile(first, last, **user_info):
    user_info['first_name'] = first
    user_info['last_name'] = last
    return user_info

user_profile = build_profile('albert', 'einstein',
                           location='princeton',
                           field='physics')
print(user_profile)
```

## Return Values

### Multiple Return Values
```python
def get_name():
    first = "John"
    last = "Doe"
    return first, last  # Returns a tuple

first_name, last_name = get_name()
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
```

### Nonlocal Variables
```python
def outer():
    x = "outer x"
    
    def inner():
        nonlocal x
        x = "inner x"
        print(f"Inner: {x}")
    
    inner()
    print(f"Outer: {x}")

outer()
```

## Modules and Packages

### Creating and Using Modules
```python
# File: mymodule.py
def greet(name):
    return f"Hello, {name}!"

PI = 3.14159

# File: main.py
import mymodule

print(mymodule.greet("Alice"))
print(mymodule.PI)

# Import specific functions
from mymodule import greet, PI

# Import with alias
import mymodule as mm

# Import all names (not recommended)
from mymodule import *
```

### Creating Packages
```
my_package/
├── __init__.py
├── module1.py
└── module2.py
```

```python
# my_package/__init__.py
from .module1 import function1
from .module2 import function2

__all__ = ['function1', 'function2']

# Using the package
from my_package import function1
import my_package
```

## The `__name__` Variable

```python
# File: mymodule.py
def main():
    print("This is the main function")

if __name__ == "__main__":
    print("Running as main program")
    main()
else:
    print("Imported as a module")
```

## Virtual Environments

### Creating a Virtual Environment
```bash
# Create a virtual environment
python -m venv myenv

# Activate (Unix/macOS)
source myenv/bin/activate

# Activate (Windows)
myenv\Scripts\activate

# Deactivate
deactivate
```

### Using requirements.txt
```bash
# Save dependencies
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt
```

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
