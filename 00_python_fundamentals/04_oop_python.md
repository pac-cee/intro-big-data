# Object-Oriented Programming in Python

## Table of Contents
1. [Introduction to OOP](#introduction-to-oop)
2. [Classes and Objects](#classes-and-objects)
3. [Inheritance](#inheritance)
4. [Encapsulation](#encapsulation)
5. [Polymorphism](#polymorphism)
6. [Magic Methods](#magic-methods)
7. [Class and Static Methods](#class-and-static-methods)
8. [Practice Exercises](#practice-exercises)

## Introduction to OOP

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of "objects", which can contain data and code that manipulates that data.

### Key Concepts
- **Class**: A blueprint for creating objects
- **Object**: An instance of a class
- **Attribute**: A variable that holds data about an object
- **Method**: A function defined in a class
- **Inheritance**: Creating a new class from an existing class
- **Encapsulation**: Restricting access to certain components
- **Polymorphism**: Using a unified interface for different data types

## Classes and Objects

### Creating a Simple Class
```python
class Dog:
    # Class attribute
    species = "Canis familiaris"
    
    # Initializer / Instance attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"
    
    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"

# Create instances of Dog
buddy = Dog("Buddy", 9)
miles = Dog("Miles", 4)

# Access attributes
print(buddy.name)  # Buddy
print(buddy.age)   # 9
print(buddy.species)  # Canis familiaris

# Call methods
print(buddy.description())  # Buddy is 9 years old
print(miles.speak("Woof Woof"))  # Miles says Woof Woof
```

### Class and Instance Variables
```python
class Dog:
    # Class variable
    species = "Canis familiaris"
    
    def __init__(self, name):
        # Instance variable
        self.name = name
        self.tricks = []  # creates a new empty list for each dog
    
    def add_trick(self, trick):
        self.tricks.append(trick)

# Create two dogs
buddy = Dog('Buddy')
buddy.add_trick('roll over')

miles = Dog('Miles')
miles.add_trick('play dead')

print(buddy.tricks)  # ['roll over']
print(miles.tricks)  # ['play dead']
print(Dog.species)   # Canis familiaris
```

## Inheritance

### Basic Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Create instances
dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!
```

### Method Resolution Order (MRO)
```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")
        super().show()

class C(A):
    def show(self):
        print("C")
        super().show()

class D(B, C):
    def show(self):
        print("D")
        super().show()

d = D()
d.show()
# Output:
# D
# B
# C
# A

print(D.mro())  # Shows the method resolution order
```

## Encapsulation

### Private Members
```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}")
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}")
        else:
            print("Insufficient funds")
    
    def get_balance(self):
        return self.__balance

# Create account
account = BankAccount("Alice", 1000)

# Can't access private attribute directly
# print(account.__balance)  # Error
print(account.get_balance())  # 1000

# Name mangling (not recommended, but possible)
print(account._BankAccount__balance)  # 1000
```

### Properties
```python
class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @full_name.setter
    def full_name(self, name):
        first, last = name.split()
        self.first_name = first
        self.last_name = last

person = Person("John", "Doe")
print(person.full_name)  # John Doe

person.full_name = "Jane Smith"
print(person.first_name)  # Jane
print(person.last_name)   # Smith
```

## Polymorphism

### Duck Typing
```python
class Duck:
    def quack(self):
        return "Quack!"

class Person:
    def quack(self):
        return "I'm quacking like a duck!"

def make_it_quack(duck):
    print(duck.quack())

# Both work because they implement quack()
make_it_quack(Duck())   # Quack!
make_it_quack(Person()) # I'm quacking like a duck!
```

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # Error: Can't instantiate abstract class
rect = Rectangle(4, 5)
print(rect.area())  # 20
```

## Magic Methods

### Common Magic Methods
```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        return f"{self.title} by {self.author}"
    
    def __len__(self):
        return self.pages
    
    def __add__(self, other):
        return Book(f"{self.title} & {other.title}", 
                   f"{self.author} and {other.author}", 
                   self.pages + other.pages)

# Create book instances
book1 = Book("Python Crash Course", "Eric Matthes", 544)
book2 = Book("Fluent Python", "Luciano Ramalho", 792)

# Using magic methods
print(book1)          # Python Crash Course by Eric Matthes
print(len(book1))     # 544
combined = book1 + book2
print(combined)       # Python Crash Course & Fluent Python by Eric Matthes and Luciano Ramalho
print(len(combined))  # 1336
```

## Class and Static Methods

### @classmethod
```python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients
    
    def __repr__(self):
        return f"Pizza({self.ingredients!r})"
    
    @classmethod
    def margherita(cls):
        return cls(["mozzarella", "tomatoes"])
    
    @classmethod
    def prosciutto(cls):
        return cls(["mozzarella", "tomatoes", "ham"])

# Create instances using class methods
p1 = Pizza.margherita()
p2 = Pizza.prosciutto()
print(p1)  # Pizza(['mozzarella', 'tomatoes'])
print(p2)  # Pizza(['mozzarella', 'tomatoes', 'ham'])
```

### @staticmethod
```python
class Math:
    @staticmethod
    def add(x, y):
        return x + y
    
    @staticmethod
    def multiply(x, y):
        return x * y

# Call static methods without creating an instance
print(Math.add(5, 3))      # 8
print(Math.multiply(5, 3)) # 15
```

## Practice Exercises

1. **Basic Class**
   Create a `BankAccount` class with methods to deposit, withdraw, and check balance.

2. **Inheritance**
   Create a `Vehicle` class with subclasses `Car` and `Motorcycle`. Each should have appropriate attributes and methods.

3. **Encapsulation**
   Create a `Temperature` class that stores the temperature in Celsius but can be set and retrieved in both Celsius and Fahrenheit.

4. **Polymorphism**
   Create a list of different shape objects (Circle, Square, etc.) and call an `area()` method on each one.

5. **Magic Methods**
   Create a `Vector` class that supports addition, subtraction, and string representation using magic methods.

6. **Class Methods**
   Create a `Student` class with a class method `from_birth_year` that creates a student given their birth year.

7. **Static Methods**
   Create a `StringUtils` class with static methods for common string operations.

8. **Property Decorators**
   Create a `Person` class with a `full_name` property that combines `first_name` and `last_name`.

9. **Advanced**
   Implement a custom context manager using a class with `__enter__` and `__exit__` methods.

10. **Project**
    Design a simple library management system with classes for `Book`, `Library`, and `Member`. Include methods for checking out and returning books.

---
Next: [File Handling in Python](./05_file_handling.md)
