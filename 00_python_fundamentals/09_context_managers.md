# Context Managers in Python

## Table of Contents
1. [Introduction to Context Managers](#introduction-to-context-managers)
2. [The `with` Statement](#the-with-statement)
3. [Creating Context Managers with Classes](#creating-context-managers-with-classes)
4. [Creating Context Managers with `contextlib`](#creating-context-managers-with-contextlib)
5. [Nested Context Managers](#nested-context-managers)
6. [Asynchronous Context Managers](#asynchronous-context-managers)
7. [Built-in Context Managers](#built-in-context-managers)
8. [Real-world Use Cases](#real-world-use-cases)
9. [Best Practices](#best-practices)
10. [Practice Exercises](#practice-exercises)

## Introduction to Context Managers

Context managers are a way to manage resources and ensure they are properly initialized and cleaned up. They're commonly used for:
- File operations
- Database connections
- Locks and semaphores
- Network connections
- Temporary changes to system state

### Why Use Context Managers?
- **Resource Management**: Ensure resources are properly released
- **Readability**: Make code more readable by grouping setup and teardown logic
- **Error Handling**: Handle exceptions and cleanup automatically
- **Reduced Boilerplate**: Eliminate repetitive try-finally blocks

## The `with` Statement

The `with` statement is used to wrap the execution of a block of code within methods defined by a context manager.

### Basic Usage
```python
# Without context manager
file = open('example.txt', 'r')
try:
    content = file.read()
    # Process the file
finally:
    file.close()

# With context manager
with open('example.txt', 'r') as file:
    content = file.read()
    # Process the file
# File is automatically closed when the block is exited
```

## Creating Context Managers with Classes

A context manager class must implement `__enter__` and `__exit__` methods.

### Basic Implementation
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Handle exceptions if needed
        if exc_type is not None:
            print(f"An exception occurred: {exc_val}")
        # Return True to suppress the exception, False to propagate it
        return False

# Usage
with FileManager('example.txt', 'w') as f:
    f.write('Hello, World!')
```

### The `__exit__` Method Parameters
- `exc_type`: The exception type (e.g., `ValueError`)
- `exc_val`: The exception instance
- `exc_tb`: The traceback object

## Creating Context Managers with `contextlib`

The `contextlib` module provides utilities for creating context managers.

### Using `@contextmanager` Decorator
```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# Usage
with file_manager('example.txt', 'w') as f:
    f.write('Hello, World!')
```

### Handling Exceptions
```python
@contextmanager
def handle_errors():
    try:
        print('Entering the context')
        yield
    except Exception as e:
        print(f'Error: {e}')
        # Re-raise the exception
        raise
    finally:
        print('Exiting the context')

# Usage
with handle_errors():
    print('Inside the context')
    # raise ValueError('Something went wrong')
```

## Nested Context Managers

You can nest multiple context managers.

### Manual Nesting
```python
with open('file1.txt', 'r') as f1:
    with open('file2.txt', 'w') as f2:
        content = f1.read()
        f2.write(content.upper())
```

### Using `contextlib.ExitStack`
```python
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname)) for fname in filenames]
        # All files will be closed when the block is exited
        # Process files here
        pass
```

## Asynchronous Context Managers

For working with async/await syntax.

### Using `async with`
```python
class AsyncResource:
    async def __aenter__(self):
        print('Opening resource')
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print('Closing resource')
        return False

# Usage
async def main():
    async with AsyncResource() as resource:
        print('Inside the async context')

# Run the async function
import asyncio
asyncio.run(main())
```

## Built-in Context Managers

### `contextlib` Utilities

#### `redirect_stdout` and `redirect_stderr`
```python
from contextlib import redirect_stdout
import io

# Redirect stdout to a file
with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        print('This will be written to output.txt')

# Capture stdout
f = io.StringIO()
with redirect_stdout(f):
    print('Hello, World!')
print(f'Captured: {f.getvalue()}')
```

#### `suppress`
```python
from contextlib import suppress

# Suppress specific exceptions
with suppress(FileNotFoundError):
    os.remove('nonexistent_file.txt')
# No FileNotFoundError will be raised
```

#### `closing`
```python
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen('http://example.com')) as page:
    content = page.read()
# Connection is automatically closed
```

## Real-world Use Cases

### Database Connections
```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Usage
with db_connection('mydb.sqlite') as cursor:
    cursor.execute('SELECT * FROM users')
    results = cursor.fetchall()
```

### Timing Blocks of Code
```python
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timer():
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print(f'Elapsed: {end - start:.3f} seconds')

# Usage
with timer():
    # Code to time
    sum(range(1000000))
```

### Temporary Directory
```python
import tempfile
import shutil

@contextmanager
def temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

# Usage
with temp_directory() as temp_dir:
    print(f'Working in temporary directory: {temp_dir}')
    # Work with files in temp_dir
# Directory is automatically deleted
```

## Best Practices

1. **Always Clean Up**: Ensure resources are properly released in the `__exit__` method or after the `yield` in a generator-based context manager.

2. **Handle Exceptions**: Properly handle exceptions in the `__exit__` method or after the `yield`.

3. **Use `contextlib` Utilities**: Leverage the `contextlib` module for simpler context manager creation.

4. **Document Resource Usage**: Clearly document what resources are managed by your context manager.

5. **Consider Reusability**: Make context managers reusable when possible.

6. **Use `@contextmanager` for Simple Cases**: For simple cases, prefer the `@contextmanager` decorator over implementing the full protocol.

7. **Be Careful with State**: Be mindful of the state that persists across the `yield` in generator-based context managers.

## Practice Exercises

1. **File Writer**
   Create a context manager that opens a file for writing and automatically adds timestamps to each line written.

2. **Database Transaction**
   Implement a context manager for database transactions that automatically commits on success and rolls back on failure.

3. **Performance Timer**
   Create a context manager that measures and logs the execution time of the code block.

4. **Temporary Environment Variables**
   Write a context manager that temporarily sets environment variables and restores them afterward.

5. **Lock Manager**
   Implement a context manager for thread synchronization using `threading.Lock`.

6. **Directory Changer**
   Create a context manager that changes the working directory and restores it afterward.

7. **HTTP Session**
   Implement a context manager for making HTTP requests that ensures the session is properly closed.

8. **Logging Context**
   Create a context manager that adds contextual information to log messages within its block.

9. **Resource Pool**
   Implement a context manager that manages a pool of resources (e.g., database connections).

10. **Nested Context**
    Create a context manager that can be nested and tracks the nesting level.

---
Next: [Working with the Python Standard Library](./10_standard_library.md)
