# File Handling in Python

## Table of Contents
1. [Reading and Writing Files](#reading-and-writing-files)
2. [Working with File Paths](#working-with-file-paths)
3. [Working with JSON](#working-with-json)
4. [Working with CSV](#working-with-csv)
5. [Working with Excel Files](#working-with-excel-files)
6. [Working with Binary Files](#working-with-binary-files)
7. [Context Managers](#context-managers)
8. [Practice Exercises](#practice-exercises)

## Reading and Writing Files

### Opening and Closing Files
```python
# Basic file operations
file = open('example.txt', 'r')  # Open for reading (default)
content = file.read()
file.close()

# Better way using with statement (automatically closes file)
with open('example.txt', 'r') as file:
    content = file.read()
```

### File Modes
- `'r'` - Read (default)
- `'w'` - Write (truncates existing file)
- `'a'` - Append
- `'x'` - Exclusive creation (fails if file exists)
- `'b'` - Binary mode
- `'+'` - Updating (read/write)

### Reading Files
```python
# Read entire file
with open('example.txt', 'r') as file:
    content = file.read()

# Read line by line
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes trailing newline

# Read all lines into a list
with open('example.txt', 'r') as file:
    lines = file.readlines()
```

### Writing to Files
```python
# Write to a file (overwrites existing content)
with open('output.txt', 'w') as file:
    file.write('Hello, World!\n')
    file.write('This is a new line.')

# Append to a file
with open('output.txt', 'a') as file:
    file.write('\nThis is appended text.')
```

## Working with File Paths

### Using `os.path`
```python
import os

# Get current working directory
current_dir = os.getcwd()

# Join paths in a platform-independent way
file_path = os.path.join('folder', 'subfolder', 'file.txt')

# Get absolute path
abs_path = os.path.abspath('file.txt')

# Check if file exists
if os.path.exists('file.txt'):
    print("File exists")
```

### Using `pathlib` (Python 3.4+)
```python
from pathlib import Path

# Create a Path object
file_path = Path('folder') / 'subfolder' / 'file.txt'

# Get file name
print(file_path.name)  # file.txt

# Get parent directory
print(file_path.parent)  # folder/subfolder

# Check if file exists
if file_path.exists():
    print("File exists")

# Read file
content = file_path.read_text()

# Write to file
file_path.write_text('New content')
```

## Working with JSON

### Reading and Writing JSON
```python
import json

# Python object to JSON string
data = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

# Write JSON to file
with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)

# Read JSON from file
with open('data.json', 'r') as file:
    loaded_data = json.load(file)
    print(loaded_data['name'])  # John

# Convert JSON string to Python object
json_str = '{"name": "John", "age": 30}'
data = json.loads(json_str)

# Convert Python object to JSON string
json_str = json.dumps(data, indent=2)
```

## Working with CSV

### Reading CSV Files
```python
import csv

# Reading CSV into a list of dictionaries
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['name'], row['email'])

# Reading CSV into a list of lists
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        print(row[0], row[1])  # Access columns by index
```

### Writing CSV Files
```python
# Writing a list of dictionaries to CSV
import csv

data = [
    {'name': 'John', 'age': 30, 'city': 'New York'},
    {'name': 'Jane', 'age': 25, 'city': 'Chicago'}
]

with open('output.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['name', 'age', 'city'])
    writer.writeheader()
    writer.writerows(data)

# Writing a list of lists to CSV
rows = [
    ['Name', 'Age', 'City'],
    ['John', '30', 'New York'],
    ['Jane', '25', 'Chicago']
]

with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
```

## Working with Excel Files

### Using `openpyxl`
```python
from openpyxl import Workbook, load_workbook

# Create a new workbook
wb = Workbook()
ws = wb.active
ws.title = "Sheet1"

# Write data
ws['A1'] = "Name"
ws['B1'] = "Age"
ws.append(["John", 30])
ws.append(["Jane", 25])

# Save workbook
wb.save('example.xlsx')

# Load existing workbook
wb = load_workbook('example.xlsx')
ws = wb.active

# Read data
for row in ws.iter_rows(values_only=True):
    print(row)
```

## Working with Binary Files

### Reading and Writing Binary Files
```python
# Copying a binary file (e.g., an image)
with open('source.jpg', 'rb') as source_file:
    with open('copy.jpg', 'wb') as dest_file:
        chunk_size = 1024
        while True:
            chunk = source_file.read(chunk_size)
            if not chunk:
                break
            dest_file.write(chunk)
```

### Working with `struct`
```python
import struct

# Pack values into binary data
packed_data = struct.pack('iif', 10, 20, 3.14)  # int, int, float

# Write binary data to file
with open('binary.bin', 'wb') as file:
    file.write(packed_data)

# Read and unpack binary data
with open('binary.bin', 'rb') as file:
    data = file.read()
    unpacked_data = struct.unpack('iif', data)
    print(unpacked_data)  # (10, 20, 3.140000104904175)
```

## Context Managers

### Using `with` Statement
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

# Using the custom context manager
with FileManager('example.txt', 'w') as file:
    file.write('Hello, World!')
```

### Using `contextlib`
```python
from contextlib import contextmanager

@contextmanager
def open_file(filename, mode):
    try:
        file = open(filename, mode)
        yield file
    finally:
        file.close()

# Using the context manager
with open_file('example.txt', 'w') as file:
    file.write('Hello, World!')
```

## Practice Exercises

1. **File Copy**
   Write a function that copies a file from one location to another.

2. **Word Count**
   Write a program that counts the number of words in a text file.

3. **CSV to JSON**
   Convert a CSV file to a JSON file.

4. **Log Parser**
   Parse a log file and extract all error messages.

5. **File Search**
   Search for a specific string in all files within a directory.

6. **Excel Report**
   Read data from a CSV file and create an Excel report with charts.

7. **Binary File**
   Create a simple binary file format to store student records.

8. **Context Manager**
   Create a context manager that times how long a block of code takes to execute.

9. **File Organizer**
   Organize files in a directory by their extensions.

10. **Config Parser**
    Create a configuration file parser that can read and write INI-style config files.

---
Next: [Error Handling in Python](./06_error_handling.md)
