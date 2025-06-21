# Working with the Python Standard Library

## Table of Contents
1. [Introduction to the Standard Library](#introduction-to-the-standard-library)
2. [File and Directory Access](#file-and-directory-access)
3. [Data Types and Collections](#data-types-and-collections)
4. [File Formats](#file-formats)
5. [Data Compression and Archiving](#data-compression-and-archiving)
6. [Concurrent Execution](#concurrent-execution)
7. [Networking and Internet](#networking-and-internet)
8. [Date and Time](#date-and-time)
9. [Mathematics](#mathematics)
10. [Performance Measurement](#performance-measurement)
11. [Practice Exercises](#practice-exercises)

## Introduction to the Standard Library

The Python Standard Library is a collection of modules that come with Python and provide a wide range of functionality. This document covers some of the most useful modules for data processing and general programming.

### Importing Modules
```python
# Import entire module
import os

# Import specific items
from collections import defaultdict

# Import with alias
import pandas as pd

# Import all names from a module (not recommended)
from math import *
```

## File and Directory Access

### `os` and `os.path`
```python
import os

# File operations
os.rename('old.txt', 'new.txt')
os.remove('file.txt')
os.mkdir('new_dir')
os.makedirs('path/to/dir', exist_ok=True)

# Path operations
current_dir = os.getcwd()
file_path = os.path.join('folder', 'file.txt')
abs_path = os.path.abspath('file.txt')
base_name = os.path.basename('/path/to/file.txt')  # 'file.txt'
dir_name = os.path.dirname('/path/to/file.txt')    # '/path/to'

# Environment variables
home_dir = os.environ.get('HOME')
os.environ['MY_VAR'] = 'value'
```

### `pathlib` (Python 3.4+)
```python
from pathlib import Path

# Create Path object
file_path = Path('folder') / 'file.txt'

# File operations
file_path.touch()  # Create empty file
file_path.write_text('Hello, World!')
content = file_path.read_text()

# Directory operations
new_dir = Path('new_dir')
new_dir.mkdir(exist_ok=True)

# Get all .txt files in directory
txt_files = list(Path('.').glob('*.txt'))
```

### `glob`
```python
import glob

# Find all .txt files in current directory
txt_files = glob.glob('*.txt')

# Recursive search
all_py_files = glob.glob('**/*.py', recursive=True)
```

### `shutil`
```python
import shutil

# Copy files
shutil.copy('source.txt', 'destination.txt')
shutil.copytree('src_dir', 'dst_dir')  # Copy directory

# Move/rename
shutil.move('old.txt', 'new.txt')

# Remove directory tree
shutil.rmtree('directory')
```

## Data Types and Collections

### `collections`
```python
from collections import defaultdict, Counter, namedtuple, deque, OrderedDict

# Default dictionary
d = defaultdict(int)
d['key'] += 1  # No KeyError

# Counter
words = ['apple', 'banana', 'apple', 'orange']
word_counts = Counter(words)  # {'apple': 2, 'banana': 1, 'orange': 1}

# Named tuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20

# Deque (double-ended queue)
d = deque([1, 2, 3])
d.append(4)      # Add to end
d.appendleft(0)  # Add to beginning
```

### `itertools`
```python
import itertools

# Infinite iterators
count = itertools.count(1, 2)  # 1, 3, 5, 7, ...
cycle = itertools.cycle('ABC')  # A, B, C, A, B, C, ...
repeat = itertools.repeat(10, 3)  # 10, 10, 10

# Combinatoric iterators
combinations = list(itertools.combinations('ABCD', 2))  # AB, AC, AD, BC, BD, CD
permutations = list(itertools.permutations('ABC', 2))   # AB, AC, BA, BC, CA, CB

# Grouping
for key, group in itertools.groupby('AAAABBBCCDAABBB'):
    print(key, list(group))
# A ['A', 'A', 'A', 'A']
# B ['B', 'B', 'B']
# C ['C', 'C']
# D ['D']
# A ['A', 'A']
# B ['B', 'B', 'B']
```

### `functools`
```python
import functools

# Partial function
add_five = functools.partial(lambda x, y: x + y, 5)
print(add_five(3))  # 8

# Reduce
product = functools.reduce(lambda x, y: x * y, [1, 2, 3, 4])  # 24

# LRU Cache
@functools.lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## File Formats

### `json`
```python
import json

# Python object to JSON string
data = {'name': 'John', 'age': 30}
json_str = json.dumps(data, indent=2)

# JSON string to Python object
parsed = json.loads('{"name": "John", "age": 30}')

# Read from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Write to file
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### `csv`
```python
import csv

# Reading CSV
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['name'], row['email'])

# Writing CSV
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Email'])
    writer.writerow(['John', 'john@example.com'])
```

### `configparser`
```python
import configparser

# Create config
config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45', 'Compression': 'yes'}
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'

# Write to file
with open('config.ini', 'w') as f:
    config.write(f)

# Read from file
config.read('config.ini')
print(config['DEFAULT']['Compression'])  # 'yes'
```

## Data Compression and Archiving

### `gzip`
```python
import gzip

# Compress file
with open('file.txt', 'rb') as f_in:
    with gzip.open('file.txt.gz', 'wb') as f_out:
        f_out.writelines(f_in)

# Decompress file
with gzip.open('file.txt.gz', 'rb') as f:
    content = f.read()
```

### `zipfile`
```python
import zipfile

# Create ZIP archive
with zipfile.ZipFile('archive.zip', 'w') as zipf:
    zipf.write('file1.txt')
    zipf.write('file2.txt')

# Extract ZIP archive
with zipfile.ZipFile('archive.zip', 'r') as zipf:
    zipf.extractall('extracted')
```

## Concurrent Execution

### `threading`
```python
import threading
import time

def worker(num):
    print(f'Worker {num} starting')
    time.sleep(2)
    print(f'Worker {num} done')

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
```

### `multiprocessing`
```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(square, range(10))
    print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### `concurrent.futures`
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

URLS = ['http://example.com', 'http://example.org', 'http://example.net']

def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
            print(f'{url} page is {len(data)} bytes')
        except Exception as e:
            print(f'{url} generated an exception: {e}')
```

## Networking and Internet

### `urllib`
```python
from urllib.request import urlopen
from urllib.parse import urlencode

# Simple GET request
with urlopen('http://example.com') as response:
    content = response.read()

# POST request
from urllib.request import Request, urlopen

data = urlencode({'key1': 'value1', 'key2': 'value2'}).encode()
req = Request('http://example.com/post', data=data, method='POST')
with urlopen(req) as response:
    content = response.read()
```

### `http.server`
```python
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Simple HTTP server
PORT = 8000
server_address = ('', PORT)
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
print(f'Serving on port {PORT}')
# httpd.serve_forever()
```

### `socketserver`
```python
import socketserver

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.data = self.request.recv(1024).strip()
        print(f"{self.client_address[0]} wrote:")
        print(self.data)
        self.request.sendall(self.data.upper())

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # server.serve_forever()
        pass
```

## Date and Time

### `datetime`
```python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
today = date.today()

# Formatting
formatted = now.strftime('%Y-%m-%d %H:%M:%S')  # '2023-04-15 14:30:00'

# Parsing
dt = datetime.strptime('2023-04-15', '%Y-%m-%d')

# Time delta
tomorrow = today + timedelta(days=1)
last_week = today - timedelta(weeks=1)
```

### `time`
```python
import time

# Current time in seconds since epoch
current_time = time.time()

# Sleep for 2.5 seconds
time.sleep(2.5)

# Measure execution time
start = time.perf_counter()
# Code to measure
end = time.perf_counter()
elapsed = end - start
```

## Mathematics

### `math`
```python
import math

# Constants
print(math.pi)     # 3.141592...
print(math.e)      # 2.718281...
print(math.inf)    # inf
print(math.nan)    # nan

# Basic functions
print(math.sqrt(16))      # 4.0
print(math.pow(2, 3))     # 8.0
print(math.factorial(5))  # 120
print(math.gcd(54, 24))   # 6

# Trigonometry
print(math.sin(math.pi/2))  # 1.0
print(math.degrees(math.pi)) # 180.0
```

### `random`
```python
import random

# Random float in [0.0, 1.0)
print(random.random())

# Random float in range [a, b]
print(random.uniform(1, 10))

# Random integer in range [a, b]
print(random.randint(1, 6))  # Simulate a die roll

# Random element from sequence
items = ['red', 'green', 'blue']
print(random.choice(items))

# Shuffle list in place
random.shuffle(items)
print(items)

# Sample without replacement
print(random.sample(range(100), 10))  # 10 unique numbers
```

### `statistics`
```python
import statistics

data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]

print(statistics.mean(data))     # 1.6071428571428572
print(statistics.median(data))   # 1.25
print(statistics.mode(data))     # 1.25
print(statistics.stdev(data))    # 1.1151877031898467
```

## Performance Measurement

### `timeit`
```python
import timeit

# Time a single statement
t = timeit.timeit('"".join(str(n) for n in range(100))', number=10000)
print(t)

# Compare two approaches
setup = 'data = ["a"] * 1000'
stmt1 = '"".join(str(x) for x in data)'
stmt2 = '"".join(map(str, data))'

t1 = timeit.timeit(stmt1, setup, number=1000)
t2 = timeit.timeit(stmt2, setup, number=1000)
print(f'Generator: {t1:.3f} seconds')
print(f'Map: {t2:.3f} seconds')
```

### `cProfile`
```python
import cProfile

def slow_function():
    total = 0
    for i in range(10000):
        for j in range(10000):
            total += i * j
    return total

# Profile the function
cProfile.run('slow_function()')
```

## Practice Exercises

1. **File Search**
   Write a function that searches for files with a specific extension in a directory and its subdirectories.

2. **CSV to JSON Converter**
   Create a script that converts a CSV file to JSON format.

3. **URL Status Checker**
   Write a program that checks the status of multiple URLs concurrently.

4. **Log Parser**
   Create a script that parses a log file and extracts error messages with timestamps.

5. **Directory Synchronization**
   Write a script that synchronizes two directories (like rsync).

6. **Simple Web Server**
   Create a simple HTTP server that serves files from a directory and handles basic requests.

7. **Data Analysis**
   Use the `statistics` module to analyze a dataset from a CSV file.

8. **Password Generator**
   Create a secure password generator using the `secrets` module.

9. **File Compression**
   Write a script that compresses all files in a directory and saves them with timestamps.

10. **Process Monitor**
    Create a simple process monitor using the `psutil` library (needs to be installed).

---
Next: [Virtual Environments and Package Management](./11_virtual_environments.md)
