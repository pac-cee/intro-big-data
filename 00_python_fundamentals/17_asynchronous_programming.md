# Asynchronous Programming in Python

## Table of Contents
1. [Understanding Asyncio](#understanding-asyncio)
2. [Async/Await Syntax](#asyncawait-syntax)
3. [Concurrent Tasks](#concurrent-tasks)
4. [Async HTTP Requests](#async-http-requests)
5. [Async Database Access](#async-database-access)
6. [WebSockets](#websockets)
7. [Performance Considerations](#performance-considerations)

## Understanding Asyncio

### What is Asynchronous Programming?
- Non-blocking I/O operations
- Single-threaded concurrency
- Cooperative multitasking

### Event Loop
```python
import asyncio

async def main():
    print('Hello')
    await asyncio.sleep(1)
    print('World')

# Python 3.7+
asyncio.run(main())
```

## Async/Await Syntax

### Basic Coroutine
```python
async def fetch_data():
    print("Fetching data...")
    await asyncio.sleep(2)  # Simulate I/O operation
    print("Data fetched!")
    return {"data": 123}

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

## Concurrent Tasks

### Running Multiple Coroutines
```python
async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    # Run concurrently
    await asyncio.gather(count(), count(), count())

asyncio.run(main())
```

### Task Objects
```python
async def task(name, seconds):
    print(f"Task {name} started")
    await asyncio.sleep(seconds)
    print(f"Task {name} completed")

async def main():
    # Create tasks
    task1 = asyncio.create_task(task("A", 2))
    task2 = asyncio.create_task(task("B", 1))
    
    # Wait for both tasks to complete
    await task1
    await task2

asyncio.run(main())
```

## Async HTTP Requests

### Using aiohttp
```bash
pip install aiohttp
```

### Making Concurrent Requests
```python
import aiohttp
import asyncio

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        'https://api.github.com',
        'https://httpbin.org/get',
        'https://example.com'
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for url, content in zip(urls, results):
            print(f"{url} returned {len(content)} bytes")

asyncio.run(main())
```

## Async Database Access

### Using asyncpg (PostgreSQL)
```bash
pip install asyncpg
```

### Basic Database Operations
```python
import asyncpg
import asyncio

async def main():
    # Connect to database
    conn = await asyncpg.connect(
        user='user',
        password='password',
        database='database',
        host='127.0.0.1'
    )
    
    # Create table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    ''')
    
    # Insert data
    await conn.execute(
        'INSERT INTO users(name, email) VALUES($1, $2)',
        'John Doe', 'john@example.com'
    )
    
    # Query data
    rows = await conn.fetch('SELECT * FROM users')
    for row in rows:
        print(f"ID: {row['id']}, Name: {row['name']}")
    
    # Close connection
    await conn.close()

asyncio.run(main())
```

## WebSockets

### WebSocket Server with FastAPI
```bash
pip install websockets fastapi uvicorn
```

### WebSocket Implementation
```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Demo</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <input id="message" />
        <button onclick="sendMessage()">Send</button>
        <div id="messages"></div>
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            ws.onmessage = function(event) {
                const messages = document.getElementById('messages');
                const message = document.createElement('div');
                message.textContent = event.data;
                messages.appendChild(message);
            };
            function sendMessage() {
                const input = document.getElementById('message');
                ws.send(input.value);
                input.value = '';
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message: {data}")
```

## Performance Considerations

1. **CPU-bound vs I/O-bound**
   - Asyncio is great for I/O-bound operations
   - For CPU-bound tasks, consider `multiprocessing`

2. **Common Pitfalls**
   - Blocking the event loop
   - Not using `await` properly
   - Mixing sync and async code incorrectly

3. **Best Practices**
   - Use `async with` for resource management
   - Limit concurrency with semaphores
   - Handle exceptions properly
   - Use task groups for better control

## Next Steps
1. Explore more advanced asyncio patterns
2. Learn about async database ORMs (e.g., Tortoise ORM, GINO)
3. Study WebSocket implementations
4. Look into async task queues (e.g., Celery with asyncio)
