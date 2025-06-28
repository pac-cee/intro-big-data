# Testing in Python

## Table of Contents
1. [Testing Fundamentals](#testing-fundamentals)
2. [unittest Framework](#unittest-framework)
3. [pytest Framework](#pytest-framework)
4. [Testing Web Applications](#testing-web-applications)
5. [Testing Async Code](#testing-async-code)
6. [Test Coverage](#test-coverage)
7. [Mocking and Patching](#mocking-and-patching)
8. [Best Practices](#best-practices)

## Testing Fundamentals

### Why Test?
- Catch bugs early
- Ensure code works as expected
- Enable safe refactoring
- Document behavior
- Improve code design

### Types of Tests
- **Unit Tests**: Test individual components
- **Integration Tests**: Test interactions between components
- **Functional Tests**: Test complete features
- **End-to-End Tests**: Test the entire application

## unittest Framework

### Basic Test Case
```python
import unittest

def add(a, b):
    return a + b

class TestAddFunction(unittest.TestCase):
    def test_add_integers(self):
        self.assertEqual(add(1, 2), 3)
    
    def test_add_strings(self):
        self.assertEqual(add('hello', ' world'), 'hello world')
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -2), -3)

if __name__ == '__main__':
    unittest.main()
```

### Common Assertions
```python
self.assertEqual(a, b)        # a == b
self.assertNotEqual(a, b)     # a != b
self.assertTrue(x)            # bool(x) is True
self.assertFalse(x)           # bool(x) is False
self.assertIs(a, b)           # a is b
self.assertIsNone(x)          # x is None
self.assertIn(a, b)           # a in b
self.assertRaises(Error, func, *args, **kwargs)
```

## pytest Framework

### Installation
```bash
pip install pytest
```

### Basic Test with pytest
```python
# test_sample.py
def add(a, b):
    return a + b

def test_add_integers():
    assert add(1, 2) == 3

def test_add_strings():
    assert add('hello', ' world') == 'hello world'

def test_add_negative_numbers():
    assert add(-1, -2) == -3
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test_sample.py

# Run specific test function
pytest test_sample.py::test_add_integers

# Show output from passing tests
pytest -v

# Stop after first failure
pytest -x
```

## Testing Web Applications

### Testing Flask Applications
```python
# test_app.py
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        with app.app_context():
            pass
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data

def test_login(client):
    response = client.post('/login', data={
        'username': 'test',
        'password': 'password'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b'Welcome, test' in response.data
```

## Testing Async Code

### Using pytest-asyncio
```bash
pip install pytest-asyncio
```

### Async Test Example
```python
import pytest
import asyncio

async def async_function():
    await asyncio.sleep(0.1)
    return 42

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == 42
```

## Test Coverage

### Using pytest-cov
```bash
pip install pytest-cov

# Run tests with coverage
pytest --cov=myproject tests/

# Generate HTML report
pytest --cov=myproject --cov-report=html tests/
```

## Mocking and Patching

### Using unittest.mock
```python
from unittest.mock import patch, MagicMock
import requests

def get_data():
    response = requests.get('https://api.example.com/data')
    return response.json()

def test_get_data():
    # Create a mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {'key': 'value'}
    
    # Patch requests.get to return our mock
    with patch('requests.get', return_value=mock_response):
        result = get_data()
        
    assert result == {'key': 'value'}
    requests.get.assert_called_once_with('https://api.example.com/data')
```

## Best Practices

1. **Test Naming**
   - Name tests clearly (test_what_we_are_testing)
   - Follow the Arrange-Act-Assert pattern

2. **Test Isolation**
   - Each test should be independent
   - Use setup and teardown methods
   - Consider using fixtures

3. **Test Organization**
   - Keep tests close to the code they test
   - Use a `tests/` directory
   - Group related tests in classes

4. **Test Data**
   - Use factories or fixtures for test data
   - Consider using Faker for realistic test data
   - Keep tests deterministic

5. **Continuous Integration**
   - Run tests on every push
   - Enforce test coverage requirements
   - Test in multiple environments

## Next Steps
1. Learn about property-based testing with Hypothesis
2. Explore contract testing
3. Study performance testing
4. Learn about test-driven development (TDD)
