# Building RESTful APIs with Python

## Table of Contents
1. [REST Fundamentals](#rest-fundamentals)
2. [Flask-RESTful](#flask-restful)
3. [FastAPI](#fastapi)
4. [API Authentication](#api-authentication)
5. [API Documentation](#api-documentation)
6. [Best Practices](#best-practices)
7. [Testing APIs](#testing-apis)

## REST Fundamentals

### What is REST?
REST (Representational State Transfer) is an architectural style for designing networked applications.

### Key Principles:
- **Stateless**: Each request contains all necessary information
- **Client-Server**: Separation of concerns
- **Cacheable**: Responses define themselves as cacheable or not
- **Uniform Interface**: Consistent resource identification and manipulation
- **Layered System**: Client can't tell if connected to end server or intermediary

### HTTP Methods
- `GET`: Retrieve a resource
- `POST`: Create a new resource
- `PUT`: Update a resource
- `DELETE`: Remove a resource
- `PATCH`: Partially update a resource

## Flask-RESTful

### Setup
```bash
pip install flask-restful
```

### Basic API
```python
from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'message': 'Hello, World!'}

api.add_resource(HelloWorld, '/hello')

if __name__ == '__main__':
    app.run(debug=True)
```

### Resource with Parameters
```python
class User(Resource):
    def get(self, user_id):
        return {'user': user_id, 'name': 'John Doe'}

api.add_resource(User, '/user/<int:user_id>')
```

## FastAPI

### Setup
```bash
pip install fastapi uvicorn
```

### Basic API
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### Run with:
```bash
uvicorn main:app --reload
```

## API Authentication

### JWT Authentication Example (Flask)
```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app.config['JWT_SECRET_KEY'] = 'super-secret'
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    # Validate credentials
    if username != 'admin' or password != 'password':
        return {'message': 'Bad credentials'}, 401
        
    access_token = create_access_token(identity=username)
    return {'access_token': access_token}

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    return {'message': 'Protected endpoint'}
```

## API Documentation

### OpenAPI/Swagger with FastAPI
FastAPI automatically generates interactive API documentation:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

### Adding Descriptions
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Best Practices

1. **Versioning**: Use URL versioning (`/api/v1/resource`)
2. **Status Codes**: Use appropriate HTTP status codes
3. **Error Handling**: Consistent error responses
4. **Pagination**: For large datasets
5. **Rate Limiting**: Prevent abuse
6. **CORS**: Enable if needed
7. **HTTPS**: Always use in production

## Testing APIs

### Using pytest
```python
# test_api.py
def test_read_item():
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "q": None}
```

### Running Tests
```bash
pytest test_api.py -v
```

## Next Steps
1. Implement API versioning
2. Add rate limiting
3. Set up API monitoring
4. Implement API caching
5. Explore GraphQL as an alternative
