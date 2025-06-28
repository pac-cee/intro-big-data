# Web Development with Flask

## Table of Contents
1. [Introduction to Flask](#introduction-to-flask)
2. [Setting Up Flask](#setting-up-flask)
3. [Basic Routing](#basic-routing)
4. [Templates with Jinja2](#templates-with-jinja2)
5. [Handling Forms](#handling-forms)
6. [Database Integration](#database-integration)
7. [RESTful API Development](#restful-api-development)
8. [Authentication](#authentication)
9. [Deployment](#deployment)

## Introduction to Flask

Flask is a lightweight WSGI web application framework. It's designed to make getting started quick and easy, with the ability to scale up to complex applications.

### Key Features:
- Development server and debugger
- Integrated support for unit testing
- RESTful request dispatching
- Jinja2 templating
- Support for secure cookies (client side sessions)
- 100% WSGI 1.0 compliant
- Unicode-based
- Extensive documentation

## Setting Up Flask

### Installation
```bash
pip install flask
```

### Minimal Application
Create a file named `app.py`:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

Run the application:
```bash
python app.py
```

## Basic Routing

### Route Parameters
```python
@app.route('/user/<username>')
def show_user_profile(username):
    return f'User {username}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post {post_id}'
```

### HTTP Methods
```python
from flask import request

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return 'Processing login'
    else:
        return 'Show login form'
```

## Templates with Jinja2

### Basic Template Rendering
Create a `templates` folder and add `index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>My App</title>
</head>
<body>
    <h1>Hello, {{ name }}!</h1>
</body>
</html>
```

Render the template:
```python
from flask import render_template

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('index.html', name=name)
```

## Handling Forms

### Simple Form Handling
```python
from flask import request, redirect, url_for

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    # Process the form data
    return redirect(url_for('success'))
```

## Database Integration

### SQLAlchemy Setup
```bash
pip install flask-sqlalchemy
```

### Database Models
```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"
```

## RESTful API Development

### Basic API Endpoint
```python
from flask import jsonify

@app.route('/api/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{'username': user.username, 'email': user.email} for user in users])
```

## Authentication

### Basic Authentication
```python
from werkzeug.security import generate_password_hash, check_password_hash

# Hashing password
hashed_password = generate_password_hash('mypassword')

# Verifying password
check_password_hash(hashed_password, 'mypassword')  # Returns True if match
```

## Deployment

### Gunicorn Setup
```bash
pip install gunicorn
```

### Running with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Next Steps
1. Learn about Flask Blueprints for better project organization
2. Explore Flask extensions like Flask-Login, Flask-WTF, Flask-Mail
3. Implement JWT authentication for APIs
4. Set up database migrations with Flask-Migrate
