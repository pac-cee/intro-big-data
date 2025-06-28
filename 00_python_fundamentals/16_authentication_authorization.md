# Authentication & Authorization in Python

## Table of Contents
1. [Authentication vs Authorization](#authentication-vs-authorization)
2. [Password Hashing](#password-hashing)
3. [JWT Authentication](#jwt-authentication)
4. [OAuth 2.0 & OpenID Connect](#oauth-20--openid-connect)
5. [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
6. [Security Best Practices](#security-best-practices)

## Authentication vs Authorization

### Authentication
Verifying who someone is (e.g., username/password, social login, API keys)

### Authorization
Determining what resources a user can access and what they can do with them

## Password Hashing

### Using bcrypt
```bash
pip install bcrypt
```

### Hashing and Verifying Passwords
```python
import bcrypt

# Hash a password
password = b"mysecurepassword"
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)

# Verify password
if bcrypt.checkpw(password, hashed):
    print("Password matches!")
```

## JWT Authentication

### Installation
```bash
pip install pyjwt python-dotenv
```

### JWT Implementation
```python
import jwt
import datetime
from functools import wraps
from flask import request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')

def generate_token(user_id):
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
        'iat': datetime.datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
            
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
            
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = data['sub']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated
```

## OAuth 2.0 & OpenID Connect

### Using Authlib
```bash
pip install authlib
```

### OAuth2 Provider Example
```python
from authlib.integrations.flask_oauth2 import AuthorizationServer, ResourceProtector
from authlib.oauth2.rfc6749 import grants

# Configuration
server = AuthorizationServer()
require_oauth = ResourceProtector()

def config_oauth(app):
    server.init_app(app)
    # Add grant types
    # server.register_grant()
    # Protect views
    # require_oauth.register_token_validator()
```

## Role-Based Access Control (RBAC)

### Basic Implementation
```python
from functools import wraps

# User roles
ROLES = {
    'admin': 3,
    'editor': 2,
    'user': 1
}

def requires_roles(*roles):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get user's role from JWT or session
            user_role = get_current_user_role()
            
            if user_role not in [ROLES[role] for role in roles]:
                return jsonify({'message': 'Insufficient permissions'}), 403
                
            return f(*args, **kwargs)
        return wrapped
    return wrapper

# Usage
@app.route('/admin')
@token_required
@requires_roles('admin')
def admin_dashboard():
    return jsonify({'message': 'Welcome Admin!'})
```

## Security Best Practices

1. **Password Security**
   - Use strong hashing algorithms (bcrypt, Argon2)
   - Enforce password policies
   - Implement account lockout after failed attempts

2. **Session Security**
   - Use secure and httpOnly cookies
   - Implement proper session timeout
   - Regenerate session IDs after login

3. **JWT Security**
   - Use strong secret keys
   - Set appropriate expiration times
   - Implement token refresh mechanism
   - Store tokens securely (not in localStorage)

4. **General Security**
   - Use HTTPS
   - Implement CORS properly
   - Sanitize all user inputs
   - Use security headers (CSP, XSS-Protection)
   - Regular security audits

## Next Steps
1. Implement multi-factor authentication (MFA)
2. Set up rate limiting
3. Add security headers using Flask-Talisman or similar
4. Implement password reset flow
5. Set up security monitoring and logging

## Common Vulnerabilities
- SQL Injection (use ORM or parameterized queries)
- Cross-Site Scripting (XSS)
- Cross-Site Request Forgery (CSRF)
- Insecure Direct Object References (IDOR)
- Security Misconfiguration
