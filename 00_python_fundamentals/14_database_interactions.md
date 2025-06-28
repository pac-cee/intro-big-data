# Database Interactions in Python

## Table of Contents
1. [SQL with SQLite](#sql-with-sqlite)
2. [SQLAlchemy ORM](#sqlalchemy-orm)
3. [MongoDB with PyMongo](#mongodb-with-pymongo)
4. [Database Best Practices](#database-best-practices)

## SQL with SQLite

### Basic SQLite Operations
```python
import sqlite3

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, name TEXT, email TEXT)''')

# Insert data
c.execute("INSERT INTO users (name, email) VALUES (?, ?)",
          ('John Doe', 'john@example.com'))

# Commit changes
conn.commit()

# Query data
for row in c.execute('SELECT * FROM users'):
    print(row)

# Close connection
conn.close()
```

## SQLAlchemy ORM

### Setup
```bash
pip install sqlalchemy
```

### Basic Usage
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create engine
engine = create_engine('sqlite:///example.db', echo=True)
Base = declarative_base()

# Define model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)

# Create tables
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Add new user
new_user = User(name='Jane Doe', email='jane@example.com')
session.add(new_user)
session.commit()

# Query users
users = session.query(User).all()
for user in users:
    print(user.name, user.email)
```

## MongoDB with PyMongo

### Setup
```bash
pip install pymongo
```

### Basic Operations
```python
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
collection = db['users']

# Insert document
user = {
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
}
user_id = collection.insert_one(user).inserted_id

# Find documents
for user in collection.find({'age': {'$gt': 20}}):
    print(user)
```

## Database Best Practices

### SQL Best Practices
- Use parameterized queries to prevent SQL injection
- Create indexes for frequently queried columns
- Use transactions for multiple related operations
- Close connections properly
- Use connection pooling in production

### NoSQL Best Practices
- Design your schema based on query patterns
- Use appropriate indexes
- Consider document size and nesting depth
- Implement proper error handling
- Use connection pooling

### General Best Practices
- Use environment variables for database credentials
- Implement proper error handling
- Use database migrations for schema changes
- Backup your database regularly
- Monitor database performance

## Advanced Topics
1. Connection Pooling
2. Database Migrations (Alembic)
3. Asynchronous Database Access
4. Sharding and Replication
5. Data Validation

## Next Steps
1. Learn about database indexing strategies
2. Explore database migrations with Alembic
3. Study database normalization (for SQL)
4. Learn about database transactions and ACID properties
5. Explore database monitoring and optimization techniques
