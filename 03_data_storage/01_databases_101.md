# Databases 101: SQL and NoSQL

## Table of Contents
1. [Introduction to Databases](#introduction-to-databases)
2. [Relational Databases (SQL)](#relational-databases-sql)
3. [NoSQL Databases](#nosql-databases)
4. [Database Design](#database-design)
5. [Working with Python](#working-with-python)
6. [Big Data Storage Solutions](#big-data-storage-solutions)

## Introduction to Databases

A database is an organized collection of data stored and accessed electronically.

### Types of Databases
- **Relational (SQL)**: Structured data with predefined schemas
- **NoSQL**: Flexible schema for unstructured/semi-structured data
- **NewSQL**: Combines SQL and NoSQL features
- **Time-Series**: Optimized for time-stamped data
- **Graph**: For highly connected data

## Relational Databases (SQL)

### Key Concepts
- **Tables**: Rows and columns
- **Primary Key**: Unique identifier for each record
- **Foreign Key**: Creates relationships between tables
- **Indexes**: Improve query performance
- **ACID Properties**: Atomicity, Consistency, Isolation, Durability

### SQL Basics

```sql
-- Create a table
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data
INSERT INTO users (id, username, email) 
VALUES (1, 'john_doe', 'john@example.com');

-- Query data
SELECT * FROM users WHERE username = 'john_doe';

-- Update data
UPDATE users SET email = 'john.doe@example.com' WHERE id = 1;

-- Delete data
DELETE FROM users WHERE id = 1;
```

### Popular SQL Databases
1. **PostgreSQL**: Advanced open-source RDBMS
2. **MySQL**: Popular open-source database
3. **SQLite**: Lightweight, file-based database
4. **Microsoft SQL Server**: Enterprise RDBMS
5. **Oracle**: Enterprise-grade RDBMS

## NoSQL Databases

### Types of NoSQL Databases

1. **Document Stores**
   - MongoDB, CouchDB
   - Store data in JSON-like documents
   
2. **Key-Value Stores**
   - Redis, DynamoDB
   - Simple key-value pairs
   
3. **Wide-Column Stores**
   - Cassandra, HBase
   - Columns are stored together
   
4. **Graph Databases**
   - Neo4j, Amazon Neptune
   - Nodes, edges, and properties

### MongoDB Example

```javascript
// Insert document
db.users.insertOne({
    name: "John Doe",
    email: "john@example.com",
    age: 30,
    address: {
        street: "123 Main St",
        city: "New York"
    },
    tags: ["customer", "premium"]
});

// Find documents
db.users.find({ age: { $gt: 25 } });

// Update document
db.users.updateOne(
    { name: "John Doe" },
    { $set: { age: 31 } }
);
```

## Database Design

### Normalization
1. **1NF**: Atomic values, no repeating groups
2. **2NF**: No partial dependencies
3. **3NF**: No transitive dependencies

### Indexing
- Improves query performance
- Slows down write operations
- Common index types: B-tree, Hash, Bitmap

### Partitioning
- **Horizontal**: Splitting rows across multiple tables
- **Vertical**: Splitting columns into separate tables

## Working with Python

### SQLite with Python
```python
import sqlite3

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS users
                  (id INTEGER PRIMARY KEY, name TEXT, email TEXT)''')

# Insert data
cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
               ('John Doe', 'john@example.com'))

# Query data
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())

# Commit changes and close
conn.commit()
conn.close()
```

### MongoDB with Python
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
collection.insert_one(user)

# Find documents
for user in collection.find({'age': {'$gt': 25}}):
    print(user)
```

## Big Data Storage Solutions

### Data Lakes
- Store raw data in native format
- Examples: Amazon S3, Azure Data Lake, Google Cloud Storage

### Data Warehouses
- Optimized for analytics
- Examples: Snowflake, Google BigQuery, Amazon Redshift

### Distributed File Systems
- HDFS (Hadoop Distributed File System)
- Google File System (GFS)

## Practice Exercises
1. Design a normalized database schema for a blog
2. Write SQL queries for common operations
3. Set up MongoDB and perform CRUD operations
4. Compare performance of different indexing strategies
5. Design a data lake architecture for a large organization

---
Next: [Working with Data Warehouses](./02_data_warehousing.md)
