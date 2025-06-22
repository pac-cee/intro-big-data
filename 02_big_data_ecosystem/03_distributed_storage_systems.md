# Distributed Storage Systems

## Table of Contents
1. [Introduction to Distributed Storage](#introduction-to-distributed-storage)
2. [HDFS Architecture](#hdfs-architecture)
3. [Apache HBase](#apache-hbase)
4. [Apache Cassandra](#apache-cassandra)
5. [Amazon S3](#amazon-s3)
6. [Google Cloud Storage](#google-cloud-storage)
7. [Data Replication and Consistency](#data-replication-and-consistency)
8. [Performance Considerations](#performance-considerations)

## Introduction to Distributed Storage

### Characteristics of Distributed Storage
- **Scalability**: Ability to handle growing amounts of data
- **Fault Tolerance**: Continue operating properly in case of failures
- **High Availability**: Ensure data is always accessible
- **Consistency**: Maintain data consistency across nodes
- **Partition Tolerance**: Continue operating despite network partitions

### CAP Theorem
- **Consistency**: Every read receives the most recent write
- **Availability**: Every request receives a response
- **Partition Tolerance**: System continues to operate despite network partitions

## HDFS Architecture

### Core Components
- **NameNode**: Manages file system metadata
- **DataNode**: Stores actual data blocks
- **Secondary NameNode**: Performs periodic checkpoints

### HDFS Features
- **Block-based storage** (default 128MB blocks)
- **Data replication** (default 3x)
- **Rack awareness**
- **High throughput access**

### HDFS Commands
```bash
# List files
hdfs dfs -ls /path

# Copy from local to HDFS
hdfs dfs -put localfile /hdfs/path

# Copy from HDFS to local
hdfs dfs -get /hdfs/path localfile

# Check file size
hdfs dfs -du -h /path

# Check disk usage
hdfs dfs -df -h
```

## Apache HBase

### Overview
- Distributed, scalable, big data store
- Modeled after Google's Bigtable
- Provides real-time read/write access to large datasets

### Data Model
- **Tables**: Collections of rows
- **Row Keys**: Unique identifier for each row
- **Column Families**: Logical grouping of columns
- **Cells**: Intersection of row and column, containing a value and timestamp

### Basic Operations
```java
// Create table
HBaseAdmin admin = new HBaseAdmin(config);
HTableDescriptor table = new HTableDescriptor(TableName.valueOf("test"));
table.addFamily(new HColumnDescriptor("cf"));
admin.createTable(table);

// Put data
Table t = connection.getTable(TableName.valueOf("test"));
Put p = new Put(Bytes.toBytes("row1"));
p.addColumn(Bytes.toBytes("cf"), Bytes.toBytes("col1"), Bytes.toBytes("value1"));
t.put(p);

// Get data
Get g = new Get(Bytes.toBytes("row1"));
Result r = t.get(g);
byte[] value = r.getValue(Bytes.toBytes("cf"), Bytes.toBytes("col1"));
```

## Apache Cassandra

### Overview
- Highly scalable, high-performance distributed database
- Designed to handle large amounts of data across many commodity servers
- No single point of failure

### Data Model
- **Keyspace**: Similar to a database in RDBMS
- **Table**: Collection of rows
- **Partition Key**: Determines data distribution across nodes
- **Clustering Columns**: Determine sort order within a partition

### CQL (Cassandra Query Language)
```sql
-- Create keyspace
CREATE KEYSPACE IF NOT EXISTS my_keyspace
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 3
};

-- Create table
CREATE TABLE my_keyspace.users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT,
    created_at TIMESTAMP
);

-- Insert data
INSERT INTO my_keyspace.users (user_id, name, email, created_at)
VALUES (uuid(), 'John Doe', 'john@example.com', toTimestamp(now()));

-- Query data
SELECT * FROM my_keyspace.users WHERE user_id = ?;
```

## Amazon S3

### Overview
- Object storage service
- 99.999999999% (11 9's) durability
- 99.99% availability
- Virtually unlimited storage

### Key Concepts
- **Buckets**: Containers for objects
- **Objects**: Fundamental entities (files)
- **Keys**: Unique identifiers for objects
- **Regions**: Geographic locations for storing buckets

### S3 Operations with AWS CLI
```bash
# Create bucket
aws s3 mb s3://my-bucket --region us-west-2

# Upload file
aws s3 cp localfile.txt s3://my-bucket/

# Download file
aws s3 cp s3://my-bucket/remotefile.txt ./

# List objects
aws s3 ls s3://my-bucket/

# Sync directories
aws s3 sync ./local/folder s3://my-bucket/remote/folder
```

## Google Cloud Storage

### Overview
- Unified object storage
- Strongly consistent
- Integrated with Google Cloud services

### Storage Classes
- **Standard**: Frequently accessed data
- **Nearline**: Accessed less than once a month
- **Coldline**: Accessed less than once a quarter
- **Archive**: Lowest cost, long-term storage

### GCS Operations with gsutil
```bash
# Create bucket
gsutil mb -l US-EAST1 gs://my-bucket

# Upload file
gsutil cp localfile.txt gs://my-bucket/

# Download file
gsutil cp gs://my-bucket/remotefile.txt ./

# List objects
gsutil ls gs://my-bucket/

# Set bucket storage class
gsutil defstorageclass set COLDLINE gs://my-bucket
```

## Data Replication and Consistency

### Replication Strategies
1. **Synchronous Replication**
   - Data written to multiple replicas before acknowledging write
   - Strong consistency
   - Higher latency

2. **Asynchronous Replication**
   - Acknowledge write after primary
   - Eventual consistency
   - Lower latency

### Consistency Models
- **Strong Consistency**: All reads see most recent write
- **Eventual Consistency**: All replicas will eventually converge
- **Read-Your-Writes Consistency**: Read after write sees own writes
- **Monotonic Reads**: Successive reads see same or more recent data

### Quorum-Based Replication
- **Write Quorum (W)**: Number of nodes that must acknowledge write
- **Read Quorum (R)**: Number of nodes that must respond to read
- **Replication Factor (N)**: Total number of copies
- **Consistency**: R + W > N

## Performance Considerations

### Storage Optimization
- **Compression**: Reduce storage space and I/O
- **Partitioning**: Distribute data across nodes
- **Caching**: Keep frequently accessed data in memory
- **Data Locality**: Process data where it's stored

### Monitoring and Tuning
- **Metrics to Monitor**:
  - I/O throughput
  - Latency
  - CPU and memory usage
  - Network bandwidth
  - Disk space

- **Tuning Parameters**:
  - Block size
  - Replication factor
  - Memory allocation
  - Thread pool sizes

## Practice Exercises
1. Set up a multi-node HDFS cluster
2. Create and query a table in HBase
3. Design a data model for a time-series application in Cassandra
4. Upload and manage files in S3 using AWS CLI
5. Implement a quorum-based replication strategy
6. Benchmark read/write performance with different consistency levels

---
Next: [Stream Processing with Apache Kafka](./04_stream_processing_kafka.md)
