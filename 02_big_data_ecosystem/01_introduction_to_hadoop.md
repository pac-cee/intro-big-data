# Introduction to Hadoop Ecosystem

## Table of Contents
1. [What is Big Data?](#what-is-big-data)
2. [Hadoop Overview](#hadoop-overview)
3. [HDFS (Hadoop Distributed File System)](#hdfs)
4. [MapReduce](#mapreduce)
5. [YARN](#yarn)
6. [Hadoop Ecosystem Components](#hadoop-ecosystem-components)
7. [Setting Up Hadoop](#setting-up-hadoop)

## What is Big Data?

Big Data refers to extremely large datasets that cannot be processed using traditional computing techniques. It is characterized by the 3Vs:

- **Volume**: Scale of data
- **Velocity**: Speed of data processing
- **Variety**: Different forms of data (structured, semi-structured, unstructured)

## Hadoop Overview

Apache Hadoop is an open-source framework that allows for the distributed processing of large datasets across clusters of computers using simple programming models.

### Key Features
- **Fault Tolerance**: Data is replicated across multiple nodes
- **Scalability**: Can scale from single servers to thousands of machines
- **Cost-Effective**: Uses commodity hardware
- **Flexible**: Can process any type of data

## HDFS (Hadoop Distributed File System)

HDFS is the primary storage system used by Hadoop applications.

### Architecture
- **NameNode**: Master server that manages the file system namespace
- **DataNode**: Slaves that store and retrieve blocks
- **Secondary NameNode**: Performs periodic checkpoints

### HDFS Commands
```bash
# List files
hadoop fs -ls /

# Create directory
hadoop fs -mkdir /user/hadoop/dir1

# Copy from local to HDFS
hadoop fs -put localfile /user/hadoop/hadoopfile

# View file
hadoop fs -cat /user/hadoop/hadoopfile
```

## MapReduce

A programming model for processing large datasets in parallel.

### MapReduce Example: Word Count

**Mapper (Python)**
```python
#!/usr/bin/env python
import sys

for line in sys.stdin:
    words = line.strip().split()
    for word in words:
        print(f"{word}\t1")
```

**Reducer (Python)**
```python
#!/usr/bin/env python
import sys

current_word = None
current_count = 0

for line in sys.stdin:
    word, count = line.strip().split('\t')
    count = int(count)
    
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

if current_word:
    print(f"{current_word}\t{current_count}")
```

## YARN (Yet Another Resource Negotiator)

YARN is the resource management layer of Hadoop.

### Components
- **ResourceManager**: Global resource scheduler
- **NodeManager**: Per-machine framework agent
- **ApplicationMaster**: Per-application framework

## Hadoop Ecosystem Components

1. **HBase**: NoSQL database
2. **Hive**: Data warehouse infrastructure
3. **Pig**: High-level data flow language
4. **Sqoop**: Data transfer between Hadoop and relational databases
5. **Flume**: Service for collecting and moving large amounts of log data
6. **Kafka**: Distributed streaming platform

## Setting Up Hadoop

### Prerequisites
- Java 8 or later
- SSH installed and running
- Password-less SSH login

### Installation Steps
1. Download Hadoop from [Apache Hadoop Releases](https://hadoop.apache.org/releases.html)
2. Extract the tarball
3. Configure environment variables
4. Set up configuration files:
   - core-site.xml
   - hdfs-site.xml
   - mapred-site.xml
   - yarn-site.xml
5. Format the HDFS filesystem
6. Start the Hadoop daemons

## Practice Exercises
1. Set up a single-node Hadoop cluster
2. Run the WordCount example
3. Store and retrieve files from HDFS
4. Write a simple MapReduce program
5. Explore Hive and Pig for data processing

---
Next: [Introduction to Apache Spark](./02_introduction_to_spark.md)
