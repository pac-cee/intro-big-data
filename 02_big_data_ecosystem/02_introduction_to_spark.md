# Introduction to Apache Spark

## Table of Contents
1. [What is Apache Spark?](#what-is-apache-spark)
2. [Spark Architecture](#spark-architecture)
3. [Resilient Distributed Datasets (RDDs)](#resilient-distributed-datasets-rdds)
4. [DataFrames and Datasets](#dataframes-and-datasets)
5. [Spark SQL](#spark-sql)
6. [Structured Streaming](#structured-streaming)
7. [Setting Up Spark](#setting-up-spark)
8. [Spark on Kubernetes](#spark-on-kubernetes)

## What is Apache Spark?

Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java, Scala, Python, and R, and an optimized engine that supports general execution graphs.

### Key Features
- **Speed**: Up to 100x faster than Hadoop MapReduce in memory
- **Ease of Use**: Rich APIs in multiple languages
- **Generality**: Combines SQL, streaming, and complex analytics
- **Compatibility**: Runs on Hadoop, Kubernetes, Mesos, standalone, or in the cloud
- **Fault Tolerance**: Automatically recovers from node failures

## Spark Architecture

### Components
1. **Driver Program**: Runs the main() function and creates SparkContext
2. **Cluster Manager**: Manages and allocates resources (Standalone, YARN, Mesos, Kubernetes)
3. **Worker Node**: Runs the application code in the cluster
4. **Executor**: Runs tasks and keeps data in memory or disk storage

### Execution Model
- **Transformations**: Lazy operations (e.g., map, filter, join)
- **Actions**: Trigger computation (e.g., count, collect, save)
- **DAG Scheduler**: Optimizes the execution plan

## Resilient Distributed Datasets (RDDs)

RDDs are the fundamental data structure of Spark, representing an immutable, partitioned collection of elements that can be operated on in parallel.

### Creating RDDs
```python
# From a collection
data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(data)

# From a file
rdd = spark.sparkContext.textFile("hdfs://path/to/file.txt")
```

### RDD Operations
```python
# Transformations (lazy)
mapped_rdd = rdd.map(lambda x: (x, 1))
filtered_rdd = rdd.filter(lambda x: x > 2)
reduced_rdd = rdd.reduceByKey(lambda a, b: a + b)

# Actions (eager)
count = rdd.count()
first = rdd.first()
collected = rdd.collect()
```

## DataFrames and Datasets

### DataFrames
A distributed collection of data organized into named columns, similar to a table in a relational database.

```python
# Create DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Example") \
    .getOrCreate()

df = spark.createDataFrame([
    (1, "Alice", 25),
    (2, "Bob", 30),
    (3, "Charlie", 35)
], ["id", "name", "age"])

# Show the first few rows
df.show()

# Perform operations
df.select("name", "age").filter("age > 30").show()
```

### Datasets
Type-safe, object-oriented programming interface, available in Scala and Java.

## Spark SQL

Spark SQL lets you query structured data using SQL or the DataFrame API.

```python
# Register DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")

# SQL query
result = spark.sql("""
    SELECT name, age 
    FROM people 
    WHERE age > 25
""")
result.show()
```

### Reading/Writing Data
```python
# Read from various sources
df_csv = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)
df_json = spark.read.json("path/to/file.json")
df_parquet = spark.read.parquet("path/to/file.parquet")

# Write to different formats
df.write.parquet("output.parquet")
df.write.json("output.json")
```

## Structured Streaming

Spark's engine for scalable and fault-tolerant stream processing.

```python
# Read from socket
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Process the streaming data
word_counts = lines.groupBy("value").count()

# Start the streaming query
query = word_counts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

## Setting Up Spark

### Prerequisites
- Java 8 or later
- Python 3.6+ (for PySpark)
- (Optional) Scala (for Scala API)

### Installation
1. Download Spark from [Apache Spark Downloads](https://spark.apache.org/downloads.html)
2. Extract the tarball
3. Set environment variables:
   ```bash
   export SPARK_HOME=/path/to/spark
   export PATH=$PATH:$SPARK_HOME/bin
   ```
4. Test the installation:
   ```bash
   pyspark --version
   ```

### Running Spark Applications
```bash
# Run interactively with PySpark
pyspark

# Submit a Python application
spark-submit your_script.py

# Run with specific configuration
spark-submit --master local[4] --executor-memory 4g your_script.py
```

## Spark on Kubernetes

### Deploying Spark on Kubernetes
1. Build Spark with Kubernetes support
2. Create a Docker image
3. Deploy to a Kubernetes cluster

### Example: Running a Spark job on Kubernetes
```bash
# Build Spark with Kubernetes support
./build/mvn -Pkubernetes -DskipTests clean package

# Build the Docker image
./bin/docker-image-tool.sh -r <registry> -t my-tag build

# Submit a job
bin/spark-submit \
    --master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    --deploy-mode cluster \
    --name spark-pi \
    --class org.apache.spark.examples.SparkPi \
    --conf spark.executor.instances=5 \
    --conf spark.kubernetes.container.image=<spark-image> \
    local:///path/to/examples.jar 1000
```

## Practice Exercises
1. Set up a local Spark environment
2. Create an RDD and perform basic transformations and actions
3. Load a CSV file into a DataFrame and run SQL queries
4. Implement a simple streaming application
5. Deploy a Spark application on Kubernetes

---
Next: [Introduction to Distributed Storage Systems](./03_distributed_storage_systems.md)
