# Batch Processing with Apache Flink

## Table of Contents
1. [Introduction to Batch Processing](#introduction-to-batch-processing)
2. [Flink Architecture](#flink-architecture)
3. [DataSet API](#dataset-api)
4. [Table API & SQL](#table-api--sql)
5. [Connectors](#connectors)
6. [Performance Tuning](#performance-tuning)
7. [Flink Deployment](#flink-deployment)
8. [Monitoring and Metrics](#monitoring-and-metrics)

## Introduction to Batch Processing

### What is Batch Processing?
- Processing of finite datasets
- High throughput, high latency
- Ideal for large-scale data analysis

### Use Cases
- ETL (Extract, Transform, Load) pipelines
- Data warehousing
- Business intelligence
- Large-scale data analysis

### Batch vs. Streaming
| Feature          | Batch Processing       | Stream Processing      |
|-----------------|------------------------|------------------------|
| Data            | Bounded (finite)       | Unbounded (infinite)   |
| Latency         | High (minutes to hours)| Low (milliseconds)     |
| Throughput      | Very high             | High                  |
| Fault Tolerance | Checkpointing          | Exactly-once semantics |
| Examples        | Daily reports, ETL    | Fraud detection, monitoring |

## Flink Architecture

### Core Components
- **JobManager**: Coordinates distributed execution
- **TaskManager**: Executes tasks and manages resources
- **Client**: Submits jobs to the JobManager
- **ResourceManager**: Manages task slots
- **Dispatcher**: Provides REST interface

### Execution Model
```
+-------------------+     +-------------------+     +-------------------+
|    Data Source   | --> |   Flink Job      | --> |   Data Sink      |
|  (e.g., HDFS)    |     |  (Transformations)|     |  (e.g., HBase)   |
+-------------------+     +-------------------+     +-------------------+
```

### Execution Plans
```java
// Define the execution environment
ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

// Read input data
DataSet<String> text = env.readTextFile("hdfs://path/to/input");

// Define transformations
DataSet<Tuple2<String, Integer>> counts = 
    text.flatMap(new Tokenizer())
        .groupBy(0)
        .sum(1);

// Write output
counts.writeAsText("hdfs://path/to/output");

// Execute the job
env.execute("Word Count Example");
```

## DataSet API

### Transformations
```java
// Map: 1-to-1 transformation
DataSet<Integer> numbers = env.fromElements(1, 2, 3, 4, 5);
DataSet<Integer> doubled = numbers.map(x -> x * 2);

// FlatMap: 1-to-many transformation
DataSet<String> lines = env.fromElements("hello world", "hello flink");
DataSet<String> words = lines.flatMap((String line, Collector<String> out) -> {
    for (String word : line.split(" ")) {
        out.collect(word);
    }
}).returns(Types.STRING);

// Filter: Keep elements that satisfy a condition
DataSet<Integer> evenNumbers = numbers.filter(x -> x % 2 == 0);

// Join: Join two datasets on a key
DataSet<Tuple2<Integer, String>> persons = ...
DataSet<Tuple2<Integer, String>> salaries = ...
DataSet<Tuple3<Integer, String, String>> joined = 
    persons.join(salaries)
           .where(0)  // key of the first dataset (persons)
           .equalTo(0)  // key of the second dataset (salaries)
           .with((p, s) -> new Tuple3<>(p.f0, p.f1, s.f1));

// Reduce: Combine elements
DataSet<Integer> sum = numbers.reduce((a, b) -> a + b);
```

### Data Sources and Sinks
```java
// Read from a CSV file
DataSet<Tuple3<Integer, String, Double>> csvInput = 
    env.readCsvFile("hdfs://path/to/input.csv")
       .types(Integer.class, String.class, Double.class);

// Write to a text file
counts.writeAsText("hdfs://path/to/output");

// Write to a CSV file
csvInput.writeAsCsv("hdfs://path/to/output.csv", "\n", ",");

// Write to a database
counts.output(new JDBCOutputFormat(
    JDBC_PROPERTIES,
    "INSERT INTO wordcount (word, count) VALUES (?, ?)",
    new JdbcStatementBuilder<Tuple2<String, Integer>>() {
        @Override
        public void accept(PreparedStatement stmt, Tuple2<String, Integer> record) 
                throws SQLException {
            stmt.setString(1, record.f0);
            stmt.setInt(2, record.f1);
        }
    }
));
```

## Table API & SQL

### Table Environment Setup
```java
// Create a batch TableEnvironment
ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
BatchTableEnvironment tableEnv = BatchTableEnvironment.create(env);

// Register a table
Table table = tableEnv.fromDataSet(dataset, "name, age");
tableEnv.createTemporaryView("People", table);

// Execute SQL query
Table result = tableEnv.sqlQuery(
    "SELECT name, AVG(age) FROM People GROUP BY name");

// Convert Table back to DataSet
DataSet<Row> resultSet = tableEnv.toDataSet(result, Row.class);
```

### SQL Operations
```sql
-- Create a table from a CSV file
CREATE TABLE Orders (
    order_id INT,
    product_id INT,
    amount INT,
    order_time TIMESTAMP
) WITH (
    'connector' = 'filesystem',
    'path' = 'file:///path/to/orders.csv',
    'format' = 'csv'
);

-- Query the table
SELECT 
    product_id, 
    SUM(amount) as total_amount
FROM Orders
GROUP BY product_id
ORDER BY total_amount DESC;

-- Join with another table
SELECT 
    o.order_id,
    p.product_name,
    o.amount
FROM Orders o
JOIN Products p ON o.product_id = p.id;
```

## Connectors

### File System Connector
```java
// Read from a text file
DataSet<String> text = env.readTextFile("hdfs://path/to/input");

// Write to a text file
counts.writeAsText("hdfs://path/to/output");

// Read a CSV file
DataSet<Tuple3<Integer, String, Double>> csvInput = 
    env.readCsvFile("hdfs://path/to/input.csv")
       .types(Integer.class, String.class, Double.class);
```

### JDBC Connector
```java
// Read from a database
DataSet<Tuple2<Integer, String>> dbData = env.createInput(
    JDBCInputFormat.buildJDBCInputFormat()
        .setDrivername("org.postgresql.Driver")
        .setDBUrl("jdbc:postgresql://localhost:5432/mydb")
        .setUsername("user")
        .setPassword("password")
        .setQuery("SELECT id, name FROM users")
        .finish(),
    new TupleTypeInfo<Tuple2<Integer, String>>(
        BasicTypeInfo.INT_TYPE_INFO,
        BasicTypeInfo.STRING_TYPE_INFO
    )
);

// Write to a database
counts.output(new JDBCOutputFormat(
    JDBC_PROPERTIES,
    "INSERT INTO wordcount (word, count) VALUES (?, ?)",
    new JdbcStatementBuilder<Tuple2<String, Integer>>() {
        @Override
        public void accept(PreparedStatement stmt, Tuple2<String, Integer> record) 
                throws SQLException {
            stmt.setString(1, record.f0);
            stmt.setInt(2, record.f1);
        }
    }
));
```

## Performance Tuning

### Memory Configuration
```bash
# Set task manager memory
taskmanager.memory.process.size: 1600m

taskmanager.memory.task.heap.size: 1g
taskmanager.memory.managed.size: 256m

# Set network memory
taskmanager.memory.network.min: 64m
taskmanager.memory.network.max: 128m

# Set JVM parameters
taskmanager.jvm-opts: >
    -XX:+UseG1GC
    -XX:MaxGCPauseMillis=50
    -XX:G1HeapRegionSize=32m
```

### Parallelism and Resources
```java
// Set default parallelism
env.setParallelism(4);

// Set parallelism for a specific operator
DataSet<String> result = input
    .map(new MyMapper())
    .setParallelism(4);

// Configure task slots
taskmanager.numberOfTaskSlots: 4
```

### Optimizing Joins
```java
// Broadcast join for small datasets
DataSet<Tuple2<Integer, String>> smallDataSet = ...
DataSet<Tuple2<Integer, String>> largeDataSet = ...

// Broadcast the small dataset
largeDataSet.join(smallDataSet.broadcast())
    .where(0)  // key of the first dataset
    .equalTo(0) // key of the second dataset
    .with(new JoinFunction<...>() {
        // Join logic
    });

// Sort-merge join for large datasets
dataSet1.join(dataSet2)
    .where(0)  // key of the first dataset
    .equalTo(0) // key of the second dataset
    .withJoinHint(JoinHint.OPTIMIZER_CHOOSES);
```

## Flink Deployment

### Standalone Cluster
```bash
# Start a local cluster
./bin/start-cluster.sh

# Submit a job
./bin/flink run -c com.example.MainJob /path/to/your-job.jar

# List running jobs
./bin/flink list

# Cancel a job
./bin/flink cancel <job-id>
```

### YARN Deployment
```bash
# Start a Flink session on YARN
./bin/yarn-session.sh -n 4 -jm 1024 -tm 4096

# Submit a job to the YARN session
./bin/flink run -m yarn-cluster -yn 4 -yjm 1024 -ytm 4096 /path/to/your-job.jar
```

### Kubernetes Deployment
```yaml
# flink-configuration-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flink-config
  labels:
    app: flink
    component: jobmanager
    type: configmap
    version: 1.15.0
data:
  flink-conf.yaml: |+
    jobmanager.rpc.address: flink-jobmanager
    taskmanager.numberOfTaskSlots: "2"
    blob.server.port: 6124
    jobmanager.rpc.port: 6123
    taskmanager.rpc.port: 6122
    query.server.port: 6125
```

## Monitoring and Metrics

### Built-in Web UI
- Accessible at http://jobmanager:8081
- Shows job status, task metrics, and logs
- Provides backpressure monitoring

### Metrics System
```java
// Register a custom metric
getRuntimeContext()
    .getMetricGroup()
    .gauge("my-gauge", new Gauge<Long>() {
        @Override
        public Long getValue() {
            return System.currentTimeMillis();
        }
    });

// Counter example
private transient Counter counter;

@Override
public void open(Configuration config) {
    this.counter = getRuntimeContext()
        .getMetricGroup()
        .counter("my-counter");
}

@Override
public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
    this.counter.inc();
    // ...
}
```

### External Monitoring
- **Prometheus**: Pull-based metrics collection
- **Grafana**: Visualization of metrics
- **ELK Stack**: Log aggregation and analysis

## Practice Exercises
1. Set up a local Flink cluster
2. Implement a batch job that reads from a CSV file, processes the data, and writes to a database
3. Optimize a join operation between two large datasets
4. Deploy a Flink job on a YARN cluster
5. Set up monitoring for a Flink job using Prometheus and Grafana
6. Implement a custom source and sink for your specific data format
7. Tune the performance of a Flink job by adjusting memory and parallelism settings

---
Next: [Real-time Analytics with Apache Druid](./06_realtime_analytics_druid.md)
