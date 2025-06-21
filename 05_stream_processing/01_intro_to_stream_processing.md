# Introduction to Stream Processing

## Table of Contents
1. [What is Stream Processing?](#what-is-stream-processing)
2. [Key Concepts](#key-concepts)
3. [Stream Processing Architectures](#stream-processing-architectures)
4. [Stream Processing with Apache Kafka](#stream-processing-with-apache-kafka)
5. [Stream Processing with Apache Flink](#stream-processing-with-apache-flink)
6. [Real-time Analytics](#real-time-analytics)
7. [Handling Late and Out-of-Order Data](#handling-late-and-out-of-order-data)
8. [Practice Exercises](#practice-exercises)

## What is Stream Processing?

Stream processing is a data processing paradigm designed to handle continuous streams of data in real-time or near real-time.

### Batch vs Stream Processing

| Feature          | Batch Processing          | Stream Processing           |
|-----------------|--------------------------|----------------------------|
| Data            | Bounded data             | Unbounded data streams     |
| Latency        | High (minutes to hours)  | Low (milliseconds to seconds) |
| Processing Time | Fixed intervals          | Continuous                 |
| Examples       | Daily reports, ETL jobs  | Fraud detection, monitoring |

## Key Concepts

### 1. Event Time vs Processing Time
- **Event Time**: When the event actually occurred
- **Processing Time**: When the event is processed by the system

### 2. Windowing
Breaking up a data stream into finite chunks for processing:
- **Tumbling Windows**: Fixed-size, non-overlapping windows
- **Sliding Windows**: Fixed-size, overlapping windows
- **Session Windows**: Windows of activity separated by gaps of inactivity

### 3. State Management
- **Operator State**: State that is scoped to an operator instance
- **Keyed State**: State that is scoped to a key in the stream
- **Checkpointing**: Periodically saving state to persistent storage

## Stream Processing Architectures

### 1. Lambda Architecture
- Batch Layer: Handles historical data
- Speed Layer: Handles real-time data
- Serving Layer: Merges results

### 2. Kappa Architecture
- Single processing layer for both batch and streaming
- Uses stream processing for all data

### 3. Stateful Stream Processing
- Maintains state across events
- Enables complex event processing

## Stream Processing with Apache Kafka

### Kafka Streams Basics

```java
// Java example
StreamsBuilder builder = new StreamsBuilder();
KStream<String, String> textLines = builder.stream("input-topic");

// Word count example
KTable<String, Long> wordCounts = textLines
    .flatMapValues(textLine -> Arrays.asList(textLine.toLowerCase().split("\\W+")))
    .groupBy((key, word) -> word)
    .count();

wordCounts.toStream().to("word-count-output", Produced.with(Serdes.String(), Serdes.Long()));

KafkaStreams streams = new KafkaStreams(builder.build(), config);
streams.start();
```

### Kafka Connect
- Source Connectors: Ingest data into Kafka
- Sink Connectors: Export data from Kafka

## Stream Processing with Apache Flink

### Flink DataStream API

```java
// Java example
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Read from socket
DataStream<String> text = env.socketTextStream("localhost", 9999);

// Process stream
DataStream<Tuple2<String, Integer>> wordCounts = text
    .flatMap((String line, Collector<Tuple2<String, Integer>> out) -> {
        for (String word : line.split("\\s")) {
            out.collect(new Tuple2<>(word, 1));
        }
    })
    .returns(Types.TUPLE(Types.STRING, Types.INT))
    .keyBy(0)
    .sum(1);

// Print results
wordCounts.print();

// Execute program
env.execute("WordCount");
```

### Flink Table API
```python
# Python example
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Create source table
t_env.execute_sql("""
    CREATE TABLE orders (
        order_id STRING,
        product STRING,
        amount DOUBLE,
        order_time TIMESTAMP(3),
        WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'orders',
        'properties.bootstrap.servers' = 'localhost:9092',
        'properties.group.id' = 'order_consumer',
        'format' = 'json',
        'scan.startup.mode' = 'earliest-offset'
    )
""")

# Execute query
result = t_env.sql_query("""
    SELECT 
        window_start, 
        window_end, 
        product, 
        SUM(amount) as total_amount,
        COUNT(*) as order_count
    FROM TABLE(
        TUMBLE(TABLE orders, DESCRIPTOR(order_time), INTERVAL '1' HOURS)
    )
    GROUP BY window_start, window_end, product
""")

# Print results
result.execute().print()
```

## Real-time Analytics

### Use Cases
1. **Fraud Detection**
   - Detect suspicious transactions in real-time
   - Example: Multiple transactions from different locations in short time

2. **IoT Monitoring**
   - Process sensor data in real-time
   - Trigger alerts for abnormal conditions

3. **Recommendation Systems**
   - Update recommendations based on user activity
   - Personalize content in real-time

### Example: Real-time Dashboard
```python
from flask import Flask, render_template
from pykafka import KafkaClient
import json

app = Flask(__name__)

def get_kafka_consumer():
    client = KafkaClient(hosts="localhost:9092")
    topic = client.topics['metrics']
    return topic.get_simple_consumer()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/metrics')
def metrics():
    consumer = get_kafka_consumer()
    metrics = []
    
    # Get latest 100 messages
    for _ in range(100):
        message = consumer.consume()
        if message:
            metrics.append(json.loads(message.value.decode()))
    
    return json.dumps(metrics[-100:])  # Return at most 100 latest metrics

if __name__ == '__main__':
    app.run(debug=True)
```

## Handling Late and Out-of-Order Data

### Watermarks
- Mechanism to track progress of event time
- Defines how long to wait for late data

### Allowed Lateness
- Configurable time to wait for late data
- Updates results when late data arrives

### Side Outputs
- Route late data to a separate stream
- Process late data differently

## Practice Exercises
1. **Basic Word Count**
   - Set up a Kafka producer to send text lines
   - Create a Flink job to count words in real-time
   - Output results to a Kafka topic

2. **Real-time Aggregation**
   - Process a stream of e-commerce transactions
   - Calculate total sales by category in 5-minute windows
   - Handle late-arriving transactions

3. **Anomaly Detection**
   - Monitor a stream of server metrics
   - Detect and alert on abnormal CPU/memory usage
   - Implement a simple moving average for baseline

4. **Session Analysis**
   - Process user clickstream data
   - Identify user sessions with 30 minutes of inactivity
   - Calculate session duration and pages viewed

5. **Joining Streams**
   - Join clickstream data with user profile data
   - Enrich events with user information in real-time
   - Handle late-arriving dimension data

## Resources
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/streams/)
- [Apache Flink Documentation](https://ci.apache.org/projects/flink/flink-docs-stable/)
- [Kafka Streams Developer Guide](https://kafka.apache.org/26/documentation/streams/developer-guide/)
- [Flink Stateful Functions](https://ci.apache.org/projects/flink-statefun/stable/)

---
Next: [Advanced Stream Processing Patterns](./02_advanced_stream_processing.md)
