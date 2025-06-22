# Introduction to Stream Processing

## Table of Contents
1. [What is Stream Processing?](#what-is-stream-processing)
2. [Key Concepts](#key-concepts)
3. [Stream Processing Architectures](#stream-processing-architectures)
4. [Stream Processing with Apache Kafka](#stream-processing-with-apache-kafka)
5. [Stream Processing with Apache Flink](#stream-processing-with-apache-flink)
6. [Stream Processing with Apache Spark](#stream-processing-with-apache-spark)
7. [Real-time Analytics](#real-time-analytics)
8. [Handling Late and Out-of-Order Data](#handling-late-and-out-of-order-data)
9. [Use Cases and Applications](#use-cases-and-applications)
10. [Practice Exercises](#practice-exercises)
11. [Further Reading](#further-reading)

## What is Stream Processing?

Stream processing is a data processing paradigm designed to handle continuous, unbounded streams of data in real-time or near real-time. Unlike traditional batch processing, which operates on finite datasets, stream processing systems are designed to handle data that is continuously generated, often with high velocity and volume.

### Batch vs Stream Processing

| Feature                | Batch Processing                      | Stream Processing                     |
|------------------------|--------------------------------------|--------------------------------------|
| **Data Scope**        | Bounded, finite datasets             | Unbounded, continuous streams        |
| **Latency**          | High (minutes to hours)              | Low (milliseconds to seconds)        |
| **Processing Model**  | Process and then store               | Store then process                   |
| **Analysis Type**     | Historical, comprehensive            | Real-time, incremental               |
| **Fault Tolerance**   | Easier to implement                  | More complex to implement            |
| **State Management**  | Simpler, at rest                    | Complex, continuously updated        |
| **Use Cases**         | Monthly reports, analytics           | Fraud detection, monitoring, alerts  |

### When to Use Stream Processing

Stream processing is particularly useful when:

1. **Low Latency** is required (e.g., fraud detection, monitoring)
2. **Continuous Data** is being generated (e.g., IoT, clickstreams)
3. **Real-time Decisions** are needed (e.g., stock trading, recommendations)
4. **Event-driven** architectures are in use

## Key Concepts

### 1. Event Time vs Processing Time

- **Event Time**: When the event actually occurred
- **Processing Time**: When the event is processed by the system

```python
# Example of event time vs processing time in Python
from datetime import datetime
import time

# Event creation (event time)
event = {
    'user_id': 'user123',
    'action': 'click',
    'timestamp': '2025-06-22T10:00:00Z',  # Event time
    'processing_time': datetime.utcnow().isoformat()  # Processing time
}
```

### 2. Windows

Windows define how to group events for processing:

- **Tumbling Windows**: Fixed-size, non-overlapping windows
- **Sliding Windows**: Fixed-size, overlapping windows
- **Session Windows**: Activity-based windows

```python
# Example of windowing in PySpark Streaming
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, count

spark = SparkSession.builder.appName("WindowExample").getOrCreate()

# Read from socket for demonstration
lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

# Group by 10-minute tumbling windows
windowed_counts = lines.groupBy(
    window(lines.timestamp, "10 minutes"),
    lines.value
).count()

# Start the streaming query
query = windowed_counts.writeStream.outputMode("complete").format("console").start()
query.awaitTermination()
```

### 3. Watermarks

Watermarks help handle late-arriving data by defining how long to wait for late events:

```python
# Example of watermarks in PyFlink
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.descriptors import Schema, Json, Kafka

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define a table with event-time and watermark
t_env.execute_sql("""
    CREATE TABLE sensor_readings (
        sensor_id STRING,
        reading DOUBLE,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'sensor_readings',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")
```

## Stream Processing Architectures

### 1. Lambda Architecture

- **Batch Layer**: Processes all available data with high latency
- **Speed Layer**: Processes recent data with low latency
- **Serving Layer**: Merges results from both layers

### 2. Kappa Architecture

- Single processing layer for both batch and streaming
- Uses stream processing for all data
- Replays historical data when needed

### 3. Stateful Stream Processing

- Maintains state across events
- Enables complex event processing
- Examples: Session windows, aggregations

## Stream Processing with Apache Kafka

Kafka Streams is a client library for building applications that process and analyze data stored in Kafka.

```java
// Example Kafka Streams application in Java
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.*;

public class WordCountApp {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "wordcount-app");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        StreamsBuilder builder = new StreamsBuilder();
        KStream<String, String> textLines = builder.stream("text-lines");
        
        KTable<String, Long> wordCounts = textLines
            .flatMapValues(textLine -> Arrays.asList(textLine.toLowerCase().split("\\W+")))
            .groupBy((key, word) -> word)
            .count(Materialized.as("counts-store"));
            
        wordCounts.toStream().to("words-with-counts", Produced.with(Serdes.String(), Serdes.Long()));
        
        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
    }
}
```

## Stream Processing with Apache Flink

Apache Flink is a framework for stateful computations over unbounded and bounded data streams.

```java
// Example Flink streaming job in Java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;

public class SocketTextStreamWordCount {
    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        DataStream<String> text = env.socketTextStream("localhost", 9999, "\n");
        
        DataStream<WordWithCount> wordCounts = text
            .flatMap(new FlatMapFunction<String, WordWithCount>() {
                @Override
                public void flatMap(String value, Collector<WordWithCount> out) {
                    for (String word : value.split("\\s")) {
                        out.collect(new WordWithCount(word, 1L));
                    }
                }
            })
            .keyBy("word")
            .sum("count");
            
        wordCounts.print().setParallelism(1);
        
        env.execute("Socket WordCount");
    }
    
    public static class WordWithCount {
        public String word;
        public long count;
        
        public WordWithCount() {}
        
        public WordWithCount(String word, long count) {
            this.word = word;
            this.count = count;
        }
        
        @Override
        public String toString() {
            return word + " : " + count;
        }
    }
}
```

## Stream Processing with Apache Spark

Apache Spark Streaming provides a high-level abstraction called discretized stream or DStream.

```python
# Example Spark Streaming application in Python
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working threads and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.socketTextStream("localhost", 9999)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
word_counts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
word_counts.pprint()

# Start the computation
ssc.start()
# Wait for the computation to terminate
ssc.awaitTermination()
```

## Real-time Analytics

### 1. Real-time Aggregations

```sql
-- Example using ksqlDB for real-time aggregations
CREATE STREAM page_views (
    viewtime BIGINT,
    userid VARCHAR,
    pageid VARCHAR
) WITH (
    kafka_topic='page_views',
    value_format='JSON'
);

-- Create a table with the number of views per page
CREATE TABLE page_view_counts AS
    SELECT
        pageid,
        COUNT(*) AS view_count
    FROM page_views
    WINDOW TUMBLING (SIZE 1 MINUTE)
    GROUP BY pageid
    EMIT CHANGES;
```

## Handling Late and Out-of-Order Data

### 1. Allowed Lateness

```python
# Example in PyFlink showing allowed lateness
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common import Duration

# Define watermark strategy with 5 seconds allowed lateness
watermark_strategy = WatermarkStrategy\
    .for_bounded_out_of_orderness(Duration.of_seconds(5))\
    .with_timestamp_assigner(MyTimestampAssigner())

# Apply to data stream
stream = env.add_source(source).assign_timestamps_and_watermarks(watermark_strategy)
```

## Use Cases and Applications

### 1. Real-time Fraud Detection

- Analyze transactions in real-time
- Detect patterns indicating fraudulent activity
- Block suspicious transactions before completion

### 2. IoT Data Processing

- Process sensor data from connected devices
- Trigger alerts based on thresholds
- Monitor equipment health in real-time

### 3. Recommendation Systems

- Update user profiles in real-time
- Provide personalized recommendations
- Track user behavior across sessions

## Practice Exercises

1. **Basic Streaming Word Count**
   - Create a simple streaming application that counts words from a socket
   - Try different window sizes and types

2. **Event Time Processing**
   - Process a stream of events with timestamps
   - Handle late-arriving data using watermarks

3. **Stateful Processing**
   - Implement a session-based analysis
   - Maintain state across events

## Further Reading

1. [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
2. [Apache Flink Documentation](https://flink.apache.org/)
3. [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
4. [Stream Processing with Apache Flink](https://www.oreilly.com/library/view/stream-processing-with/9781491974285/)
5. [Designing Data-Intensive Applications](https://dataintensive.net/) by Martin Kleppmann
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
