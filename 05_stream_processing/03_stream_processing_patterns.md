# Stream Processing Patterns

## Table of Contents
1. [Introduction to Stream Processing Patterns](#introduction)
2. [Basic Processing Patterns](#basic-processing-patterns)
3. [Windowing Patterns](#windowing-patterns)
4. [State Management Patterns](#state-management-patterns)
5. [Time Processing Patterns](#time-processing-patterns)
6. [Fault Tolerance Patterns](#fault-tolerance-patterns)
7. [Exactly-Once Processing](#exactly-once-processing)
8. [Patterns for Scaling](#patterns-for-scaling)
9. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
10. [Case Studies](#case-studies)
11. [Further Reading](#further-reading)

## Introduction

Stream processing patterns are reusable solutions to common problems encountered when building streaming applications. Understanding these patterns helps in designing robust, scalable, and maintainable stream processing systems.

## Basic Processing Patterns

### 1. Filtering

Select only relevant events based on certain conditions.

```python
# Python example using PySpark Streaming
filtered_stream = stream.filter(lambda x: x["severity"] == "HIGH")
```

### 2. Mapping

Transform each event into a new format.

```python
# Python example using PyFlink
mapped_stream = stream.map(lambda x: {"user": x["user_id"], "action": x["event_type"]})
```

### 3. Aggregation

Compute aggregates over a stream of events.

```python
# Python example using Kafka Streams
word_counts = text_lines.flat_map_values(lambda text_line: text_line.lower().split("\\s+")) \
    .group_by(lambda (dummy, word): word) \
    .count()
```

### 4. Joining Streams

Combine data from multiple streams.

```python
# Python example using PySpark Streaming
joined_stream = stream1.join(stream2, "user_id")
```

## Windowing Patterns

### 1. Tumbling Windows

Fixed-size, non-overlapping time windows.

```python
# PyFlink example
t_env.execute_sql("""
    SELECT 
        user_id, 
        TUMBLE_START(ts, INTERVAL '5' MINUTE) as window_start,
        COUNT(*) as click_count
    FROM user_clicks
    GROUP BY 
        user_id,
        TUMBLE(ts, INTERVAL '5' MINUTE)
""")
```

### 2. Sliding Windows

Fixed-size, overlapping time windows.

```python
# PySpark example
windowed_counts = events \
    .withWatermark("timestamp", "5 minutes") \
    .groupBy(
        window("timestamp", "10 minutes", "5 minutes"),
        "user_id"
    ) \
    .count()
```

### 3. Session Windows

Windows of activity separated by periods of inactivity.

```python
# PyFlink example
session_window = env.add_source(source) \
    .key_by(lambda x: x[0]) \
    .window(EventTimeSessionWindows.with_gap(Time.minutes(15))) \
    .process(MyProcessWindowFunction())
```

## State Management Patterns

### 1. Keyed State

State that is scoped to a specific key.

```python
# PyFlink example with keyed state
class KeyedStateFunction(KeyedProcessFunction):
    def __init__(self):
        self.state = None
    
    def open(self, parameters):
        state_descriptor = ValueStateDescriptor("state", Types.INT())
        self.state = self.get_runtime_context().get_state(state_descriptor)
    
    def process_element(self, value, ctx):
        current = self.state.value() or 0
        current += 1
        self.state.update(current)
        yield (value[0], current)
```

### 2. Operator State

State that is scoped to an operator instance.

```python
# PyFlink example with operator state
class BufferingSink(SinkFunction):
    def __init__(self, threshold):
        self.threshold = threshold
        self.checkpointed_state = None
    
    def initialize_state(self, context):
        state_descriptor = ListStateDescriptor("buffered-elements", Types.STRING())
        self.checkpointed_state = context.get_operator_state(state_descriptor)
        
        if context.is_restored():
            for element in self.checkpointed_state.get():
                self.buffered_elements.append(element)
    
    def invoke(self, value, context):
        self.buffered_elements.append(value)
        if len(self.buffered_elements) >= self.threshold:
            self.send_to_external_system()
            self.buffered_elements = []
    
    def snapshot_state(self, context):
        self.checkpointed_state.clear()
        for element in self.buffered_elements:
            self.checkpointed_state.add(element)
```

## Time Processing Patterns

### 1. Event Time Processing

Process events based on when they actually occurred.

```python
# PyFlink example with event time
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)

stream = env.add_source(source) \
    .assign_timestamps_and_watermarks(
        WatermarkStrategy
            .for_bounded_out_of_orderness(Duration.of_seconds(5))
            .with_timestamp_assigner(MyTimestampAssigner())
    )
```

### 2. Handling Late Data

```python
# PySpark example with late data handling
windowed_counts = events \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window("timestamp", "5 minutes"),
        "user_id"
    ) \
    .count()
    .withColumn("is_late", col("window.end") < current_timestamp() - expr("interval 10 minutes"))
```

## Fault Tolerance Patterns

### 1. Checkpointing

Periodically save application state for recovery.

```python
# PyFlink checkpointing configuration
env.enable_checkpointing(10000)  # Checkpoint every 10 seconds
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
env.get_checkpoint_config().set_min_pause_between_checkpoints(5000)
env.get_checkpoint_config().set_checkpoint_timeout(60000)
```

### 2. Savepoints

Manual checkpoints for versioning and updates.

```bash
# Create a savepoint
flink savepoint <job_id> [savepoint_directory]

# Resume from savepoint
flink run -s :savepoint_path [:runArgs]
```

## Exactly-Once Processing

### 1. End-to-End Exactly-Once

```python
# PyFlink example with Kafka exactly-once
kafka_consumer = KafkaSource.builder() \
    .set_bootstrap_servers("kafka:9092") \
    .set_topics("input-topic") \
    .set_group_id("my-group") \
    .set_starting_offsets(KafkaOffsetsInitializer.earliest()) \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()

kafka_producer = KafkaSink.builder() \
    .set_bootstrap_servers("kafka:9092") \
    .set_record_serializer(KafkaRecordSerializationSchema.builder()
        .set_topic("output-topic")
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    ) \
    .set_delivery_guarantee(DeliveryGuarantee.EXACTLY_ONCE) \
    .build()

# Build and execute the pipeline
env.from_source(kafka_consumer, WatermarkStrategy.no_watermarks(), "Kafka Source") \
    .sink_to(kafka_producer)

env.execute("Exactly-Once Processing")
```

## Patterns for Scaling

### 1. Key-Based Partitioning

```python
# PySpark example with custom partitioning
stream.repartition("user_id")  # Distribute by user_id
```

### 2. Dynamic Scaling

```python
# Flink example with reactive scaling
# In flink-conf.yaml
jobmanager.scheduler: adaptive
jobmanager.adaptive-scheduler.resource-wait-timeout: 5min
jobmanager.adaptive-scheduler.resource-stabilization-timeout: 30s
```

## Anti-Patterns to Avoid

### 1. Global State in Operators

```python
# Bad: Using global state
count = 0
def process(value):
    global count
    count += 1
    return (value, count)

# Good: Use framework's state management
class StatefulProcess(ProcessFunction):
    def process_element(self, value, ctx):
        current = self.state.value() or 0
        current += 1
        self.state.update(current)
        return (value, current)
```

### 2. Blocking Operations in Process Functions

```python
# Bad: Blocking call in process function
def process(value):
    # This blocks the thread!
    result = requests.get("http://external-service/api")
    return process_result(result)

# Good: Use async I/O
class AsyncDatabaseRequest(AsyncFunction):
    async def async_invoke(self, input, result):
        # Non-blocking async call
        result.complete(await async_database_lookup(input))
```

## Case Studies

### 1. Real-time Fraud Detection at Scale
- Pattern: Complex Event Processing (CEP)
- Technology: Apache Flink
- Key Insight: Stateful processing with time windows to detect suspicious patterns

### 2. Real-time Recommendations
- Pattern: Session-based processing
- Technology: Apache Kafka Streams
- Key Insight: Maintaining user session state for personalized recommendations

### 3. IoT Data Processing
- Pattern: Time-series aggregation
- Technology: Apache Spark Streaming
- Key Insight: Efficient windowing for sensor data aggregation

## Further Reading

1. [Streaming Systems](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) by Tyler Akidau et al.
2. [Designing Data-Intensive Applications](https://dataintensive.net/) by Martin Kleppmann
3. [Flink Documentation: Event Time](https://ci.apache.org/projects/flink/flink-docs-stable/dev/event_time.html)
4. [Kafka Streams Developer Guide](https://kafka.apache.org/documentation/streams/)
5. [The Dataflow Model](https://research.google/pubs/pub43864/)
