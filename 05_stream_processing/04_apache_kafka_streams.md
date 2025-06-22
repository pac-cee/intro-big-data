# Apache Kafka Streams

## Table of Contents
1. [Introduction to Kafka Streams](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Getting Started](#getting-started)
5. [Stream Processing with KStream and KTable](#stream-processing)
6. [Stateful Operations](#stateful-operations)
7. [Windowing and Time](#windowing-and-time)
8. [Interactive Queries](#interactive-queries)
9. [Scaling and Fault Tolerance](#scaling-and-fault-tolerance)
10. [Monitoring and Operations](#monitoring-and-operations)
11. [Best Practices](#best-practices)
12. [Case Studies](#case-studies)
13. [Further Reading](#further-reading)

## Introduction to Kafka Streams

Apache Kafka Streams is a client library for building applications and microservices that process and analyze data stored in Kafka. It provides a simple and lightweight way to perform stream processing on data in Kafka topics.

### Key Features

- **Lightweight**: Runs in your application as a library, no separate processing cluster needed
- **Scalable**: Automatically distributes work across your application instances
- **Fault-tolerant**: Handles machine failures transparently
- **Exactly-once processing**: Ensures each record is processed exactly once
- **Interactive queries**: Enables querying the current state of your application

## Core Concepts

### 1. Stream
A stream is the most important concept when working with Kafka Streams. It represents an unbounded, continuously updating sequence of immutable data records.

### 2. KStream
A KStream is an abstraction of a record stream, where each data record represents a self-contained datum in the unbounded data set.

### 3. KTable
A KTable is an abstraction of a changelog stream, where each data record represents an update.

### 4. GlobalKTable
A GlobalKTable is similar to a KTable but has a complete copy of the data on each instance.

### 5. Processor Topology
A graph of stream processors (nodes) that are connected by streams (edges).

## Architecture

### Components

1. **Streams DSL**
   - High-level API for most common operations
   - Built on top of the Processor API
   - Provides operators like `map`, `filter`, `join`, `aggregate`

2. **Processor API**
   - Lower-level API for more control
   - Allows custom processors and state stores
   - More complex but more flexible

3. **State Stores**
   - Local storage for stateful operations
   - Backed by RocksDB by default
   - Can be queried interactively

### Processing Guarantees

- **At-least-once**: Each record is processed at least once
- **Exactly-once**: Each record is processed exactly once
- **At-most-once**: Each record is processed at most once

## Getting Started

### Maven Dependencies

```xml
<dependency>
    <groupId>org.apache.kafka</groupId>
    <artifactId>kafka-streams</artifactId>
    <version>3.3.1</version>
</dependency>
<dependency>
    <groupId>org.apache.kafka</groupId>
    <artifactId>kafka-clients</artifactId>
    <version>3.3.1</version>
</dependency>
```

### Basic Example: Word Count

```java
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.*;

import java.util.Arrays;
import java.util.Properties;

public class WordCountApp {
    public static void main(String[] args) {
        // Configure the application
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "wordcount-application");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        // Define the processing topology
        StreamsBuilder builder = new StreamsBuilder();
        
        // Source: Read from input topic
        KStream<String, String> textLines = builder.stream("text-lines-topic");
        
        // Processing: Split text into words and count occurrences
        KTable<String, Long> wordCounts = textLines
            // Split each text line into words
            .flatMapValues(textLine -> Arrays.asList(textLine.toLowerCase().split("\\W+")))
            // Group by word
            .groupBy((key, word) -> word)
            // Count occurrences of each word
            .count();
        
        // Sink: Write results to output topic
        wordCounts.toStream().to("word-counts-topic", Produced.with(Serdes.String(), Serdes.Long()));

        // Build and start the streams application
        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
        
        // Add shutdown hook to respond to SIGTERM and shut down gracefully
        Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
    }
}
```

## Stream Processing with KStream and KTable

### KStream Operations

```java
// Filtering
KStream<String, String> filtered = stream.filter((key, value) -> value.contains("important"));

// Mapping
KStream<String, Integer> mapped = stream.map((key, value) -> 
    new KeyValue<>(key, value.length()));

// Branching
KStream<String, String>[] branches = stream.branch(
    (key, value) -> value.startsWith("A"),  // First predicate
    (key, value) -> value.startsWith("B"),  // Second predicate
    (key, value) -> true                    // Catch-all
);
KStream<String, String> startsWithA = branches[0];
KStream<String, String> startsWithB = branches[1];
KStream<String, String> others = branches[2];
```

### KTable Operations

```java
// Create a KTable from a topic
KTable<String, String> userTable = builder.table("users-topic");

// Filter
KTable<String, String> filteredTable = userTable.filter((key, value) -> 
    value != null && value.contains("active"));

// Map values
KTable<String, Integer> mappedTable = userTable.mapValues(value -> value.length());
```

### Joining Streams and Tables

```java
// Stream-Table Join
KStream<String, Order> orders = ...;
KTable<String, Customer> customers = ...;

// Join orders with customer data
KStream<String, EnrichedOrder> enrichedOrders = orders.leftJoin(
    customers,
    (orderId, order) -> order.getCustomerId(),  // Select key for join
    (order, customer) -> {
        EnrichedOrder enriched = new EnrichedOrder();
        enriched.setOrder(order);
        enriched.setCustomer(customer);
        return enriched;
    }
);

// Stream-Stream Join
KStream<String, Order> orders = ...;
KStream<String, Payment> payments = ...;

// Join orders with payments within 1-hour window
KStream<String, OrderPayment> orderPayments = orders.join(
    payments,
    (orderId, order) -> order.getId(),  // Select key for join
    (order, payment) -> new OrderPayment(order, payment),
    JoinWindows.of(Duration.ofHours(1)),
    StreamJoined.with(Serdes.String(), orderSerde, paymentSerde)
);
```

## Stateful Operations

### Aggregations

```java
// Count clicks per user
KTable<String, Long> userClickCounts = clickEvents
    .groupBy((key, click) -> click.getUserId())
    .count();

// Aggregate with custom logic
KTable<String, UserSession> userSessions = clickEvents
    .groupBy((key, click) -> click.getUserId())
    .aggregate(
        UserSession::new,  // Initializer
        (userId, click, session) -> session.updateWith(click),  // Aggregator
        Materialized.<String, UserSession, KeyValueStore<Bytes, byte[]>>as("user-sessions-store")
            .withKeySerde(Serdes.String())
            .withValueSerde(userSessionSerde)
    );
```

### State Stores

```java
// Create a state store
StoreBuilder<KeyValueStore<String, Long>> storeBuilder = Stores.keyValueStoreBuilder(
    Stores.persistentKeyValueStore("my-state-store"),
    Serdes.String(),
    Serdes.Long()
);

// Add the store to the topology
builder.addStateStore(storeBuilder);

// Use the store in a processor
KStream<String, String> stream = builder.stream("input-topic");
stream.process(() -> new Processor<String, String>() {
    private KeyValueStore<String, Long> store;
    
    @Override
    public void init(ProcessorContext context) {
        this.store = (KeyValueStore<String, Long>) context.getStateStore("my-state-store");
    }
    
    @Override
    public void process(String key, String value) {
        Long count = store.get(key);
        if (count == null) {
            count = 0L;
        }
        count++;
        store.put(key, count);
        context().forward(key, count);
    }
    
    @Override
    public void close() {}
}, "my-state-store");
```

## Windowing and Time

### Tumbling Windows

```java
// 5-minute tumbling window
KTable<Windowed<String>, Long> windowedCounts = clickEvents
    .groupBy((key, click) -> click.getPageId())
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .count();
```

### Hopping Windows

```java
// 5-minute window, advancing by 1 minute
KTable<Windowed<String>, Long> hoppingCounts = clickEvents
    .groupBy((key, click) -> click.getPageId())
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)).advanceBy(Duration.ofMinutes(1)))
    .count();
```

### Session Windows

```java
// Session window with 5-minute inactivity gap
KTable<Windowed<String>, Long> sessionCounts = clickEvents
    .groupBy((key, click) -> click.getUserId())
    .windowedBy(SessionWindows.with(Duration.ofMinutes(5)))
    .count();
```

## Interactive Queries

### Querying Local State Stores

```java
// Get a reference to the state store
ReadOnlyKeyValueStore<String, Long> keyValueStore =
    streams.store("word-counts-store", QueryableStoreTypes.keyValueStore());

// Query the state store
Long count = keyValueStore.get("hello");

// Get all key-value pairs
KeyValueIterator<String, Long> range = keyValueStore.all();
while (range.hasNext()) {
    KeyValue<String, Long> next = range.next();
    System.out.println("Word: " + next.key + " -> Count: " + next.value);
}
range.close();
```

### Querying Remote State Stores

```java
// Get metadata about all instances of the application
Set<HostInfo> hosts = streams.allMetadataForStore("word-counts-store");

// For each host, make an HTTP request to query its state
for (HostInfo host : hosts) {
    // Make HTTP request to http://host.hostname:8080/wordcount/hello
    // (You need to implement the REST endpoint in your application)
}
```

## Scaling and Fault Tolerance

### Scaling Out

1. **Partitioning**: Kafka Streams automatically partitions data based on the input topic's partitioning
2. **Rebalancing**: When instances are added or removed, Kafka Streams rebalances the workload
3. **State Migration**: State is migrated between instances during rebalancing

### Handling Failures

1. **Task Recovery**: Failed tasks are restarted on available instances
2. **State Recovery**: Local state is restored from the changelog topic
3. **Processing Guarantees**: Exactly-once processing ensures no duplicates or lost data

### Configuration

```java
Properties props = new Properties();
// Enable exactly-once processing
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, StreamsConfig.EXACTLY_ONCE_V2);
// Set number of standby replicas for faster recovery
props.put(StreamsConfig.NUM_STANDBY_REPLICAS_CONFIG, 1);
// Set commit interval for offset commits
props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 5000);
```

## Monitoring and Operations

### Metrics

Kafka Streams exposes metrics via JMX:

- **Processing rate**: `process-rate`, `process-total`
- **Commit rate**: `commit-rate`, `commit-total`
- **Task metrics**: `active-tasks`, `standby-tasks`
- **State store metrics**: `put-rate`, `get-rate`, `flush-rate`

### Logging

```java
// Enable debug logging for troubleshooting
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyStreamsApp {
    private static final Logger log = LoggerFactory.getLogger(MyStreamsApp.class);
    
    public static void main(String[] args) {
        // ...
        streams.setUncaughtExceptionHandler((thread, throwable) -> {
            log.error("Uncaught exception in thread " + thread + ": ", throwable);
        });
    }
}
```

## Best Practices

1. **Use Exactly-Once Processing**
   ```java
   props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, StreamsConfig.EXACTLY_ONCE_V2);
   ```

2. **Optimize State Stores**
   - Use appropriate serdes
   - Consider using in-memory stores for high-throughput, low-latency use cases
   - Tune RocksDB configuration for state stores

3. **Handle Deserialization Errors**
   ```java
   props.put(StreamsConfig.DEFAULT_DESERIALIZATION_EXCEPTION_HANDLER_CLASS_CONFIG, 
       LogAndContinueExceptionHandler.class);
   ```

4. **Monitor and Tune Performance**
   - Monitor consumer lag
   - Adjust `num.stream.threads` based on CPU cores
   - Tune `commit.interval.ms` based on your latency requirements

## Case Studies

### 1. Real-time Analytics at LinkedIn
- **Use Case**: Real-time metrics and monitoring
- **Scale**: Processes trillions of events per day
- **Key Insight**: Used Kafka Streams for its simplicity and exactly-once processing

### 2. Fraud Detection at a Payment Processor
- **Use Case**: Real-time fraud detection
- **Scale**: Millions of transactions per hour
- **Key Insight**: Used stateful processing to detect patterns of fraudulent activity

### 3. Personalized Recommendations at a Media Company
- **Use Case**: Real-time content recommendations
- **Scale**: Billions of user interactions per day
- **Key Insight**: Used KTable-KTable joins to maintain user profiles and content metadata

## Further Reading

1. [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
2. [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936153/)
3. [Designing Event-Driven Systems](https://www.oreilly.com/library/view/designing-event-driven-systems/9781492038252/)
4. [Kafka Streams in Action](https://www.manning.com/books/kafka-streams-in-action)
5. [Confluent Kafka Streams Tutorials](https://developer.confluent.io/tutorials/)
