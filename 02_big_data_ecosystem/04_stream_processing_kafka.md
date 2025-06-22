# Stream Processing with Apache Kafka

## Table of Contents
1. [Introduction to Stream Processing](#introduction-to-stream-processing)
2. [Kafka Core Concepts](#kafka-core-concepts)
3. [Kafka Architecture](#kafka-architecture)
4. [Producers and Consumers](#producers-and-consumers)
5. [Kafka Streams](#kafka-streams)
6. [Kafka Connect](#kafka-connect)
7. [Kafka Security](#kafka-security)
8. [Deployment and Operations](#deployment-and-operations)
9. [Kafka with Other Technologies](#kafka-with-other-technologies)

## Introduction to Stream Processing

### What is Stream Processing?
- Processing of data in motion
- Real-time or near real-time analysis
- Enables immediate insights and actions

### Use Cases
- Real-time analytics
- Event-driven architectures
- Fraud detection
- IoT data processing
- Log and metrics collection

### Stream Processing Patterns
- Event sourcing
- Complex event processing
- Stream-table join
- Windowed aggregations

## Kafka Core Concepts

### Key Components
- **Topics**: Categories or feed names to which messages are published
- **Partitions**: Ordered, immutable sequence of records
- **Offsets**: Unique ID of a record within a partition
- **Brokers**: Kafka servers that store data and serve clients
- **Cluster**: Collection of brokers
- **ZooKeeper**: Manages and coordinates Kafka brokers (being phased out in newer versions)

### Message Structure
- **Key** (optional): Used for partitioning
- **Value**: The actual message content
- **Headers** (optional): Key-value pairs for metadata
- **Timestamp**: When the message was created

## Kafka Architecture

### Broker Architecture
```
+----------------+      +----------------+      +----------------+
|   Producer 1   |      |   Producer 2   |      |   Producer N   |
+--------+-------+      +--------+-------+      +--------+-------+
         |                    |                         |
         +--------------------+-------------------------+
                              |
                     +--------v--------+
                     |   Kafka Broker  |
                     |   (Leader)     |
                     +--------+--------+
                              |
         +--------------------+-------------------------+
         |                    |                         |
+--------v-------+  +--------v-------+  +----------------+
|  Follower 1    |  |  Follower 2    |  |  Follower N    |
+----------------+  +----------------+  +----------------+
```

### Topic Partitions and Replication
- Each partition has one leader and multiple followers
- Producers write to the leader
- Followers replicate the leader's data
- If leader fails, a follower becomes the new leader

## Producers and Consumers

### Producer API
```java
// Producer properties
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// Create producer
Producer<String, String> producer = new KafkaProducer<>(props);

// Send message
ProducerRecord<String, String> record = 
    new ProducerRecord<>("my-topic", "key", "value");
producer.send(record, (metadata, exception) -> {
    if (exception == null) {
        System.out.printf("Sent record to %s-%d [%d]\n",
            metadata.topic(), metadata.partition(), metadata.offset());
    } else {
        exception.printStackTrace();
    }
});

// Close producer
producer.close();
```

### Consumer API
```java
// Consumer properties
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "test-group");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

// Create consumer
KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);

// Subscribe to topic
consumer.subscribe(Collections.singletonList("my-topic"));

try {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        for (ConsumerRecord<String, String> record : records) {
            System.out.printf("offset = %d, key = %s, value = %s%n",
                record.offset(), record.key(), record.value());
        }
    }
} finally {
    consumer.close();
}
```

## Kafka Streams

### Stream Processing with Kafka Streams
```java
// Streams configuration
Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "wordcount-application");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

// Streams builder
StreamsBuilder builder = new StreamsBuilder();

// Create source stream from input topic
KStream<String, String> textLines = builder.stream("text-lines");

// Process the stream
KTable<String, Long> wordCounts = textLines
    .flatMapValues(textLine -> Arrays.asList(textLine.toLowerCase().split("\\W+")))
    .groupBy((key, word) -> word)
    .count();

// Write results to output topic
wordCounts.toStream().to("words-with-counts", Produced.with(Serdes.String(), Serdes.Long()));

// Build and start the topology
KafkaStreams streams = new KafkaStreams(builder.build(), props);
streams.start();

// Add shutdown hook to clean up
Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
```

## Kafka Connect

### Source and Sink Connectors
```properties
# Source connector configuration
name=file-source
connector.class=FileStreamSource
tasks.max=1
file=/tmp/input.txt
topic=connect-test

# Sink connector configuration
name=file-sink
connector.class=FileStreamSink
tasks.max=1
file=/tmp/output.txt
topics=connect-test
```

### Running Connect
```bash
# Start a standalone connector
bin/connect-standalone.sh config/connect-standalone.properties \
    config/connect-file-source.properties \
    config/connect-file-sink.properties
```

## Kafka Security

### Security Features
- **Encryption**: SSL/TLS for data in transit
- **Authentication**: SASL (Simple Authentication and Security Layer)
- **Authorization**: ACLs (Access Control Lists)
- **Auditing**: Logging and monitoring

### Configuration Example
```properties
# SSL configuration
security.protocol=SSL
ssl.truststore.location=/path/to/truststore.jks
ssl.truststore.password=password
ssl.keystore.location=/path/to/keystore.jks
ssl.keystore.password=password
ssl.key.password=password

# SASL configuration
security.protocol=SASL_SSL
sasl.mechanism=SCRAM-SHA-256
sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required \
    username="alice" \
    password="alice-secret";
```

## Deployment and Operations

### Hardware Requirements
- **CPU**: 8+ cores for production
- **Memory**: 16GB+ RAM for brokers
- **Storage**: High-performance SSDs
- **Network**: 10Gbps+ recommended

### Configuration Tuning
```properties
# Broker configuration
broker.id=1
log.dirs=/data/kafka
num.partitions=8
log.retention.hours=168
log.segment.bytes=1073741824
num.io.threads=8
num.network.threads=3
num.replica.fetchers=2

# Producer configuration
acks=all
retries=2147483647
max.in.flight.requests.per.connection=5
compression.type=snappy

# Consumer configuration
group.id=my-group
enable.auto.commit=true
auto.commit.interval.ms=1000
session.timeout.ms=30000
```

### Monitoring
- **JMX Metrics**: Enable JMX for monitoring
- **Kafka Manager**: Web-based tool for managing Kafka
- **Burrow**: Consumer lag monitoring
- **Prometheus + Grafana**: Metrics collection and visualization

## Kafka with Other Technologies

### Kafka and Spark Streaming
```scala
// Create a direct stream
val kafkaStream = KafkaUtils.createDirectStream[
  String, String,
  StringDecoder, StringDecoder](
    ssc, kafkaParams, topicsSet)

// Process the stream
val lines = kafkaStream.map(_._2)
val words = lines.flatMap(_.split(" "))
val wordCounts = words.map(x => (x, 1L)).reduceByKey(_ + _)
wordCounts.print()

// Start the computation
ssc.start()
ssc.awaitTermination()
```

### Kafka and Flink
```java
// Create Kafka consumer
FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
    "input-topic",
    new SimpleStringSchema(),
    properties);

// Add source to the execution environment
DataStream<String> stream = env.addSource(consumer);

// Process the stream
DataStream<Tuple2<String, Integer>> wordCounts = stream
    .flatMap((String line, Collector<Tuple2<String, Integer>> out) -> {
        for (String word : line.split(" ")) {
            out.collect(new Tuple2<>(word, 1));
        }
    })
    .returns(Types.TUPLE(Types.STRING, Types.INT))
    .keyBy(0)
    .sum(1);

// Write to Kafka sink
wordCounts.addSink(new FlinkKafkaProducer<>(
    "output-topic",
    new KafkaSerializationSchema<Tuple2<String, Integer>>() {
        @Override
        public ProducerRecord<byte[], byte[]> serialize(
            Tuple2<String, Integer> element, Long timestamp) {
            return new ProducerRecord<>(
                "output-topic",
                (element.f0 + ":" + element.f1).getBytes());
        }
    },
    properties,
    FlinkKafkaProducer.Semantic.EXACTLY_ONCE));
```

## Practice Exercises
1. Set up a local Kafka cluster with 3 brokers
2. Create a producer that sends messages to a topic
3. Create a consumer that reads messages from the topic
4. Implement a word count application using Kafka Streams
5. Set up a Kafka Connect pipeline to read from a file and write to a topic
6. Configure SSL encryption for your Kafka cluster
7. Monitor your Kafka cluster using JMX and Grafana
8. Integrate Kafka with Spark Streaming for real-time processing

---
Next: [Batch Processing with Apache Flink](./05_batch_processing_flink.md)
