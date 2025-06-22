# Apache Flink Streaming

## Table of Contents
1. [Introduction to Apache Flink](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Getting Started](#getting-started)
5. [DataStream API](#datastream-api)
6. [State & Fault Tolerance](#state-and-fault-tolerance)
7. [Event Time & Watermarks](#event-time-and-watermarks)
8. [Windowing](#windowing)
9. [Connectors](#connectors)
10. [Scalability & Performance](#scalability-and-performance)
11. [Deployment](#deployment)
12. [Monitoring & Metrics](#monitoring-and-metrics)
13. [Best Practices](#best-practices)
14. [Case Studies](#case-studies)
15. [Further Reading](#further-reading)

## Introduction to Apache Flink

Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams. Flink is designed to run in all common cluster environments, perform computations at in-memory speed, and at any scale.

### Key Features

- **Event-time processing** with out-of-order event handling
- **Exactly-once state consistency**
- **Milliseconds latency** with millions of events per second
- **Scalability** to thousands of nodes
- **Flexible windowing** based on time, count, or custom triggers
- **Fault tolerance** with lightweight snapshots
- **Backpressure handling**
- **Batch processing** as a special case of streaming

## Core Concepts

### 1. DataStream API

```java
// Example of a simple Flink streaming job
DataStream<String> text = env.socketTextStream("localhost", 9999);
DataStream<Tuple2<String, Integer>> wordCounts = text
    .flatMap((String line, Collector<Tuple2<String, Integer>> out) -> {
        for (String word : line.split("\\s")) {
            out.collect(new Tuple2<>(word, 1));
        }
    })
    .returns(Types.TUPLE(Types.STRING, Types.INT))
    .keyBy(0)
    .sum(1);

wordCounts.print();
```

### 2. Event Time vs Processing Time

```java
// Set up event time processing
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

// Create a watermark strategy
WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    .withTimestampAssigner((event, timestamp) -> event.getCreationTime());

// Apply watermark strategy to source
DataStream<Event> events = env
    .addSource(new CustomSource())
    .assignTimestampsAndWatermarks(watermarkStrategy);
```

### 3. State & Checkpoints

```java
// Enable checkpointing every 10 seconds
env.enableCheckpointing(10000, CheckpointingMode.EXACTLY_ONCE);

// Configure state backend
env.setStateBackend(new FsStateBackend("hdfs:///checkpoints"));

// Configure restart strategy
env.setRestartStrategy(RestartStrategies.fixedDelayRestart(
    3, // number of restart attempts
    Time.seconds(10) // delay
));
```

## Architecture

### Components

1. **JobManager**
   - Controls the execution of a single application
   - Schedules tasks, coordinates checkpoints, and handles failures
   - High-availability setup with multiple standby nodes

2. **TaskManager**
   - Executes the tasks of a job
   - Manages memory and network buffers
   - Reports task status to the JobManager

3. **ResourceManager**
   - Manages task slots
   - Handles resource allocation and deallocation
   - Works with external resource managers (YARN, Kubernetes, etc.)

4. **Dispatcher**
   - Accepts job submissions
   - Starts a new JobManager for each job
   - Provides a web UI for job monitoring

### Execution Model

1. **Tasks and Operator Chains**
   - Operators can be chained together in the same thread
   - Reduces serialization/deserialization overhead
   - Improves throughput and reduces latency

2. **Task Slots**
   - Unit of resource allocation
   - Multiple tasks can share the same slot (slot sharing)
   - Default: one slot per TaskManager

## Getting Started

### Maven Dependencies

```xml
<dependencies>
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-java</artifactId>
        <version>1.16.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-streaming-java_2.12</artifactId>
        <version>1.16.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-clients_2.12</artifactId>
        <version>1.16.0</version>
    </dependency>
</dependencies>
```

### Basic Example: Word Count

```java
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class WordCount {
    public static void main(String[] args) throws Exception {
        // Set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Read text from socket
        DataStream<String> text = env.socketTextStream("localhost", 9999);
        
        // Parse the data, group it, and aggregate
        DataStream<Tuple2<String, Integer>> counts = text
            .flatMap(new Tokenizer())
            .keyBy(value -> value.f0)
            .sum(1);
        
        // Print the results
        counts.print();
        
        // Execute the Flink job
        env.execute("Word Count Example");
    }
    
    // User-defined function for tokenizing the input
    public static final class Tokenizer 
        implements FlatMapFunction<String, Tuple2<String, Integer>> {
        
        @Override
        public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
            // Split the line into words
            String[] words = value.toLowerCase().split("\\W+");
            
            // Emit each word with count 1
            for (String word : words) {
                if (word.length() > 0) {
                    out.collect(new Tuple2<>(word, 1));
                }
            }
        }
    }
}
```

## DataStream API

### Sources

```java
// Read from a socket
DataStream<String> stream = env.socketTextStream("localhost", 9999);

// Read from a file
DataStream<String> stream = env.readTextFile("file:///path/to/file");

// Read from Kafka
Properties properties = new Properties();
properties.setProperty("bootstrap.servers", "localhost:9092");
properties.setProperty("group.id", "test");

FlinkKafkaConsumer<String> consumer = new FlinkKafkaConsumer<>(
    "topic",
    new SimpleStringSchema(),
    properties
);
DataStream<String> stream = env.addSource(consumer);

// Custom source
DataStream<Event> events = env.addSource(new CustomSource());
```

### Transformations

```java
// Map: 1-to-1 transformation
DataStream<Integer> lengths = stream.map(String::length);

// FlatMap: 1-to-many transformation
DataStream<String> words = stream.flatMap((String value, Collector<String> out) -> {
    for (String word : value.split("\\s")) {
        out.collect(word);
    }
}).returns(Types.STRING);

// Filter
DataStream<String> filtered = stream.filter(value -> value.startsWith("A"));

// KeyBy: Logical partitioning
DataStream<Tuple2<String, Integer>> keyed = stream
    .map(word -> new Tuple2<>(word, 1))
    .returns(Types.TUPLE(Types.STRING, Types.INT))
    .keyBy(0);

// Reduce
DataStream<Tuple2<String, Integer>> wordCounts = keyed.sum(1);

// Union: Combine multiple streams
DataStream<String> stream1 = ...;
DataStream<String> stream2 = ...;
DataStream<String> combined = stream1.union(stream2);

// Connect: Connect two streams while preserving their types
ConnectedStreams<String, Integer> connected = stream1.connect(stream2);
DataStream<String> result = connected
    .map(new CoMapFunction<String, Integer, String>() {
        @Override
        public String map1(String value) {
            return "STRING: " + value;
        }
        @Override
        public String map2(Integer value) {
            return "INT: " + value.toString();
        }
    });
```

### Sinks

```java
// Print to stdout
stream.print();

// Write to file
stream.writeAsText("file:///path/to/output");

// Write to Kafka
FlinkKafkaProducer<String> producer = new FlinkKafkaProducer<>(
    "output-topic",
    new SimpleStringSchema(),
    properties
);
stream.addSink(producer);

// Custom sink
stream.addSink(new CustomSink());
```

## State and Fault Tolerance

### Keyed State

```java
public class CountWindowAverage extends RichFlatMapFunction<Long, Tuple2<Long, Double>> {
    
    private transient ValueState<Tuple2<Long, Long>> sum; // (count, sum)
    
    @Override
    public void open(Configuration config) {
        ValueStateDescriptor<Tuple2<Long, Long>> descriptor =
            new ValueStateDescriptor<>(
                "average", // state name
                TypeInformation.of(new TypeHint<Tuple2<Long, Long>>() {}));
        sum = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void flatMap(Long input, Collector<Tuple2<Long, Double>> out) throws Exception {
        // Access the state value
        Tuple2<Long, Long> currentSum = sum.value();
        
        // Update the count and sum
        if (currentSum == null) {
            currentSum = Tuple2.of(0L, 0L);
        }
        currentSum.f0 += 1;  // count
        currentSum.f1 += input;  // sum
        
        // Update the state
        sum.update(currentSum);
        
        // Emit the average
        if (currentSum.f0 >= 2) {
            out.collect(Tuple2.of(input, (double) currentSum.f1 / currentSum.f0));
            // Clear state on every second element
            sum.clear();
        }
    }
}
```

### Operator State

```java
public class BufferingSink 
    implements SinkFunction<Tuple2<String, Integer>>, CheckpointedFunction {
    
    private final int threshold;
    private transient ListState<Tuple2<String, Integer>> checkpointedState;
    private List<Tuple2<String, Integer>> bufferedElements;
    
    public BufferingSink(int threshold) {
        this.threshold = threshold;
        this.bufferedElements = new ArrayList<>();
    }
    
    @Override
    public void invoke(Tuple2<String, Integer> value, Context context) throws Exception {
        bufferedElements.add(value);
        if (bufferedElements.size() >= threshold) {
            // Send data to external system
            for (Tuple2<String, Integer> element: bufferedElements) {
                // Send to external system
            }
            bufferedElements.clear();
        }
    }
    
    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        checkpointedState.clear();
        for (Tuple2<String, Integer> element : bufferedElements) {
            checkpointedState.add(element);
        }
    }
    
    @Override
    public void initializeState(FunctionInitializationContext context) throws Exception {
        ListStateDescriptor<Tuple2<String, Integer>> descriptor =
            new ListStateDescriptor<>(
                "buffered-elements",
                TypeInformation.of(new TypeHint<Tuple2<String, Integer>>() {}));
        
        checkpointedState = context.getOperatorStateStore().getListState(descriptor);
        
        if (context.isRestored()) {
            for (Tuple2<String, Integer> element : checkpointedState.get()) {
                bufferedElements.add(element);
            }
        }
    }
}
```

## Event Time and Watermarks

### Event Time Processing

```java
// Set up event time processing
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

// Create a watermark strategy
WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    .withTimestampAssigner((event, timestamp) -> event.getTimestamp());

// Apply watermark strategy to source
DataStream<Event> events = env
    .addSource(new CustomSource())
    .assignTimestampsAndWatermarks(watermarkStrategy);
```

### Custom Watermark Generators

```java
public class PunctuatedAssigner implements AssignerWithPunctuatedWatermarks<Event> {
    
    @Override
    public long extractTimestamp(Event element, long previousElementTimestamp) {
        return element.getTimestamp();
    }
    
    @Nullable
    @Override
    public Watermark checkAndGetNextWatermark(
            Event lastElement, 
            long extractedTimestamp) {
        return lastElement.isEndOfSequence() ? new Watermark(extractedTimestamp) : null;
    }
}
```

## Windowing

### Tumbling Windows

```java
DataStream<T> input = ...;

// Tumbling event-time windows
input
    .keyBy(<key selector>)
    .window(TumblingEventTimeWindows.of(Time.seconds(5)))
    .<window function>();

// Tumbling processing-time windows
input
    .keyBy(<key selector>)
    .window(TumblingProcessingTimeWindows.of(Time.seconds(5)))
    .<window function>();
```

### Sliding Windows

```java
// Sliding event-time windows
input
    .keyBy(<key selector>)
    .window(SlidingEventTimeWindows.of(Time.seconds(10), Time.seconds(5)))
    .<window function>();
```

### Session Windows

```java
// Session event-time windows
input
    .keyBy(<key selector>)
    .window(EventTimeSessionWindows.withGap(Time.minutes(5)))
    .<window function>();
```

### Global Windows

```java
input
    .keyBy(<key selector>)
    .window(GlobalWindows.create())
    .trigger(CountTrigger.of(100))
    .evictor(TimeEvictor.of(Time.seconds(10)))
    .<window function>();
```

## Connectors

### Kafka Connector

```java
// Add Maven dependency
// flink-connector-kafka-0.11_2.11

// Create Kafka consumer
Properties properties = new Properties();
properties.setProperty("bootstrap.servers", "localhost:9092");
properties.setProperty("group.id", "test");

FlinkKafkaConsumer011<String> consumer = new FlinkKafkaConsumer011<>(
    "topic",
    new SimpleStringSchema(),
    properties
);

// Set to read from the earliest record
consumer.setStartFromEarliest();

// Add source
DataStream<String> stream = env.addSource(consumer);

// Create Kafka producer
FlinkKafkaProducer011<String> producer = new FlinkKafkaProducer011<>(
    "output-topic",
    new SimpleStringSchema(),
    properties
);

// Add sink
stream.addSink(producer);
```

### File System Connector

```java
// Read from text file
DataStream<String> input = env.readTextFile("file:///path/to/file");

// Write to text file
stream.writeAsText("file:///path/to/output");

// Using FileSink (recommended for newer versions)
final FileSink<String> sink = FileSink
    .forRowFormat(
        new Path("/base/path"),
        new SimpleStringEncoder<String>("UTF-8"))
    .withRollingPolicy(
        DefaultRollingPolicy.builder()
            .withRolloverInterval(TimeUnit.MINUTES.toMillis(15))
            .withInactivityInterval(TimeUnit.MINUTES.toMillis(5))
            .withMaxPartSize(1024 * 1024 * 1024)
            .build())
    .build();

stream.sinkTo(sink);
```

## Scalability and Performance

### Parallelism

```java
// Set default parallelism for all operators
env.setParallelism(4);

// Set parallelism for a specific operator
stream.map(...).setParallelism(4);

// Maximum parallelism (for key groups)
env.setMaxParallelism(128);
```

### Backpressure

Flink handles backpressure automatically by:
1. Slowing down the rate of data consumption from sources
2. Using network buffers to absorb temporary spikes
3. Failing the job if backpressure persists (configurable)

### Memory Tuning

```java
// Set managed memory fraction (default: 0.7)
Configuration config = new Configuration();
config.setString("taskmanager.memory.managed.fraction", "0.5");
StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(config);

// Enable direct memory allocation for network buffers
config.setString("taskmanager.memory.network.fraction", "0.1");
config.setString("taskmanager.memory.network.min", "64mb");
config.setString("taskmanager.memory.network.max", "1gb");
```

## Deployment

### Standalone Cluster

1. **Download Flink**
   ```bash
   wget https://downloads.apache.org/flink/flink-1.16.0/flink-1.16.0-bin-scala_2.12.tgz
   tar -xzf flink-1.16.0-bin-scala_2.12.tgz
   cd flink-1.16.0
   ```

2. **Start Cluster**
   ```bash
   ./bin/start-cluster.sh
   ```

3. **Submit Job**
   ```bash
   ./bin/flink run -c com.example.MyJob /path/to/your/job.jar
   ```

### YARN

```bash
# Start a Flink session on YARN
./bin/yarn-session.sh -n 4 -jm 1024 -tm 4096 -s 2 -qu myqueue

# Submit a job to the session
./bin/flink run -m yarn-cluster -yn 4 -yjm 1024 -ytm 4096 -c com.example.MyJob /path/to/your/job.jar
```

### Kubernetes

```bash
# Deploy Flink session cluster on Kubernetes
kubectl create -f https://raw.githubusercontent.com/apache/flink-kubernetes-operator/release-1.3.0/examples/flink-session-cluster.yaml

# Submit a job
kubectl apply -f job-session.yaml
```

## Monitoring and Metrics

### Web UI

- Accessible at `http://<jobmanager-host>:8081`
- Shows running jobs, task managers, task slots
- Displays metrics and backpressure monitoring
- Provides access to logs and task managers

### Metrics System

```java
// Register a custom metric
getRuntimeContext()
    .getMetricGroup()
    .gauge("MyCustomMetric", new Gauge<Long>() {
        @Override
        public Long getValue() {
            return 42L; // Return your metric value
        }
    });
```

### Logging

```java
// Use SLF4J for logging
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyFunction extends RichFlatMapFunction<String, String> {
    private static final Logger LOG = LoggerFactory.getLogger(MyFunction.class);
    
    @Override
    public void flatMap(String value, Collector<String> out) {
        LOG.info("Processing: {}", value);
        // ...
    }
}
```

## Best Practices

### 1. Use Event Time
```java
// Always prefer event time for accurate results
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
```

### 2. Choose the Right State Backend
```java
// For large state that needs to be checkpointed
env.setStateBackend(new RocksDBStateBackend("hdfs:///checkpoints"));

// For smaller state that fits in memory
env.setStateBackend(new FsStateBackend("hdfs:///checkpoints"));
```

### 3. Tune Checkpointing
```java
// Configure checkpointing
CheckpointConfig config = env.getCheckpointConfig();
config.setCheckpointInterval(60000); // 1 minute
config.setMinPauseBetweenCheckpoints(30000); // 30 seconds
config.setCheckpointTimeout(600000); // 10 minutes
config.setTolerableCheckpointFailureNumber(3);
config.enableExternalizedCheckpoints(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
```

### 4. Handle Backpressure
- Increase parallelism
- Scale up resources (memory, CPU)
- Use async I/O for external lookups
- Tune network and memory settings

### 5. Use Keyed State Over Operator State
```java
// Prefer keyed state when possible
public class KeyedStateFunction extends RichFlatMapFunction<SensorReading, Alert> {
    private transient ValueState<Double> lastTemp;
    
    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Double> descriptor =
            new ValueStateDescriptor<>("lastTemp", Double.class);
        lastTemp = getRuntimeContext().getState(descriptor);
    }
    
    @Override
    public void flatMap(SensorReading reading, Collector<Alert> out) throws Exception {
        Double prevTemp = lastTemp.value();
        lastTemp.update(reading.temperature);
        
        if (prevTemp != null && Math.abs(reading.temperature - prevTemp) > 10.0) {
            out.collect(new Alert(reading.id, "Rapid temperature change"));
        }
    }
}
```

## Case Studies

### 1. Real-time Analytics at Alibaba
- **Use Case**: Real-time monitoring and alerting
- **Scale**: Processes trillions of events per day
- **Key Insight**: Used Flink's event time processing and exactly-once state consistency

### 2. Fraud Detection at Uber
- **Use Case**: Real-time fraud detection for rides and payments
- **Scale**: Millions of events per second
- **Key Insight**: Used Flink's stateful processing and CEP for pattern detection

### 3. Recommendation System at Netflix
- **Use Case**: Real-time content recommendations
- **Scale**: Billions of user interactions per day
- **Key Insight**: Used Flink's windowing and state management for real-time feature computation

## Further Reading

1. [Flink Documentation](https://flink.apache.org/)
2. [Stream Processing with Apache Flink](https://www.oreilly.com/library/view/stream-processing-with/9781491974285/)
3. [Flink Forward Conference Talks](https://www.flink-forward.org/)
4. [Flink Training](https://training.ververica.com/)
5. [Flink GitHub Repository](https://github.com/apache/flink)
