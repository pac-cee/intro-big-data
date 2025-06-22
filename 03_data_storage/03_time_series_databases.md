# Time-Series Databases

## Table of Contents
1. [Introduction to Time-Series Data](#introduction-to-time-series-data)
2. [Time-Series Database Concepts](#time-series-database-concepts)
3. [Popular Time-Series Databases](#popular-time-series-databases)
4. [Data Modeling for Time-Series](#data-modeling-for-time-series)
5. [Querying Time-Series Data](#querying-time-series-data)
6. [Downsampling and Retention](#downsampling-and-retention)
7. [Performance Optimization](#performance-optimization)
8. [Use Cases and Examples](#use-cases-and-examples)
9. [Integration with Other Systems](#integration-with-other-systems)
10. [Best Practices](#best-practices)

## Introduction to Time-Series Data

### What is Time-Series Data?
Time-series data is a sequence of data points collected at regular time intervals.

### Characteristics
- **Time-stamped**: Each data point is associated with a timestamp
- **Append-heavy**: New data is typically appended, not updated
- **Time-ordered**: Data is always processed in time order
- **High volume**: Can generate millions of data points per second

### Common Sources
- IoT devices and sensors
- Application metrics and monitoring
- Financial market data
- Industrial telemetry
- System and network monitoring

## Time-Series Database Concepts

### Key Features
- **Efficient storage**: Optimized for time-ordered data
- **Fast writes**: High ingestion rates
- **Time-based queries**: Specialized for time-range queries
- **Downsampling**: Aggregate data over time periods
- **Retention policies**: Automatically expire old data

### Data Structure
```
+---------------------+--------+-------+
|      timestamp      | metric | value |
+---------------------+--------+-------+
| 2023-01-01 00:00:00 | temp   |  22.5 |
| 2023-01-01 00:01:00 | temp   |  22.6 |
| 2023-01-01 00:00:00 | power  | 120.3 |
| 2023-01-01 00:01:00 | power  | 121.0 |
+---------------------+--------+-------+
```

### Time-Series vs Traditional Databases
| Feature          | Time-Series DB | Traditional RDBMS |
|-----------------|----------------|-------------------|
| Write Throughput | Very High      | Moderate          |
| Schema          | Flexible       | Rigid             |
| Time Queries    | Optimized      | Not optimized     |
| Data Retention  | Built-in       | Manual            |
| Compression     | High           | Moderate          |

## Popular Time-Series Databases

### 1. InfluxDB
- Open-source time-series database
- SQL-like query language (Flux)
- High write and query performance
- Built-in visualization tools

### 2. TimescaleDB
- PostgreSQL extension
- Full SQL support
- Time-based partitioning
- Continuous aggregates

### 3. Prometheus
- Monitoring and alerting toolkit
- Pull-based metrics collection
- PromQL query language
- Strong ecosystem (Grafana, Alertmanager)

### 4. OpenTSDB
- Built on HBase
- Scalable for large deployments
- Supports downsampling and pre-aggregation

### 5. ClickHouse
- Column-oriented database
- Extremely fast for analytical queries
- Supports time-series use cases

## Data Modeling for Time-Series

### Basic Concepts
- **Measurement/Table**: Container for time-series data (e.g., "cpu_usage")
- **Tags**: Indexed metadata (e.g., host, region)
- **Fields**: Actual values being measured (e.g., value=75.3)
- **Timestamp**: When the data point was recorded

### Schema Design
```typescript
// Example: Server metrics
{
  measurement: "server_metrics",
  tags: {
    host: "web-01",
    region: "us-west-2",
    app: "api-service"
  },
  fields: {
    cpu_usage: 45.2,
    memory_used: 1234567890,
    disk_io: 1200
  },
  timestamp: 1672531200000
}
```

### Best Practices
1. **Use tags for filtering and grouping**
2. **Keep tag cardinality low** (avoid high-cardinality tags)
3. **Batch writes** for better performance
4. **Use appropriate timestamp precision**
5. **Consider downsampling** for long-term storage

## Querying Time-Series Data

### InfluxDB Query Language (Flux)
```sql
from(bucket: "telegraf")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "cpu")
  |> filter(fn: (r) => r["_field"] == "usage_idle")
  |> aggregateWindow(every: 1m, fn: mean)
  |> yield(name: "mean")
```

### TimescaleDB (PostgreSQL)
```sql
-- Time bucket aggregation
SELECT 
  time_bucket('1 hour', time) AS hour,
  device_id,
  AVG(temperature) as avg_temp
FROM sensor_data
WHERE time > NOW() - INTERVAL '1 day'
GROUP BY hour, device_id
ORDER BY hour;

-- Last point per series
SELECT DISTINCT ON (device_id) *
FROM sensor_data
ORDER BY device_id, time DESC;
```

### PromQL (Prometheus)
```promql
# CPU usage rate
rate(node_cpu_seconds_total{mode!="idle"}[5m])

# Memory usage percentage
100 - (node_memory_MemAvailable_bytes * 100) / node_memory_MemTotal_bytes

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) 
/ 
sum(rate(http_requests_total[5m]))
```

## Downsampling and Retention

### Why Downsample?
- Reduce storage requirements
- Improve query performance
- Maintain historical trends while discarding fine-grained data

### Implementation Examples

#### InfluxDB Retention Policy
```sql
-- Create a retention policy
CREATE RETENTION POLICY "one_year" 
ON "mydb" 
DURATION 365d 
REPLICATION 1;

-- Create continuous query for downsampling
CREATE CONTINUOUS QUERY "downsample_1h"
ON "mydb"
BEGIN
  SELECT mean(*) INTO "one_year".:MEASUREMENT
  FROM "autogen"./.*/
  GROUP BY time(1h), *
END
```

#### TimescaleDB Continuous Aggregates
```sql
-- Create a continuous aggregate
CREATE MATERIALIZED VIEW sensor_data_1h
WITH (timescaledb.continuous) AS
  SELECT 
    time_bucket('1 hour', time) AS bucket,
    device_id,
    AVG(temperature) as avg_temp
  FROM sensor_data
  GROUP BY bucket, device_id;

-- Add a refresh policy
SELECT add_continuous_aggregate_policy('sensor_data_1h',
  start_offset => INTERVAL '1 month',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour');
```

## Performance Optimization

### Storage Optimization
1. **Compression**: Most time-series databases automatically compress data
2. **Partitioning**: Split data by time ranges
3. **Columnar storage**: Store values for the same metric together

### Query Optimization
1. **Time-based indexing**: Ensure queries use time ranges
2. **Avoid SELECT *** : Only query needed fields
3. **Use appropriate time ranges**: Don't fetch more data than needed
4. **Pre-aggregation**: Store common aggregations

### Hardware Considerations
- **SSDs**: For write-heavy workloads
- **Memory**: For query performance
- **CPU**: For compression and query processing

## Use Cases and Examples

### 1. IoT Monitoring
```python
# Example: Writing sensor data to InfluxDB
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(url="http://localhost:8086", token="my-token")
write_api = client.write_api(write_options=SYNCHRONOUS)

point = Point("sensor") \
    .tag("device_id", "sensor-001") \
    .tag("location", "building-a") \
    .field("temperature", 23.5) \
    .field("humidity", 45.2)

write_api.write(bucket="iot", record=point)
```

### 2. Application Metrics
```python
# Example: Using Prometheus client
from prometheus_client import start_http_server, Counter, Histogram
import random
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Request latency')

# Simulate requests
@REQUEST_LATENCY.time()
def handle_request():
    time.sleep(random.uniform(0, 1))
    REQUEST_COUNT.inc()

# Start metrics server
start_http_server(8000)


# Simulate traffic
while True:
    handle_request()
```

### 3. Financial Data
```sql
-- Store and analyze stock prices in TimescaleDB
CREATE TABLE stock_quotes (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    volume BIGINT NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('stock_quotes', 'time');

-- Query for moving average
SELECT 
    time_bucket('1 day', time) AS day,
    symbol,
    AVG(price) OVER (PARTITION BY symbol ORDER BY time 
                     RANGE BETWEEN INTERVAL '5 days' PRECEDING 
                     AND CURRENT ROW) AS moving_avg
FROM stock_quotes
WHERE symbol = 'AAPL';
```

## Integration with Other Systems

### Data Collection
- **Telegraf**: Plugin-driven server agent for collecting metrics
- **Prometheus**: Pull-based monitoring system
- **Fluentd/Fluent Bit**: Log and event collection

### Visualization
- **Grafana**: Popular dashboarding tool
- **Chronograf**: Built-in UI for InfluxDB
- **Superset**: Open-source data exploration

### Processing
- **Kafka**: Stream processing integration
- **Spark/Flink**: For complex event processing
- **Airflow**: Orchestrating ETL pipelines

## Best Practices

### Schema Design
- **Use meaningful measurement/table names**
- **Tag values should have low cardinality**
- **Fields should contain the actual metrics**
- **Be consistent with naming conventions**

### Data Management
- **Set appropriate retention policies**
- **Implement downsampling for long-term storage**
- **Monitor disk usage and performance**
- **Regularly back up your data**

### Performance Tuning
- **Batch writes** (1000-5000 points per batch)
- **Use appropriate time precision** (ms, μs, ns)
- **Optimize chunk size** for time-based partitioning
- **Monitor and tune memory settings**

### Monitoring
- **Track ingestion rates**
- **Monitor query performance**
- **Set up alerts** for system health
- **Regularly review and optimize queries**

## Practice Exercises
1. Set up InfluxDB and create a database for IoT sensor data
2. Design a schema for monitoring web application performance metrics
3. Write a script to generate and insert synthetic time-series data
4. Create continuous queries for downsampling data
5. Build a Grafana dashboard to visualize your time-series data
6. Optimize a slow-running time-series query
7. Implement a retention policy for your time-series data
8. Set up alerts for anomaly detection in your time-series data

---
Next: [Graph Databases](./04_graph_databases.md)
