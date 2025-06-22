# Data Lake Architecture

## Table of Contents
1. [Introduction to Data Lakes](#introduction-to-data-lakes)
2. [Data Lake Architecture Components](#data-lake-architecture-components)
3. [Storage Layer](#storage-layer)
4. [Ingestion Layer](#ingestion-layer)
5. [Processing Layer](#processing-layer)
6. [Serving Layer](#serving-layer)
7. [Metadata Management](#metadata-management)
8. [Security and Governance](#security-and-governance)
9. [Implementation Patterns](#implementation-patterns)
10. [Best Practices](#best-practices)

## Introduction to Data Lakes

### What is a Data Lake?
A data lake is a centralized repository that allows you to store all your structured and unstructured data at any scale.

### Key Characteristics
- **Schema-on-read**: No need to define schema before loading data
- **Store now, analyze later**: Collect data first, determine usage later
- **Support for all data types**: Structured, semi-structured, unstructured
- **Scalable storage**: Petabyte-scale storage capabilities
- **Multiple processing engines**: Support for various analytics tools

### Benefits
- Centralized data repository
- Cost-effective storage
- Support for advanced analytics and machine learning
- Faster data onboarding
- Preservation of raw data

## Data Lake Architecture Components

### Core Layers
1. **Storage Layer**: Raw data storage (S3, ADLS, HDFS)
2. **Ingestion Layer**: Data collection and import tools
3. **Processing Layer**: Data transformation and preparation
4. **Serving Layer**: Data access and analytics
5. **Metadata Layer**: Data catalog and governance
6. **Security Layer**: Access control and encryption

### Reference Architecture
```
┌───────────────────────────────────────────────────────────┐
│                   Analytics & Visualization               │
└───────────────┬─────────────────────────────┬───────────┘
                │                             │
┌───────────────▼─────────────┐ ┌───────────▼───────────┐
│        Serving Layer        │ │   Processing Layer    │
│  - SQL Query Engines       │ │  - ETL/ELT Pipelines  │
│  - Data Warehouses        │ │  - Data Preparation   │
│  - Machine Learning       │ │  - Data Quality       │
└───────────────┬─────────────┘ └───────────┬───────────┘
                │                             │
┌───────────────▼─────────────────────────────▼───────────┐
│                   Storage Layer                         │
│  - Raw Zone (Landing)                                 │
│  - Processed Zone (Cleaned, Transformed)              │
│  - Curated Zone (Business-ready)                      │
└───────────────┬─────────────────────────────┬───────────┘
                │                             │
┌───────────────▼─────────────┐ ┌───────────▼───────────┐
│      Ingestion Layer        │ │    Metadata &         │
│  - Batch Ingestion         │ │    Governance         │
│  - Stream Ingestion        │ │  - Data Catalog       │
│  - CDC                     │ │  - Lineage            │
└─────────────────────────────┘ └───────────────────────┘
```

## Storage Layer

### Storage Zones
1. **Landing Zone (Raw/Bronze)**
   - Raw, unprocessed data
   - Maintains original fidelity
   - Immutable storage

2. **Processing Zone (Cleaned/Silver)**
   - Cleaned and validated data
   - Standardized formats
   - May include some transformations

3. **Curated Zone (Gold)**
   - Business-ready data
   - Aggregated and enriched
   - Optimized for consumption

### Storage Formats
- **File Formats**: Parquet, ORC, Avro, JSON, CSV
- **Table Formats**: Delta Lake, Iceberg, Hudi
- **Storage Systems**: S3, ADLS, GCS, HDFS

### Example: S3 Bucket Structure
```
s3://data-lake/
├── raw/
│   ├── sales/
│   │   ├── year=2023/
│   │   │   ├── month=01/
│   │   │   └── month=02/
│   │   └── year=2024/
│   └── customers/
│       └── ingest_date=2024-01-15/
├── processed/
│   ├── sales/
│   └── customers/
└── curated/
    ├── customer_360/
    └── sales_analytics/
```

## Ingestion Layer

### Ingestion Patterns
1. **Batch Ingestion**
   - Scheduled data loads
   - Tools: AWS Glue, Azure Data Factory, Apache NiFi

2. **Stream Ingestion**
   - Real-time data streaming
   - Tools: Apache Kafka, AWS Kinesis, Azure Event Hubs

3. **Change Data Capture (CDC)**
   - Captures database changes
   - Tools: Debezium, AWS DMS, Qlik Replicate

### Example: AWS Glue Job for Batch Ingestion
```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Read from source (S3)
datasource0 = glueContext.create_dynamic_frame.from_catalog(
    database = "sales_raw",
    table_name = "transactions",
    transformation_ctx = "datasource0"
)

# Basic transformations
transformed = ApplyMapping.apply(
    frame = datasource0,
    mappings = [
        ("transaction_id", "string", "transaction_id", "string"),
        ("customer_id", "string", "customer_id", "string"),
        ("amount", "decimal", "amount", "decimal(18,2)"),
        ("transaction_date", "string", "transaction_date", "timestamp")
    ],
    transformation_ctx = "transformed"
)

# Write to processed zone
datasink = glueContext.write_dynamic_frame.from_options(
    frame = transformed,
    connection_type = "s3",
    connection_options = {
        "path": "s3://data-lake/processed/sales/",
        "partitionKeys": ["year", "month", "day"]
    },
    format = "parquet",
    transformation_ctx = "datasink"
)

job.commit()
```

## Processing Layer

### Processing Frameworks
1. **Batch Processing**
   - Apache Spark
   - Apache Hive
   - Presto/Trino

2. **Stream Processing**
   - Apache Flink
   - Apache Spark Streaming
   - Kafka Streams

3. **Serverless**
   - AWS Lambda
   - Azure Functions
   - Google Cloud Functions

### Example: Spark Processing Job
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SalesProcessing") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .getOrCreate()

# Read from processed zone
df = spark.read.parquet("s3://data-lake/processed/sales/")


# Transformations
df_transformed = df \
    .withColumn("year", year(col("transaction_date"))) \
    .withColumn("month", month(col("transaction_date"))) \
    .withColumn("day", dayofmonth(col("transaction_date"))) \
    .groupBy("customer_id", "year", "month") \
    .agg(
        count("*").alias("transaction_count"),
        sum("amount").alias("total_spent"),
        avg("amount").alias("avg_transaction")
    )

# Write to curated zone
df_transformed.write \
    .partitionBy("year", "month") \
    .mode("overwrite") \
    .parquet("s3://data-lake/curated/customer_metrics/")
```

## Serving Layer

### Query Engines
1. **Interactive SQL**
   - Amazon Athena
   - Google BigQuery
   - Presto/Trino

2. **Data Warehouses**
   - Snowflake
   - Amazon Redshift
   - Google BigQuery

3. **Machine Learning**
   - Amazon SageMaker
   - Databricks
   - Google AI Platform

### Example: Creating a View with Athena
```sql
-- Create a database
CREATE DATABASE IF NOT EXISTS curated_data;

-- Create an external table
CREATE EXTERNAL TABLE IF NOT EXISTS curated_data.customer_metrics (
    customer_id STRING,
    transaction_count BIGINT,
    total_spent DECIMAL(18,2),
    avg_transaction DECIMAL(18,2)
)
PARTITIONED BY (year STRING, month STRING)
STORED AS PARQUET
LOCATION 's3://data-lake/curated/customer_metrics/';

-- Load partitions
MSCK REPAIR TABLE curated_data.customer_metrics;

-- Query the data
SELECT 
    year,
    month,
    COUNT(DISTINCT customer_id) AS active_customers,
    SUM(total_spent) AS monthly_revenue,
    AVG(avg_transaction) AS avg_transaction_value
FROM curated_data.customer_metrics
GROUP BY year, month
ORDER BY year, month;
```

## Metadata Management

### Metadata Categories
1. **Technical Metadata**
   - File formats, schemas
   - Data types, partitions
   - Lineage and dependencies

2. **Business Metadata**
   - Business definitions
   - Data owners
   - Data quality metrics

3. **Operational Metadata**
   - Data freshness
   - Processing metrics
   - Access patterns

### Data Catalogs
- **AWS Glue Data Catalog**
- **Apache Atlas**
- **OpenMetadata**
- **Amundsen**
- **DataHub**

### Example: AWS Glue Crawler
```python
import boto3

# Initialize Glue client
glue = boto3.client('glue')

# Create a crawler to catalog the curated zone
response = glue.create_crawler(
    Name='customer_metrics_crawler',
    Role='arn:aws:iam::123456789012:role/GlueServiceRole',
    DatabaseName='curated_data',
    Targets={
        'S3Targets': [
            {
                'Path': 's3://data-lake/curated/customer_metrics/',
                'Exclusions': []
            }
        ]
    },
    TablePrefix='cm_',
    SchemaChangePolicy={
        'UpdateBehavior': 'UPDATE_IN_DATABASE',
        'DeleteBehavior': 'DEPRECATE_IN_DATABASE'
    },
    Configuration='{"Version":1.0,"CrawlerOutput":{"Partitions":{"AddOrUpdateBehavior":"InheritFromTable"}}}'
)

# Start the crawler
glue.start_crawler(Name='customer_metrics_crawler')
```

## Security and Governance

### Security Controls
1. **Access Control**
   - IAM roles and policies
   - Attribute-based access control (ABAC)
   - Row/column level security

2. **Encryption**
   - Encryption at rest
   - Encryption in transit
   - Client-side encryption

3. **Audit Logging**
   - Access logs
   - Data modification logs
   - Query history

### Data Governance
- **Data Quality**
- **Data Lineage**
- **Data Classification**
- **Compliance** (GDPR, CCPA, HIPAA)

### Example: S3 Bucket Policy
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyUnencryptedUploads",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::data-lake/*",
            "Condition": {
                "StringNotEquals": {
                    "s3:x-amz-server-side-encryption": "AES256"
                }
            }
        },
        {
            "Sid": "EnforceTLS",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::data-lake/*",
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }
    ]
}
```

## Implementation Patterns

### 1. Data Lake Zones
- **Raw Zone**: Landing area for all raw data
- **Cleansed Zone**: Validated and standardized data
- **Curated Zone**: Business-ready data products
- **Analytics Sandbox**: For exploration and experimentation

### 2. Medallion Architecture
- **Bronze**: Raw data
- **Silver**: Validated and enriched data
- **Gold**: Business-level aggregates and metrics

### 3. Data Mesh
- Domain-oriented data ownership
- Data as a product
- Self-serve data infrastructure
- Federated computational governance

### 4. Lakehouse Architecture
- Combines data lake and data warehouse
- ACID transactions
- Schema enforcement and governance
- Support for diverse workloads

## Best Practices

### 1. Data Organization
- Use consistent naming conventions
- Implement a clear folder structure
- Use partitioning for large datasets
- Apply appropriate file formats

### 2. Performance Optimization
- Choose the right file format (Parquet, ORC)
- Optimize file sizes (100MB-1GB)
- Use partitioning and bucketing
- Consider data compaction

### 3. Cost Management
- Implement data lifecycle policies
- Use storage classes (S3 Intelligent-Tiering)
- Monitor and optimize queries
- Clean up temporary data

### 4. Data Quality
- Implement data validation checks
- Monitor data quality metrics
- Document data quality expectations
- Set up alerts for data quality issues

### 5. Documentation
- Document data sources and schemas
- Maintain a data dictionary
- Document ETL processes
- Keep runbooks for operations

## Practice Exercises
1. Design a data lake architecture for an e-commerce company
2. Set up an S3 bucket structure with appropriate IAM policies
3. Create an AWS Glue crawler to catalog your data
4. Write a Spark job to process and transform raw data
5. Set up a data quality monitoring framework
6. Implement a data catalog using AWS Glue or OpenMetadata
7. Create a data access pattern for different user roles
8. Design a data retention and archival strategy

---
Next: [Data Pipeline Orchestration](../04_data_pipelines/01_introduction_to_data_pipelines.md)
