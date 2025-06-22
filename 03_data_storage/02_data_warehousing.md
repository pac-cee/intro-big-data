# Data Warehousing

## Table of Contents
1. [Introduction to Data Warehousing](#introduction-to-data-warehousing)
2. [Data Warehouse Architecture](#data-warehouse-architecture)
3. [Dimensional Modeling](#dimensional-modeling)
4. [ETL vs ELT](#etl-vs-elt)
5. [Data Warehouse Technologies](#data-warehouse-technologies)
6. [Data Lakes vs Data Warehouses](#data-lakes-vs-data-warehouses)
7. [Working with Cloud Data Warehouses](#working-with-cloud-data-warehouses)
8. [Performance Optimization](#performance-optimization)
9. [Data Governance and Security](#data-governance-and-security)
10. [Emerging Trends](#emerging-trends)

## Introduction to Data Warehousing

### What is a Data Warehouse?
A data warehouse is a centralized repository of integrated data from one or more disparate sources designed for query and analysis.

### Key Characteristics
- **Subject-Oriented**: Organized around major subjects (e.g., customers, products, sales)
- **Integrated**: Consistent naming conventions, formats, and encoding
- **Time-Variant**: Maintains historical data
- **Non-Volatile**: Data is read-only and retained for future analysis

### Benefits
- Improved business intelligence
- Historical data analysis
- Data consistency and quality
- High query performance for analytical workloads

## Data Warehouse Architecture

### Three-Tier Architecture
1. **Bottom Tier (Data Sources)**
   - Operational databases
   - Flat files
   - External data sources

2. **Middle Tier (ETL/ELT)**
   - Data extraction, transformation, and loading
   - Data cleaning and integration
   - Staging area

3. **Top Tier (Front-End)**
   - Reporting tools
   - Analytics tools
   - Data mining tools
   - OLAP tools

### Data Warehouse Models
- **Enterprise Data Warehouse (EDW)**
- **Operational Data Store (ODS)**
- **Data Marts** (Departmental subsets)
- **Data Lakehouse** (Combines data lake and data warehouse)

## Dimensional Modeling

### Fact Tables
Contain quantitative data (measures) about business processes.

```sql
CREATE TABLE fact_sales (
    sale_id INT PRIMARY KEY,
    date_id INT,
    product_id INT,
    customer_id INT,
    store_id INT,
    quantity_sold INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(12,2),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (store_id) REFERENCES dim_store(store_id)
);
```

### Dimension Tables
Contain descriptive attributes related to fact data.

```sql
CREATE TABLE dim_customer (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    customer_since DATE,
    loyalty_tier VARCHAR(20),
    start_date DATE,
    end_date DATE,
    current_flag BOOLEAN
);
```

### Schema Types
1. **Star Schema**
   - Single fact table connected to multiple dimension tables
   - Simple and fast for queries

2. **Snowflake Schema**
   - Normalized dimension tables
   - Reduces data redundancy
   - More complex queries

3. **Galaxy Schema (Fact Constellation)**
   - Multiple fact tables share dimension tables
   - For complex analytical needs

## ETL vs ELT

### ETL (Extract, Transform, Load)
1. **Extract**: Data is collected from source systems
2. **Transform**: Data is processed and transformed in a staging area
3. **Load**: Transformed data is loaded into the data warehouse

**Best for**:
- Complex transformations
- When source systems have limited processing power
- When data quality is poor

### ELT (Extract, Load, Transform)
1. **Extract**: Data is collected from source systems
2. **Load**: Raw data is loaded into the data warehouse
3. **Transform**: Data is transformed within the data warehouse

**Best for**:
- Large volumes of data
- When transformations can leverage warehouse processing power
- When quick data availability is important

### ETL/ELT Tools
- **Open Source**: Apache NiFi, Talend Open Studio, Airflow
- **Commercial**: Informatica, Microsoft SSIS, Matillion
- **Cloud-Native**: AWS Glue, Google Dataflow, Azure Data Factory

## Data Warehouse Technologies

### On-Premises Solutions
1. **Teradata**
   - Enterprise data warehouse platform
   - MPP (Massively Parallel Processing) architecture
   - Strong in mixed workloads

2. **Oracle Exadata**
   - Engineered system for data warehousing
   - In-memory processing
   - High-performance storage

3. **IBM Db2 Warehouse**
   - Columnar storage
   - Advanced compression
   - In-database analytics

### Cloud Data Warehouses
1. **Snowflake**
   - Cloud-native data warehouse
   - Separates compute and storage
   - Support for semi-structured data

2. **Amazon Redshift**
   - Cloud data warehouse
   - Columnar storage
   - Integration with AWS ecosystem

3. **Google BigQuery**
   - Serverless data warehouse
   - Built-in machine learning
   - Strong SQL support

4. **Microsoft Azure Synapse Analytics**
   - Integrated analytics service
   - Combines data integration, warehousing, and big data analytics

## Data Lakes vs Data Warehouses

| Feature          | Data Lake | Data Warehouse |
|-----------------|-----------|----------------|
| Data Type       | Raw, all types | Processed, structured |
| Schema          | Schema-on-read | Schema-on-write |
| Users           | Data scientists, engineers | Business analysts |
| Processing      | Batch, real-time | Primarily batch |
| Cost            | Lower storage cost | Higher processing cost |
| Data Quality    | Variable | High |
| Purpose         | Store everything | Analyze structured data |

### Lakehouse Architecture
Combines the best of data lakes and data warehouses:
- Open storage formats (Delta Lake, Iceberg, Hudi)
- ACID transactions
- Schema enforcement and governance
- Support for diverse workloads

## Working with Cloud Data Warehouses

### Snowflake Example
```sql
-- Create a warehouse
CREATE WAREHOUSE analytics_wh
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE;

-- Create a database
CREATE DATABASE sales_db;

-- Create a schema
CREATE SCHEMA ecommerce;

-- Create a table
CREATE TABLE sales_db.ecommerce.transactions (
    transaction_id STRING,
    customer_id STRING,
    product_id STRING,
    quantity INT,
    amount FLOAT,
    transaction_date TIMESTAMP_NTZ
);

-- Load data from S3
COPY INTO sales_db.ecommerce.transactions
FROM 's3://bucket/path/transactions/'
CREDENTIALS = (AWS_KEY_ID = '...' AWS_SECRET_KEY = '...')
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);

-- Query data
SELECT 
    DATE_TRUNC('month', transaction_date) AS month,
    SUM(amount) AS total_sales,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM sales_db.ecommerce.transactions
GROUP BY 1
ORDER BY 1;
```

### BigQuery Example
```sql
-- Create a dataset
CREATE SCHEMA `project_id.analytics`;

-- Create a table
CREATE TABLE `project_id.analytics.user_activity` (
  user_id STRING,
  event_name STRING,
  event_timestamp TIMESTAMP,
  device_type STRING,
  country STRING
);

-- Load data from Cloud Storage
LOAD DATA OVERWRITE `project_id.analytics.user_activity`
FROM FILES(
  format = 'CSV',
  uris = ['gs://bucket/path/user_activity/*.csv']
);

-- Analyze data with BigQuery ML
CREATE OR REPLACE MODEL `analytics.user_segments`
OPTIONS(
  model_type='kmeans',
  num_clusters=5
) AS
SELECT
  user_id,
  COUNT(*) AS session_count,
  AVG(session_duration) AS avg_session_duration,
  COUNT(DISTINCT page_path) AS unique_pages_visited
FROM `analytics.sessions`
GROUP BY user_id;
```

## Performance Optimization

### Partitioning
```sql
-- Create a partitioned table (BigQuery)
CREATE TABLE `project.dataset.sales_partitioned`
PARTITION BY DATE(transaction_date)
AS SELECT * FROM `project.dataset.sales`;

-- Create a partitioned table (Snowflake)
CREATE TABLE sales_partitioned (
    id INT,
    sale_date DATE,
    amount DECIMAL(10,2)
) 
CLUSTER BY (sale_date);
```

### Clustering
```sql
-- Create a clustered columnstore index (SQL Server)
CREATE CLUSTERED COLUMNSTORE INDEX CCI_Sales 
ON Sales.SalesOrderDetail;

-- Cluster by columns (BigQuery)
CREATE TABLE `project.dataset.sales_clustered`
PARTITION BY DATE(transaction_date)
CLUSTER BY customer_id, product_category
AS SELECT * FROM `project.dataset.sales`;
```

### Materialized Views
```sql
-- Create a materialized view (Snowflake)
CREATE MATERIALIZED VIEW monthly_sales_mv
AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    region,
    SUM(amount) AS total_sales,
    COUNT(DISTINCT customer_id) AS customers
FROM sales
GROUP BY 1, 2;

-- Create a materialized view (BigQuery)
CREATE MATERIALIZED VIEW `project.dataset.monthly_sales`
AS
SELECT
    DATE_TRUNC(date, MONTH) AS month,
    product_category,
    SUM(sales_amount) AS total_sales
FROM `project.dataset.sales`
GROUP BY 1, 2;
```

## Data Governance and Security

### Data Quality
- Data profiling
- Data validation rules
- Data cleansing
- Master data management (MDM)

### Security Measures
- Role-based access control (RBAC)
- Column-level security
- Row-level security
- Data masking
- Encryption at rest and in transit

### Compliance
- GDPR
- CCPA
- HIPAA
- SOX

### Metadata Management
- Data lineage
- Data catalog
- Business glossary
- Impact analysis

## Emerging Trends

### Data Mesh
- Domain-oriented data ownership
- Data as a product
- Self-serve data infrastructure
- Federated computational governance

### Real-time Analytics
- Streaming data integration
- Real-time dashboards
- Event-driven architectures
- Complex event processing (CEP)

### AI/ML Integration
- In-database machine learning
- Automated data preparation
- AI-powered data quality
- Natural language querying

### Multi-Cloud and Hybrid
- Cross-cloud data integration
- Data lakehouse architectures
- Edge computing integration
- Unified data catalogs

## Practice Exercises
1. Design a star schema for an e-commerce data warehouse
2. Create and optimize a partitioned table in your preferred cloud data warehouse
3. Implement an ETL pipeline to load data from a source to a data warehouse
4. Set up row-level security for a sales data table
5. Design a data mesh architecture for a large organization

---
Next: [Time-Series Databases](./03_time_series_databases.md)
