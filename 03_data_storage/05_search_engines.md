# Search Engines and Full-Text Search

## Table of Contents
1. [Introduction to Search Engines](#introduction-to-search-engines)
2. [Core Search Concepts](#core-search-concepts)
3. [Elasticsearch Fundamentals](#elasticsearch-fundamentals)
4. [Indexing and Mapping](#indexing-and-mapping)
5. [Querying Data](#querying-data)
6. [Aggregations and Analytics](#aggregations-and-analytics)
7. [Performance Tuning](#performance-tuning)
8. [Use Cases and Examples](#use-cases-and-examples)
9. [Alternative Search Technologies](#alternative-search-technologies)
10. [Best Practices](#best-practices)

## Introduction to Search Engines

### What is a Search Engine?
A search engine is a software system designed to store, process, and retrieve information based on user queries.

### Key Components
- **Document Store**: Stores the actual content
- **Indexer**: Processes and indexes documents
- **Query Parser**: Interprets search queries
- **Ranking Algorithm**: Orders results by relevance
- **Distributed System**: For scalability and fault tolerance

### Common Use Cases
- Full-text search
- Log and event data analysis
- Autocomplete and search-as-you-type
- Geospatial search
- Recommendation systems

## Core Search Concepts

### Inverted Index
A data structure that maps terms to their locations in documents, enabling fast full-text searches.

### Analysis Process
1. **Character Filtering**: Clean and normalize text
2. **Tokenization**: Break text into terms
3. **Token Filtering**: Modify or remove tokens
4. **Term Storage**: Store in inverted index

### Relevance Scoring
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **BM25**: Improved scoring function
- **Vector Space Model**: Represents documents as vectors
- **Learning to Rank**: Machine learning for ranking

## Elasticsearch Fundamentals

### Key Features
- Distributed and scalable
- Near real-time search
- Multi-tenancy support
- RESTful API
- Schema-free JSON documents

### Basic Concepts
- **Index**: Collection of documents
- **Document**: JSON document with fields
- **Type**: (Deprecated in recent versions)
- **Node**: Single server instance
- **Cluster**: Collection of nodes
- **Shard**: Subdivision of an index
- **Replica**: Copy of a shard for redundancy

### Installation and Setup
```bash
# Download and extract Elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.4.3-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.4.3-linux-x86_64.tar.gz
cd elasticsearch-8.4.3/

# Start a single-node cluster
./bin/elasticsearch

# Verify it's running
curl -X GET "localhost:9200/?pretty"
```

## Indexing and Mapping

### Creating an Index
```http
PUT /products
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "my_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "stemmer"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "my_analyzer" },
      "price": { "type": "float" },
      "description": { "type": "text", "analyzer": "english" },
      "categories": { "type": "keyword" },
      "in_stock": { "type": "boolean" },
      "created_at": { "type": "date" },
      "location": { "type": "geo_point" }
    }
  }
}
```

### Indexing Documents
```http
# Index a single document
POST /products/_doc/1
{
  "name": "Wireless Headphones",
  "price": 99.99,
  "description": "High-quality wireless headphones with noise cancellation",
  "categories": ["electronics", "audio"],
  "in_stock": true,
  "created_at": "2023-01-15T10:00:00Z",
  "location": {
    "lat": 40.7128,
    "lon": -74.0060
  }
}

# Bulk indexing
POST /_bulk
{ "index" : { "_index" : "products", "_id" : "2" } }
{ "name": "Smartphone", "price": 699.99, "in_stock": true }
{ "create" : { "_index" : "products", "_id" : "3" } }
{ "name": "Laptop", "price": 1299.99, "in_stock": false }
```

## Querying Data

### Basic Search
```http
# Match query
GET /products/_search
{
  "query": {
    "match": {
      "name": "wireless headphones"
    }
  },
  "sort": [
    { "price": "asc" }
  ],
  "from": 0,
  "size": 10
}

# Multi-match query
GET /products/_search
{
  "query": {
    "multi_match": {
      "query": "wireless",
      "fields": ["name^2", "description"]
    }
  }
}
```

### Boolean Queries
```http
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "categories": "electronics" }}
      ],
      "must_not": [
        { "match": { "in_stock": false }}
      ],
      "filter": [
        { "range": { "price": { "lte": 1000 } }}
      ]
    }
  }
}
```

### Full-Text Search Features
```http
# Phrase search with slop
GET /products/_search
{
  "query": {
    "match_phrase": {
      "description": {
        "query": "wireless headphones",
        "slop": 2
      }
    }
  }
}

# Fuzzy search
GET /products/_search
{
  "query": {
    "fuzzy": {
      "name": {
        "value": "headfones",
        "fuzziness": "AUTO"
      }
    }
  }
}
```

## Aggregations and Analytics

### Metrics Aggregations
```http
# Basic metrics
GET /products/_search
{
  "size": 0,
  "aggs": {
    "avg_price": { "avg": { "field": "price" } },
    "max_price": { "max": { "field": "price" } },
    "min_price": { "min": { "field": "price" } },
    "total_products": { "value_count": { "field": "_id" } }
  }
}
```

### Bucket Aggregations
```http
# Terms aggregation
GET /products/_search
{
  "size": 0,
  "aggs": {
    "by_category": {
      "terms": {
        "field": "categories",
        "size": 10
      },
      "aggs": {
        "avg_price": { "avg": { "field": "price" } }
      }
    }
  }
}

# Date histogram
GET /products/_search
{
  "size": 0,
  "aggs": {
    "sales_over_time": {
      "date_histogram": {
        "field": "created_at",
        "calendar_interval": "month"
      },
      "aggs": {
        "total_sales": { "sum": { "field": "price" } }
      }
    }
  }
}
```

## Performance Tuning

### Index Optimization
- **Shard Sizing**: 20-40GB per shard
- **Mapping Design**: Use appropriate field types
- **Index Templates**: For consistent index creation
- **Index Aliases**: For zero-downtime reindexing

### Query Optimization
- **Use Filters**: For yes/no conditions
- **Pagination**: Use search_after for deep pagination
- **Field Data vs Doc Values**: Be mindful of memory usage
- **Query Caching**: Leverage the query cache

### Hardware Considerations
- **SSDs**: For better I/O performance
- **Memory**: At least 50% of available memory for JVM heap
- **CPU**: Multiple cores for concurrent queries

## Use Cases and Examples

### 1. E-commerce Search
```http
# Faceted search with filters
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "name": "laptop" }}
      ],
      "filter": [
        { "range": { "price": { "gte": 500, "lte": 1500 } }},
        { "term": { "in_stock": true }}
      ]
    }
  },
  "aggs": {
    "brands": { "terms": { "field": "brand" } },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 1000 },
          { "from": 1000, "to": 2000 },
          { "from": 2000 }
        ]
      }
    },
    "average_rating": { "avg": { "field": "rating" } }
  },
  "sort": [
    { "_score" },
    { "rating": { "order": "desc" } }
  ]
}
```

### 2. Log Analysis
```http
# Log analysis with date ranges
GET /logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "level": "ERROR" }},
        { "range": {
            "@timestamp": {
              "gte": "now-1d/d",
              "lt": "now/d"
            }
        }}
      ]
    }
  },
  "aggs": {
    "errors_over_time": {
      "date_histogram": {
        "field": "@timestamp",
        "fixed_interval": "1h"
      }
    },
    "top_errors": {
      "terms": {
        "field": "message.keyword",
        "size": 5
      }
    }
  }
}
```

### 3. Geospatial Search
```http
# Find products within 10km of a location
GET /products/_search
{
  "query": {
    "bool": {
      "must": {
        "match_all": {}
      },
      "filter": {
        "geo_distance": {
          "distance": "10km",
          "location": {
            "lat": 40.7128,
            "lon": -74.0060
          }
        }
      }
    }
  },
  "sort": [
    {
      "_geo_distance": {
        "location": {
          "lat": 40.7128,
          "lon": -74.0060
        },
        "order": "asc",
        "unit": "km"
      }
    }
  ]
}
```

## Alternative Search Technologies

### 1. Apache Solr
- Built on Apache Lucene
- More traditional search features
- Strong document processing pipeline
- REST-like API

### 2. OpenSearch
- Fork of Elasticsearch
- Open-source and community-driven
- Compatible with most Elasticsearch APIs
- Managed service available on AWS

### 3. MeiliSearch
- Ultra-fast, open-source search engine
- Typo-tolerant
- Easy to use and deploy
- Designed for instant search experiences

### 4. Typesense
- Open-source, typo-tolerant search engine
- Built-in ranking and relevance
- RESTful API
- Focus on developer experience

## Best Practices

### Index Design
- Use appropriate mappings for fields
- Leverage dynamic templates
- Use aliases for flexible index management
- Plan for index lifecycle management

### Query Design
- Use filters for yes/no conditions
- Be specific with field selection
- Use pagination for large result sets
- Profile queries to understand performance

### Performance
- Monitor cluster health and performance
- Use bulk API for indexing multiple documents
- Tune refresh interval based on use case
- Consider using index templates and ILM policies

### Security
- Enable authentication and authorization
- Use TLS for communication
- Implement field-level security
- Regular backups and snapshots

## Practice Exercises
1. Set up a local Elasticsearch instance and index sample product data
2. Create mappings for different data types (text, keyword, date, geo_point, etc.)
3. Implement a search page with filters and facets
4. Build an autocomplete feature using completion suggesters
5. Create a geospatial search for finding nearby locations
6. Optimize a slow-running query using the Profile API
7. Set up index templates and ILM policies
8. Implement a simple search application using a client library

---
Next: [Data Lake Architecture](./06_data_lake_architecture.md)
