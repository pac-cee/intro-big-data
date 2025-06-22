# Graph Databases

## Table of Contents
1. [Introduction to Graph Databases](#introduction-to-graph-databases)
2. [Graph Data Model](#graph-data-model)
3. [Querying Graph Data](#querying-graph-data)
4. [Popular Graph Databases](#popular-graph-databases)
5. [Data Modeling for Graphs](#data-modeling-for-graphs)
6. [Performance Considerations](#performance-considerations)
7. [Use Cases and Examples](#use-cases-and-examples)
8. [Graph Algorithms](#graph-algorithms)
9. [Integration with Other Systems](#integration-with-other-systems)
10. [Best Practices](#best-practices)

## Introduction to Graph Databases

### What is a Graph Database?
A graph database is a database designed to treat the relationships between data as equally important to the data itself.

### Key Characteristics
- **Nodes**: Represent entities (people, products, etc.)
- **Edges/Relationships**: Connect nodes with specific types and directions
- **Properties**: Key-value pairs on both nodes and relationships
- **Schema-optional**: Can be schema-less or schema-flexible

### Benefits
- Efficient relationship traversal
- Intuitive data modeling for connected data
- Better performance for complex relationship queries
- Flexible schema evolution

## Graph Data Model

### Core Components
1. **Nodes (Vertices)**
   - Represent entities (e.g., Person, Product, City)
   - Can have labels (types) and properties

2. **Relationships (Edges)**
   - Connect two nodes with a direction
   - Have a type (e.g., KNOWS, PURCHASED)
   - Can have properties

3. **Properties**
   - Key-value pairs on both nodes and relationships
   - Can be of different data types

### Example Data Model
```
(Alice:Person {name: 'Alice', age: 30})
  -[:KNOWS {since: 2015}]-> (Bob:Person {name: 'Bob', age: 35})
  -[:WORKS_AT {position: 'Engineer'}]-> (Acme:Company {name: 'Acme Inc'})
```

## Querying Graph Data

### Cypher (Neo4j)
```cypher
// Find all of Alice's friends
MATCH (alice:Person {name: 'Alice'})-[:FRIEND]->(friend:Person)
RETURN friend.name, friend.age

// Find friends of friends who aren't already friends
MATCH (alice:Person {name: 'Alice'})-[:FRIEND]->(friend:Person)-[:FRIEND]->(fof:Person)
WHERE NOT (alice)-[:FRIEND]->(fof)
RETURN fof.name

// Find the shortest path between two people
MATCH path = shortestPath((alice:Person {name: 'Alice'})-[*]-(bob:Person {name: 'Bob'}))
RETURN path
```

### Gremlin (Apache TinkerPop)
```groovy
// Find all of Alice's friends
g.V().has('Person', 'name', 'Alice').out('FRIEND')

// Find friends of friends who aren't already friends
g.V().has('Person', 'name', 'Alice')
  .repeat(out('FRIEND')).times(2)
  .where(without('x')).by('name')
  .where(neq('Alice'))

// Find the shortest path between two people
g.V().has('Person', 'name', 'Alice')
  .repeat(both().simplePath())
  .until(has('name', 'Bob'))
  .path()
  .limit(1)
```

## Popular Graph Databases

### 1. Neo4j
- Most popular native graph database
- ACID compliant
- Cypher query language
- Enterprise and community editions

### 2. Amazon Neptune
- Fully managed graph database service
- Supports both Property Graph and RDF/SPARQL
- Highly available and durable

### 3. JanusGraph
- Scalable graph database
- Supports various storage backends (Cassandra, HBase, etc.)
- Supports Gremlin query language

### 4. ArangoDB
- Multi-model database (documents, graphs, key-value)
- AQL (ArangoDB Query Language)
- Single instance or distributed

### 5. TigerGraph
- Native parallel graph database
- Supports real-time deep link analytics
- GSQL query language

## Data Modeling for Graphs

### Modeling Approaches
1. **Labeled Property Graph (LPG)**
   - Nodes with labels and properties
   - Directed relationships with types and properties
   - Used by Neo4j, Amazon Neptune (Property Graph)

2. **Resource Description Framework (RDF)**
   - Subject-Predicate-Object triples
   - Standardized by W3C
   - Used by Amazon Neptune (RDF), Stardog

### Design Patterns
1. **Adjacency List**
   - Direct connections between nodes
   - Fast for direct traversals

2. **Materialized Path**
   - Store path information in properties
   - Efficient for hierarchical data

3. **Nested Sets**
   - Store left/right values for tree traversal
   - Efficient for tree operations

### Example: Social Network Model
```cypher
// Create users
CREATE (alice:User {id: 1, name: 'Alice', email: 'alice@example.com'})
CREATE (bob:User {id: 2, name: 'Bob', email: 'bob@example.com'})
CREATE (charlie:User {id: 3, name: 'Charlie', email: 'charlie@example.com'})

// Create posts
CREATE (post1:Post {id: 100, title: 'Hello World', content: 'My first post!', timestamp: datetime()})
CREATE (post2:Post {id: 101, title: 'Graph Databases', content: 'Learning about graphs', timestamp: datetime()})

// Create relationships
MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'})
CREATE (a)-[:FOLLOWS {since: date()}]->(b)

MATCH (a:User {name: 'Alice'}), (p:Post {id: 100})
CREATE (a)-[:POSTED]->(p)

MATCH (b:User {name: 'Bob'}), (p:Post {id: 100})
CREATE (b)-[:LIKED {timestamp: datetime()}]->(p)
```

## Performance Considerations

### Indexing
- **Node Indexes**: For fast lookups of nodes by property values
- **Relationship Indexes**: For fast traversal of specific relationship types
- **Full-text Indexes**: For text search across multiple properties

### Query Optimization
- **Use labels** to limit the search space
- **Specify relationship directions** when possible
- **Use parameters** for query reuse
- **Limit path length** in variable-length relationships
- **Use PROFILE** to analyze query performance

### Scaling
- **Sharding**: Distribute graph across multiple machines
- **Replication**: For read scalability and fault tolerance
- **Caching**: Cache frequently accessed subgraphs

## Use Cases and Examples

### 1. Social Network Analysis
```cypher
// Find potential friends (friends of friends)
MATCH (me:User {id: 123})-[:FRIEND]->(friend:User)-[:FRIEND]->(suggestion:User)
WHERE NOT (me)-[:FRIEND]->(suggestion)
RETURN suggestion.name, count(*) AS mutual_friends
ORDER BY mutual_friends DESC
LIMIT 10
```

### 2. Recommendation Engine
```cypher
// Recommend products based on purchase history
MATCH (me:User {id: 123})-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(other:User)
WHERE NOT (me)-[:PURCHASED]->(p)
WITH p, count(other) AS frequency
RETURN p.name, frequency
ORDER BY frequency DESC
LIMIT 5
```

### 3. Fraud Detection
```cypher
// Find potential fraud rings
MATCH (a:Account)-[t1:TRANSFER]->(b:Account)-[t2:TRANSFER]->(c:Account)
WHERE t1.amount > 10000 AND t2.amount > 10000
  AND t1.timestamp > datetime().minus(duration('P1D'))
  AND t2.timestamp > datetime().minus(duration('P1D'))
  AND a <> c
RETURN a.id, b.id, c.id, t1.amount, t2.amount
```

### 4. Knowledge Graph
```cypher
// Find related concepts in a knowledge graph
MATCH (topic:Topic {name: 'Artificial Intelligence'})<-[:RELATED_TO*1..3]-(related:Topic)
RETURN related.name, count(*) AS connection_strength
ORDER BY connection_strength DESC
LIMIT 10
```

## Graph Algorithms

### Built-in Algorithms
1. **Path Finding**
   - Shortest path
   - All-pairs shortest path
   - Single-source shortest path

2. **Centrality**
   - PageRank
   - Betweenness centrality
   - Closeness centrality

3. **Community Detection**
   - Label Propagation
   - Louvain Modularity
   - Strongly Connected Components

4. **Similarity**
   - Jaccard Similarity
   - Cosine Similarity
   - Node Similarity

### Example: PageRank in Neo4j
```cypher
// Create a named graph
CALL gds.graph.create(
  'myGraph',
  'User',
  'FOLLOWS',
  {}
)

// Run PageRank algorithm
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10
```

## Integration with Other Systems

### ETL Tools
- **Apache NiFi**: Data flow automation
- **Apache Kafka**: Stream processing
- **Apache Spark**: Large-scale graph processing

### Visualization
- **Neo4j Bloom**: Graph visualization
- **Gephi**: Open-source network analysis
- **KeyLines**: Commercial graph visualization

### Programming Languages
- **Java/Scala**: Native support through TinkerPop
- **Python**: Py2neo, Gremlin-Python
- **JavaScript/Node.js**: Neo4j JavaScript Driver

## Best Practices

### Data Modeling
- **Start with use cases**: Model based on your queries
- **Use labels effectively**: For grouping and performance
- **Leverage relationships**: Don't be afraid of deep relationships
- **Keep properties simple**: Move complex data to external storage if needed

### Query Optimization
- **Use parameters**: For query plan caching
- **Limit result sets**: Use LIMIT and WHERE clauses
- **Profile queries**: Identify performance bottlenecks
- **Use indexes**: For frequently queried properties

### Performance Tuning
- **Batch operations**: For bulk data loading
- **Use APOC procedures**: For complex operations
- **Monitor memory usage**: Especially for large traversals
- **Consider sharding**: For very large graphs

### Security
- **Role-based access control**: Limit access to sensitive data
- **Property-level security**: Hide sensitive properties
- **Query injection**: Use parameterized queries

## Practice Exercises
1. Set up a local Neo4j instance and load a sample dataset
2. Model a social network with users, posts, and comments
3. Write queries to find the most influential users
4. Implement a recommendation system based on user behavior
5. Visualize a subgraph of your data
6. Optimize a slow-running graph query
7. Implement a graph-based fraud detection system
8. Compare the performance of different graph algorithms on your data

---
Next: [Search Engines and Full-Text Search](./05_search_engines.md)
