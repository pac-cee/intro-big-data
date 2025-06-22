# Working with Large Datasets

## Table of Contents
1. [Optimizing Pandas Performance](#optimizing-pandas-performance)
2. [Chunking Large Files](#chunking-large-files)
3. [Using Dask for Out-of-Core Computation](#using-dask-for-out-of-core-computation)
4. [Optimizing Data Types](#optimizing-data-types)
5. [Parallel Processing](#parallel-processing)
6. [Working with Databases](#working-with-databases)
7. [File Formats for Large Datasets](#file-formats-for-large-datasets)

## Optimizing Pandas Performance

### Using Efficient Data Types
```python
import pandas as pd
import numpy as np

# Before optimization
df = pd.read_csv('large_dataset.csv')
print(df.memory_usage(deep=True))

# After optimization
df_optimized = df.copy()

# Downcast numeric columns
for col in df_optimized.select_dtypes(include=['int']).columns:
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

for col in df_optimized.select_dtypes(include=['float']).columns:
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

# Convert object types to category if low cardinality
for col in df_optimized.select_dtypes(include=['object']).columns:
    if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # If less than 50% unique
        df_optimized[col] = df_optimized[col].astype('category')

print(df_optimized.memory_usage(deep=True))
```

### Using Query Instead of Boolean Indexing
```python
# Less efficient
result = df[(df['column1'] > 10) & (df['column2'] == 'value')]

# More efficient
result = df.query('column1 > 10 and column2 == "value"')
```

## Chunking Large Files

### Processing in Chunks
```python
# Define chunk size
chunk_size = 100000  # Adjust based on available memory

# Process file in chunks
chunk_list = []
for chunk in pd.read_csv('very_large_file.csv', chunksize=chunk_size):
    # Process each chunk
    chunk_processed = process_chunk(chunk)  # Your processing function
    chunk_list.append(chunk_processed)

# Combine results
df_processed = pd.concat(chunk_list, axis=0)
```

### Aggregating Chunks
```python
def process_chunk(chunk):
    # Perform aggregation on each chunk
    return chunk.groupby('category')['value'].sum()

# Initialize empty result DataFrame
result = pd.DataFrame()

# Process and aggregate chunks
for chunk in pd.read_csv('large_file.csv', chunksize=100000):
    chunk_result = process_chunk(chunk)
    if result.empty:
        result = chunk_result
    else:
        result = result.add(chunk_result, fill_value=0)
```

## Using Dask for Out-of-Core Computation

### Dask DataFrames
```python
import dask.dataframe as dd

# Read large CSV file
ddf = dd.read_csv('very_large_*.csv')

# Perform operations (lazy evaluation)
result = ddf.groupby('category')['value'].mean().compute()

# Write results
ddf.to_parquet('output_parquet', engine='pyarrow')
```

### Dask Best Practices
```python
# Set number of workers
dask.config.set(scheduler='processes', num_workers=4)

# Persist frequently used data in memory
ddf = ddf.persist()

# Use efficient operations
ddf = ddf.set_index('id')  # Set index for faster lookups
```

## Optimizing Data Types

### Memory-Efficient Data Types
```python
# Before optimization
print(df.memory_usage(deep=True))

# Convert to categorical for low-cardinality columns
for col in df.select_dtypes(include=['object']):
    num_unique = len(df[col].unique())
    num_total = len(df[col])
    if num_unique / num_total < 0.5:  # If less than 50% unique values
        df[col] = df[col].astype('category')

# Downcast numeric types
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')

# Convert boolean columns
df['bool_col'] = df['bool_col'].astype('bool')

print(df.memory_usage(deep=True))
```

## Parallel Processing

### Using multiprocessing
```python
import multiprocessing as mp
import numpy as np

def process_data(chunk):
    # Process a chunk of data
    return chunk * 2

# Split data into chunks
num_processes = mp.cpu_count()
chunks = np.array_split(df, num_processes)

# Create a pool of workers
with mp.Pool(processes=num_processes) as pool:
    # Process chunks in parallel
    results = pool.map(process_data, chunks)

# Combine results
processed_df = pd.concat(results)
```

### Using joblib
```python
from joblib import Parallel, delayed
import pandas as pd

def process_chunk(chunk):
    # Process chunk
    return chunk * 2

# Process chunks in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_chunk)(chunk) for chunk in np.array_split(df, 4)
)

# Combine results
processed_df = pd.concat(results)
```

## Working with Databases

### Using SQLite
```python
import sqlite3
import pandas as pd

# Create a connection
conn = sqlite3.connect('database.db')

# Write to database
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Query with chunks
chunk_size = 100000
for chunk in pd.read_sql_query('SELECT * FROM large_table', conn, chunksize=chunk_size):
    process(chunk)  # Your processing function

# Close connection
conn.close()
```

### Using SQLAlchemy
```python
from sqlalchemy import create_engine

# Create connection
engine = create_engine('sqlite:///database.db')

# Write to database
df.to_sql('table_name', engine, if_exists='replace', index=False, chunksize=10000)

# Read in chunks
for chunk in pd.read_sql_query('SELECT * FROM large_table', engine, chunksize=100000):
    process(chunk)
```

## File Formats for Large Datasets

### Parquet Format
```python
# Write to Parquet
df.to_parquet('data.parquet', engine='pyarrow')

# Read from Parquet
df = pd.read_parquet('data.parquet')

# Read specific columns
df = pd.read_parquet('data.parquet', columns=['col1', 'col2'])

# Read with filters
df = pd.read_parquet('data.parquet', filters=[('date', '>', '2023-01-01')])
```

### HDF5 Format
```python
# Write to HDF5
df.to_hdf('data.h5', key='df', mode='w')

# Read from HDF5
df = pd.read_hdf('data.h5', key='df')

# Store multiple DataFrames
with pd.HDFStore('store.h5') as store:
    store['df1'] = df1
    store['df2'] = df2
```

### Feather Format
```python
# Write to Feather
df.to_feather('data.feather')

# Read from Feather
df = pd.read_feather('data.feather')
```

## Practice Exercises
1. Optimize the memory usage of a large DataFrame by converting data types appropriately.
2. Process a large CSV file in chunks and calculate aggregate statistics.
3. Use Dask to perform operations on a dataset that doesn't fit in memory.
4. Compare the performance of different file formats (CSV, Parquet, HDF5) for a large dataset.
5. Implement a parallel processing pipeline for a CPU-intensive data transformation task.

---
Next: [Data Processing Best Practices](./06_best_practices.md)
