# Advanced Pandas Operations

## Table of Contents
1. [Merging and Joining DataFrames](#merging-and-joining-dataframes)
2. [Pivot Tables and Cross-tabulations](#pivot-tables-and-cross-tabulations)
3. [Time Series Operations](#time-series-operations)
4. [Handling Categorical Data](#handling-categorical-data)
5. [Performance Optimization](#performance-optimization)

## Merging and Joining DataFrames

### Concatenation
```python
import pandas as pd

# Create sample DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5']})

# Concatenate vertically
result = pd.concat([df1, df2])
```

### Merging (SQL-style joins)
```python
# Sample DataFrames
left = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                     'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K3'],
                      'C': ['C0', 'C1', 'C3'],
                      'D': ['D0', 'D1', 'D3']})

# Inner join
pd.merge(left, right, on='key')

# Left join
pd.merge(left, right, on='key', how='left')

# Right join
pd.merge(left, right, on='key', how='right')

# Outer join
pd.merge(left, right, on='key', how='outer')
```

## Pivot Tables and Cross-tabulations

### Pivot Tables
```python
# Sample data
data = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Category': ['A', 'B', 'A', 'B'],
    'Value': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

# Create pivot table
pivot = df.pivot_table(index='Date', columns='Category', values='Value', aggfunc='sum')
```

### Cross-tabulation
```python
# Sample data
data = {
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'Education': ['High School', 'College', 'High School', 'Masters', 'College', 'PhD']
}

# Create cross-tabulation
pd.crosstab(data['Gender'], data['Education'])
```

## Time Series Operations

### Working with Dates
```python
# Create date range
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]}, index=dates)

# Resample time series data
weekly = df.resample('W').sum()

# Rolling window calculations
rolling_mean = df.rolling(window=2).mean()
```

## Handling Categorical Data

### Converting to Categorical
```python
df['Category'] = df['Category'].astype('category')

# Benefits:
# - Uses less memory
# - Can specify order
# - Can specify categories
```

### Working with Categoricals
```python
# Create ordered categorical
categories = ['Low', 'Medium', 'High']
df['Priority'] = pd.Categorical(['High', 'Low', 'Medium', 'High', 'Low'],
                               categories=categories,
                               ordered=True)

# Sorting respects the order
df.sort_values('Priority')
```

## Performance Optimization

### Using apply() vs Vectorized Operations
```python
# Slower: Using apply
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Faster: Vectorized operation
df['new_col'] = df['col'] * 2
```

### Using .loc vs direct access
```python
# Slower: Chained indexing
df[df['age'] > 30]['name']

# Faster: Using .loc
df.loc[df['age'] > 30, 'name']
```

### Using appropriate data types
```python
# Check memory usage
df.memory_usage(deep=True)

# Downcast numeric columns
df['col'] = pd.to_numeric(df['col'], downcast='integer')
```

## Practice Exercises
1. Merge two DataFrames on a common column using different types of joins.
2. Create a pivot table showing average sales by category and month.
3. Convert a date column to datetime and extract month and year.
4. Optimize the memory usage of a large DataFrame.
5. Use groupby with multiple aggregation functions on different columns.

---
Next: [Data Processing with NumPy](./03_numpy_for_data_processing.md)
