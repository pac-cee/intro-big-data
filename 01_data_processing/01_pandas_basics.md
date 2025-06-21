# Pandas Basics for Data Processing

## Table of Contents
1. [Introduction to Pandas](#introduction-to-pandas)
2. [Series and DataFrames](#series-and-dataframes)
3. [Data Selection and Filtering](#data-selection-and-filtering)
4. [Data Cleaning](#data-cleaning)
5. [Basic Operations](#basic-operations)

## Introduction to Pandas
Pandas is a powerful Python library for data manipulation and analysis. It provides data structures and operations for manipulating numerical tables and time series.

### Key Features
- DataFrame object for data manipulation with integrated indexing
- Tools for reading and writing data between in-memory data structures and different file formats
- Data alignment and integrated handling of missing data
- Flexible reshaping and pivoting of datasets
- Label-based slicing, fancy indexing, and subsetting of large datasets
- High-performance merging and joining of data

## Series and DataFrames

### Series
A one-dimensional labeled array capable of holding any data type.

```python
import pandas as pd

# Create a Series from a list
s = pd.Series([1, 3, 5, 7, 9])
print(s)

# With custom index
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
```

### DataFrame
A two-dimensional labeled data structure with columns of potentially different types.

```python
# Create a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)
print(df)
```

## Data Selection and Filtering

### Selecting Columns
```python
# Select single column
ages = df['Age']

# Select multiple columns
subset = df[['Name', 'City']]
```

### Filtering Rows
```python
# Filter rows where Age > 30
older_than_30 = df[df['Age'] > 30]

# Multiple conditions
filtered = df[(df['Age'] > 25) & (df['City'] == 'New York')]
```

## Data Cleaning

### Handling Missing Values
```python
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()

# Fill missing values
df_filled = df.fillna(0)  # or df.fillna(df.mean())
```

### Data Types and Conversion
```python
# Check data types
print(df.dtypes)

# Convert data type
df['Age'] = df['Age'].astype('float64')
```

## Basic Operations

### Descriptive Statistics
```python
# Basic statistics
print(df.describe())

# Mean, median, etc.
print(df['Age'].mean())
print(df['Age'].median())
```

### Grouping and Aggregation
```python
# Group by column and calculate mean
grouped = df.groupby('City').mean()

# Multiple aggregations
agg_df = df.groupby('City').agg({
    'Age': ['mean', 'min', 'max', 'count']
})
```

## Practice Exercises
1. Load a CSV file into a DataFrame and display its first 5 rows.
2. Filter the DataFrame to show only people older than 30.
3. Calculate the average age by city.
4. Handle any missing values in the dataset.
5. Create a new column that categorizes ages into groups (e.g., 'Young', 'Adult', 'Senior').

---
Next: [Advanced Pandas Operations](./02_advanced_pandas.md)
