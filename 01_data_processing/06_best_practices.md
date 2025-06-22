# Data Processing Best Practices

## Table of Contents
1. [Code Organization](#code-organization)
2. [Performance Optimization](#performance-optimization)
3. [Reproducibility](#reproducibility)
4. [Error Handling and Logging](#error-handling-and-logging)
5. [Testing and Validation](#testing-and-validation)
6. [Documentation](#documentation)
7. [Version Control](#version-control)
8. [Collaboration](#collaboration)

## Code Organization

### Project Structure
```
project/
├── data/
│   ├── raw/           # Raw, immutable data
│   ├── processed/     # Cleaned and processed data
│   └── external/      # External data sources
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── __init__.py
│   ├── data/         # Data processing scripts
│   ├── features/     # Feature engineering
│   ├── models/       # Model training
│   └── utils/        # Utility functions
├── tests/            # Unit and integration tests
├── config/           # Configuration files
├── docs/             # Documentation
├── requirements.txt  # Python dependencies
└── README.md        # Project overview
```

### Modular Code
```python
# data_processing.py
def load_data(filepath):
    """Load data from file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the input DataFrame."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def process_data(filepath):
    """Main data processing pipeline."""
    df = load_data(filepath)
    df = clean_data(df)
    return df

if __name__ == "__main__":
    df = process_data("data/raw/dataset.csv")
    df.to_csv("data/processed/cleaned_data.csv", index=False)
```

## Performance Optimization

### Efficient Operations
- Use vectorized operations instead of loops
- Avoid chained indexing
- Use appropriate data structures
- Leverage in-place operations when possible

### Memory Management
```python
# Release memory
del large_df
import gc
gc.collect()

# Use context managers for file operations
with open('large_file.txt', 'r') as f:
    data = f.read()
```

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def process_partition(partition):
    # Process a single partition
    return partition * 2

def parallel_apply(df, func, n_jobs=4):
    """Apply a function to DataFrame in parallel."""
    partitions = np.array_split(df, n_jobs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, partitions))
    return pd.concat(results)
```

## Reproducibility

### Environment Management
```yaml
# environment.yml
name: data_processing
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - pandas>=1.3.0
  - numpy>=1.20.0
  - scikit-learn>=1.0.0
  - jupyter
  - pip
  - pip:
    - black
    - flake8
```

### Random Seeds
```python
import numpy as np
import random
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Additional seed settings as needed
```

## Error Handling and Logging

### Structured Logging
```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    # Your processing code
    logger.info("Starting data processing")
    result = process_data("input.csv")
    logger.info("Processing completed successfully")
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}", exc_info=True)
    raise
except Exception as e:
    logger.critical(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Data Validation
```python
import pandas as pd
from pandera import DataFrameSchema, Column, Check, SchemaModel

# Define schema
schema = DataFrameSchema({
    "column1": Column(int, checks=Check.greater_than(0)),
    "column2": Column(str, checks=Check.str_matches(r"^[A-Z][a-z]*$")),
    "column3": Column(float, nullable=True)
})

# Validate DataFrame
try:
    schema.validate(df, lazy=True)
except Exception as e:
    logger.error(f"Data validation failed: {e}")
    raise
```

## Testing and Validation

### Unit Tests
```python
# test_data_processing.py
import unittest
import pandas as pd
import numpy as np
from src.data_processing import clean_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['x', 'y', 'z', 'x']
        })
    
    def test_clean_data_removes_nulls(self):
        cleaned = clean_data(self.test_data)
        self.assertFalse(cleaned.isnull().any().any())
    
    def test_clean_data_preserves_columns(self):
        cleaned = clean_data(self.test_data)
        self.assertListEqual(list(cleaned.columns), ['A', 'B'])

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests
```python
# test_integration.py
import unittest
import pandas as pd
from src.data_processing import process_data

class TestIntegration(unittest.TestCase):
    def test_end_to_end_processing(self):
        # Test with sample data
        test_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'value': [10, 20]
        })
        test_file = 'tests/test_data.csv'
        test_df.to_csv(test_file, index=False)
        
        # Process and validate
        result = process_data(test_file)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
```

## Documentation

### Docstrings
```python
def calculate_metrics(y_true, y_pred, metrics=None):
    """
    Calculate evaluation metrics for model predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    metrics : list of str, default=None
        List of metric names to compute. If None, uses ['accuracy', 'precision', 'recall', 'f1'].

    Returns
    -------
    dict
        Dictionary with metric names as keys and computed values as values.
    
    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal.")
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Implementation...
```

### README Example
```markdown
# Data Processing Pipeline

## Overview
This project contains data processing pipelines for cleaning and transforming raw data into analysis-ready datasets.

## Setup
1. Create and activate a virtual environment:
   ```bash
   conda env create -f environment.yml
   conda activate data_processing
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Usage
```python
from src.data_processing import process_data

df = process_data("data/raw/input.csv")
df.to_csv("data/processed/output.csv", index=False)
```

## Project Structure
- `data/raw/`: Raw input data
- `data/processed/`: Processed data
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Source code
- `tests/`: Unit and integration tests
```

## Version Control

### .gitignore
```
# Data
/data/processed/
/data/raw/
*.csv
*.parquet
*.h5

# Environment
.env
.venv
env/
venv/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook
.ipynb_checkpoints

# Logs
logs/
*.log
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    - id: black
      language_version: python3.9

-   repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
    - id: flake8
      additional_dependencies: [flake8-black]

-   repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
    - id: isort
      name: isort (python)
      args: [--profile=black]
```

## Collaboration

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Functions have appropriate docstrings
- [ ] New features have tests
- [ ] Existing tests pass
- [ ] Documentation is updated
- [ ] No commented-out code
- [ ] No sensitive data in commits

### Pull Request Template
```markdown
## Description
Brief description of the changes in this PR.

## Changes
- Change 1
- Change 2
- ...

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing steps

## Documentation
- [ ] Docstrings added/updated
- [ ] README updated
- [ ] Code comments added

## Related Issues
Fixes #123
```

## Practice Exercises
1. Set up a new data processing project with the recommended structure.
2. Write unit tests for an existing data processing function.
3. Create a pre-commit hook configuration for code quality checks.
4. Document a complex function with a complete docstring.
5. Set up logging for a data processing pipeline.

---
Next: [Introduction to Big Data Ecosystem](../02_big_data_ecosystem/01_introduction_to_hadoop.md)
