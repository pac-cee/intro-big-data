# Big Data Processing with PySpark

## Project Overview
This project demonstrates how to process large datasets using Apache Spark with Python (PySpark). We'll cover data ingestion, transformation, analysis, and machine learning at scale.

## Project Structure
```
03_big_data_processing/
├── data/
│   ├── raw/                 # Raw input data
│   ├── processed/           # Processed data
│   └── models/             # Trained ML models
├── notebooks/
│   └── pyspark_analysis.ipynb  # Jupyter notebook with PySpark analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── data_transformer.py # Data transformation functions
│   └── ml_pipeline.py      # Machine learning pipeline
├── config/
│   └── config.yaml       # Configuration file
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # For containerization
└── README.md
```

## Setup Instructions

### Prerequisites
- Java 8 or later
- Python 3.8+
- Docker (optional, for containerized setup)

### Local Setup
1. Install PySpark:
   ```bash
   pip install pyspark==3.4.1 findspark
   ```

2. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup (Alternative)
1. Build the Docker image:
   ```bash
   docker build -t pyspark-project .
   ```

2. Run the container:
   ```bash
   docker run -p 8888:8888 pyspark-project
   ```

## Project Tasks
1. Set up a PySpark environment
2. Load and process large datasets
3. Perform data transformations and aggregations
4. Implement a machine learning pipeline
5. Analyze and visualize results

## Implementation

### 1. Data Loading (src/data_loader.py)
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from typing import Optional, Dict, Any
import yaml

class DataLoader:
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the DataLoader with configuration."""
        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_spark_session(self) -> SparkSession:
        """Initialize and return a Spark session."""
        return SparkSession.builder \
            .appName(self.config['spark']['app_name']) \
            .config("spark.executor.memory", self.config['spark']['executor_memory']) \
            .config("spark.driver.memory", self.config['spark']['driver_memory']) \
            .getOrCreate()
    
    def load_csv(self, file_path: str, schema: Optional[StructType] = None) -> DataFrame:
        """Load a CSV file into a Spark DataFrame."""
        return self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .schema(schema) if schema else None \
            .csv(file_path)
    
    def load_json(self, file_path: str) -> DataFrame:
        """Load a JSON file into a Spark DataFrame."""
        return self.spark.read.json(file_path)
    
    def load_parquet(self, file_path: str) -> DataFrame:
        """Load a Parquet file into a Spark DataFrame."""
        return self.spark.read.parquet(file_path)
```

### 2. Data Transformation (src/data_transformer.py)
```python
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from typing import List, Dict, Any, Optional

class DataTransformer:
    @staticmethod
    def clean_data(df: DataFrame, date_columns: List[str] = None) -> DataFrame:
        """Clean the input DataFrame."""
        # Convert date columns
        if date_columns:
            for col in date_columns:
                df = df.withColumn(col, F.to_date(F.col(col)))
        
        # Handle missing values
        df = df.na.fill({
            'numeric_column': 0,
            'string_column': 'Unknown'
        })
        
        return df
    
    @staticmethod
    def aggregate_data(df: DataFrame, 
                      group_columns: List[str], 
                      agg_columns: Dict[str, List[str]]) -> DataFrame:
        """Aggregate data by specified columns."""
        agg_exprs = []
        for agg_func, columns in agg_columns.items():
            for col in columns:
                if agg_func == 'avg':
                    agg_exprs.append(F.avg(col).alias(f'avg_{col}'))
                elif agg_func == 'sum':
                    agg_exprs.append(F.sum(col).alias(f'sum_{col}'))
                elif agg_func == 'count':
                    agg_exprs.append(F.count(col).alias(f'count_{col}'))
        
        return df.groupBy(group_columns).agg(*agg_exprs)
    
    @staticmethod
    def join_data(df1: DataFrame, df2: DataFrame, on: List[str], how: str = 'inner') -> DataFrame:
        """Join two DataFrames."""
        return df1.join(df2, on=on, how=how)
    
    @staticmethod
    def filter_data(df: DataFrame, condition: str) -> DataFrame:
        """Filter DataFrame based on SQL condition."""
        return df.filter(condition)
```

### 3. Machine Learning Pipeline (src/ml_pipeline.py)
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, 
    OneHotEncoder, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

class MLPipeline:
    def __init__(self, target_col: str, numeric_cols: List[str], 
                 categorical_cols: List[str] = None):
        """Initialize the ML pipeline."""
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols or []
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> Pipeline:
        """Build the ML pipeline."""
        stages = []
        
        # String indexing for categorical columns
        index_output_cols = [f"{col}_index" for col in self.categorical_cols]
        for col, col_index in zip(self.categorical_cols, index_output_cols):
            string_indexer = StringIndexer(
                inputCol=col, 
                outputCol=col_index,
                handleInvalid="keep"
            )
            stages.append(string_indexer)
        
        # One-hot encoding
        encoder_output_cols = [f"{col}_encoded" for col in self.categorical_cols]
        for col_index, col_encoded in zip(index_output_cols, encoder_output_cols):
            encoder = OneHotEncoder(
                inputCol=col_index,
                outputCol=col_encoded
            )
            stages.append(encoder)
        
        # Assemble features
        feature_columns = self.numeric_cols + encoder_output_cols
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        stages.append(assembler)
        
        # Standard scaling
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Random Forest Classifier
        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol=self.target_col,
            numTrees=100,
            maxDepth=5,
            seed=42
        )
        stages.append(rf)
        
        return Pipeline(stages=stages)
    
    def fit(self, train_df: DataFrame) -> Pipeline:
        """Fit the pipeline on training data."""
        return self.pipeline.fit(train_df)
    
    def evaluate(self, model, test_df: DataFrame) -> Dict[str, float]:
        """Evaluate the model on test data."""
        predictions = model.transform(test_df)
        
        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.target_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        accuracy = evaluator.evaluate(predictions)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
```

### 4. Example Jupyter Notebook (notebooks/pyspark_analysis.ipynb)
```python
# Initialize Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.data_loader import DataLoader
from src.data_transformer import DataTransformer
from src.ml_pipeline import MLPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark
spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load data
data_loader = DataLoader()
df = data_loader.load_csv("data/raw/sample_data.csv")

# Show schema and sample data
df.printSchema()
df.show(5)

# Data transformation
transformer = DataTransformer()
df_clean = transformer.clean_data(df, date_columns=['date_column'])

# Aggregation example
agg_df = transformer.aggregate_data(
    df_clean,
    group_columns=['category'],
    agg_columns={
        'avg': ['price', 'quantity'],
        'sum': ['revenue'],
        'count': ['order_id']
    }
)
agg_df.show()

# Machine Learning
# Prepare data
ml_df = df_clean.select(['feature1', 'feature2', 'target'])
train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

# Train model
ml_pipeline = MLPipeline(
    target_col='target',
    numeric_cols=['feature1'],
    categorical_cols=['feature2']
)

model = ml_pipeline.fit(train_df)

# Evaluate model
results = ml_pipeline.evaluate(model, test_df)
print(f"Model Accuracy: {results['accuracy']:.4f}")

# Show feature importance
rf_model = model.stages[-1]
feature_importances = rf_model.featureImportances
features = ['feature1', 'feature2_encoded']

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/processed/feature_importance.png')
plt.show()

# Stop Spark session
spark.stop()
```

### 5. Configuration (config/config.yaml)
```yaml
spark:
  app_name: "BigDataProcessing"
  executor_memory: "4g"
  driver_memory: "4g"
  
data:
  input_path: "data/raw/"
  output_path: "data/processed/"
  model_path: "data/models/"

model:
  num_trees: 100
  max_depth: 5
  seed: 42
```

### 6. Dockerfile
```dockerfile
FROM jupyter/pyspark-notebook:latest

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create working directory
WORKDIR /home/jovyan/work

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["start-notebook.sh", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

## Running the Project

### Local Execution
1. Start Jupyter Lab:
   ```bash
   jupyter lab notebooks/
   ```

2. Open and run `pyspark_analysis.ipynb`

### Cluster Execution
1. Submit the job to a Spark cluster:
   ```bash
   spark-submit \
       --master yarn \
       --deploy-mode cluster \
       --executor-memory 4G \
       --num-executors 10 \
       src/main.py
   ```

## Expected Output
- Processed data in Parquet/CSV format
- Trained ML models
- Visualizations and analysis results
- Performance metrics and logs

## Project Extensions
1. Add more data sources and implement complex joins
2. Implement streaming data processing
3. Add more machine learning models
4. Create a REST API for model serving
5. Implement monitoring and logging

## Learning Objectives
- Distributed data processing with PySpark
- Building scalable data pipelines
- Machine learning at scale
- Performance optimization
- Cluster resource management

## Next Steps
1. Explore Spark SQL for complex queries
2. Learn about Spark Streaming for real-time processing
3. Deploy to a cloud provider (AWS EMR, Databricks, etc.)
4. Implement advanced monitoring and alerting
