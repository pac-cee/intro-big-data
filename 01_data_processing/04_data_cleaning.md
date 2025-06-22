# Data Cleaning and Preprocessing

## Table of Contents
1. [Handling Missing Data](#handling-missing-data)
2. [Data Transformation](#data-transformation)
3. [Handling Outliers](#handling-outliers)
4. [Feature Scaling](#feature-scaling)
5. [Encoding Categorical Variables](#encoding-categorical-variables)
6. [Text Data Preprocessing](#text-data-preprocessing)
7. [Date and Time Processing](#date-and-time-processing)

## Handling Missing Data

### Identifying Missing Data
```python
import pandas as pd
import numpy as np

# Create sample data with missing values
data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Check for missing values
print(df.isnull())
print(df.isnull().sum())
```

### Strategies for Handling Missing Data

#### 1. Removing Missing Values
```python
# Drop rows with any missing values
df_dropped = df.dropna()

# Drop columns with any missing values
df_dropped_cols = df.dropna(axis=1)

# Drop rows where all values are missing
df_dropped_all = df.dropna(how='all')
```

#### 2. Imputation
```python
# Fill with a constant value
df_fill = df.fillna(0)

# Forward fill
df_ffill = df.fillna(method='ffill')
# Backward fill
df_bfill = df.fillna(method='bfill')

# Fill with mean/median/mode
mean_value = df['A'].mean()
df['A'].fillna(mean_value, inplace=True)

# Using SimpleImputer (scikit-learn)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

## Data Transformation

### Normalization and Standardization
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Min-Max Scaling (0-1)
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[['A', 'B']])

# Standardization (mean=0, std=1)
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df[['A', 'B']])

# Robust Scaling (for data with outliers)
scaler = RobustScaler()
df_robust = scaler.fit_transform(df[['A', 'B']])
```

### Log Transformation
```python
# For right-skewed data
df['log_A'] = np.log1p(df['A'])  # log(1+x)
```

## Handling Outliers

### Detecting Outliers
```python
# Using IQR
Q1 = df['A'].quantile(0.25)
Q3 = df['A'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['A'] < lower_bound) | (df['A'] > upper_bound)]

# Using Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['A']))
outliers = df[z_scores > 3]
```

### Treating Outliers
```python
# Capping
df['A_capped'] = np.where(df['A'] > upper_bound, upper_bound,
                        np.where(df['A'] < lower_bound, lower_bound, df['A']))

# Winsorization
from scipy.stats.mstats import winsorize
df['A_winsorized'] = winsorize(df['A'], limits=[0.05, 0.05])
```

## Feature Scaling

### When to Scale?
- Distance-based algorithms (KNN, K-means, SVM with RBF kernel)
- Regularized models (Ridge, Lasso)
- Neural Networks
- PCA

### Scaling Techniques
```python
# Already covered in Data Transformation section
# - MinMaxScaler
# - StandardScaler
# - RobustScaler
```

## Encoding Categorical Variables

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

### One-Hot Encoding
```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['category'])

# Using scikit-learn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
```

### Target Encoding
```python
from category_encoders import TargetEncoder

te = TargetEncoder()
df['category_encoded'] = te.fit_transform(df['category'], df['target'])
```

## Text Data Preprocessing

### Basic Text Cleaning
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
```

### TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['clean_text'])
```

## Date and Time Processing

### Parsing and Extracting Features
```python
# Convert string to datetime
df['date'] = pd.to_datetime(df['date_string'])

# Extract features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Time since event
df['days_since_event'] = (pd.Timestamp.now() - df['date']).dt.days
```

### Time-based Aggregation
```python
# Resample time series data
df.set_index('date', inplace=True)
df_resampled = df['value'].resample('D').mean()  # Daily average

# Rolling window
df_rolling = df['value'].rolling(window=7).mean()  # 7-day moving average
```

## Practice Exercises
1. Load a dataset with missing values and apply different imputation strategies.
2. Detect and handle outliers in a numerical column.
3. Encode a categorical variable using different encoding techniques.
4. Preprocess a text column by cleaning and vectorizing it.
5. Create time-based features from a datetime column.

---
Next: [Working with Large Datasets](./05_large_datasets.md)
