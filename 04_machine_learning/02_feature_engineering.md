# Feature Engineering for Machine Learning

## Table of Contents
1. [Introduction to Feature Engineering](#introduction-to-feature-engineering)
2. [Handling Missing Values](#handling-missing-values)
3. [Handling Categorical Variables](#handling-categorical-variables)
4. [Feature Scaling](#feature-scaling)
5. [Feature Transformation](#feature-transformation)
6. [Feature Creation](#feature-creation)
7. [Feature Selection](#feature-selection)
8. [Dimensionality Reduction](#dimensionality-reduction)
9. [Working with Text Data](#working-with-text-data)
10. [Working with Time Series Data](#working-with-time-series-data)
11. [Best Practices](#best-practices)

## Introduction to Feature Engineering

### What is Feature Engineering?
Feature engineering is the process of transforming raw data into features that better represent the underlying problem to predictive models.

### Why is it Important?
- Improves model performance
- Reduces overfitting
- Decreases training time
- Makes the model more interpretable

### The Feature Engineering Process
1. Data exploration and analysis
2. Handle missing values
3. Encode categorical variables
4. Scale and normalize features
5. Create new features
6. Select the best features
7. Validate feature importance

## Handling Missing Values

### Common Approaches
1. **Deletion**
   - Listwise deletion
   - Column-wise deletion

2. **Imputation**
   - Mean/Median/Mode imputation
   - Constant value imputation
   - K-Nearest Neighbors imputation
   - Model-based imputation

### Example: Handling Missing Values
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Sample data
data = {
    'age': [25, 30, np.nan, 35, 40],
    'income': [50000, np.nan, 70000, 80000, 90000],
    'gender': ['M', 'F', 'M', np.nan, 'F']
}
df = pd.DataFrame(data)

# Mean imputation for numerical columns
num_imputer = SimpleImputer(strategy='mean')
df[['age', 'income']] = num_imputer.fit_transform(df[['age', 'income']])

# Mode imputation for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df['gender'] = cat_imputer.fit_transform(df[['gender']])

# KNN imputation (alternative approach)
# knn_imputer = KNNImputer(n_neighbors=2)
# df[['age', 'income']] = knn_imputer.fit_transform(df[['age', 'income']])
```

## Handling Categorical Variables

### Encoding Techniques
1. **Ordinal Encoding**
   - For ordinal categories (e.g., 'low', 'medium', 'high')
   
2. **One-Hot Encoding**
   - For nominal categories (no order)
   
3. **Target Encoding**
   - Replace categories with the mean of the target variable
   
4. **Binary Encoding**
   - Convert categories to binary digits

### Example: Encoding Categorical Variables
```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder

# Sample data
data = {
    'size': ['S', 'M', 'L', 'XL', 'M', 'S'],
    'color': ['red', 'blue', 'green', 'red', 'blue', 'red'],
    'price': [10, 20, 30, 40, 50, 60]
}
df = pd.DataFrame(data)

# Ordinal encoding for size
size_order = ['S', 'M', 'L', 'XL']
ordinal_encoder = OrdinalEncoder(categories=[size_order])
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size']])

# One-hot encoding for color
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
color_encoded = onehot_encoder.fit_transform(df[['color']])
color_df = pd.DataFrame(color_encoded, 
                       columns=onehot_encoder.get_feature_names_out(['color']))
df = pd.concat([df, color_df], axis=1)

# Target encoding (mean encoding)
target_encoder = TargetEncoder()
df['color_encoded'] = target_encoder.fit_transform(df[['color']], df['price'])
```

## Feature Scaling

### Why Scale Features?
- Algorithms that use distance measures (KNN, SVM, K-means) are sensitive to feature scales
- Gradient descent converges faster with scaled features
- Some algorithms require features to be on the same scale

### Scaling Techniques
1. **Standardization (Z-score normalization)**
   - Scales features to have mean=0 and std=1
   - Formula: (x - mean) / std
   
2. **Min-Max Scaling**
   - Scales features to a fixed range (usually 0 to 1)
   - Formula: (x - min) / (max - min)
   
3. **Robust Scaling**
   - Uses median and IQR
   - Less sensitive to outliers
   - Formula: (x - median) / IQR

### Example: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample data
data = {
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_standardized = pd.DataFrame(df_scaled, columns=df.columns)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_minmax = minmax_scaler.fit_transform(df)
df_minmax = pd.DataFrame(df_minmax, columns=df.columns)

# Robust Scaling
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df)
df_robust = pd.DataFrame(df_robust, columns=df.columns)
```

## Feature Transformation

### Common Transformations
1. **Log Transformation**
   - Reduces skewness of right-skewed data
   - Formula: log(x + 1)
   
2. **Square Root Transformation**
   - For moderate skewness
   - Formula: sqrt(x + c)
   
3. **Box-Cox Transformation**
   - General power transformation
   - Handles both positive and negative values
   
4. **Quantile Transformation**
   - Maps data to a specified distribution

### Example: Feature Transformation
```python
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# Sample data with right skew
data = {'income': [10000, 20000, 30000, 40000, 50000, 100000, 200000, 300000]}
df = pd.DataFrame(data)

# Log transformation
df['log_income'] = np.log1p(df['income'])  # log(1 + x)

# Square root transformation
df['sqrt_income'] = np.sqrt(df['income'])

# Box-Cox transformation
pt = PowerTransformer(method='box-cox')
df['boxcox_income'] = pt.fit_transform(df[['income']])

# Quantile transformation (normal distribution)
qt = QuantileTransformer(output_distribution='normal')
df['quantile_income'] = qt.fit_transform(df[['income']])
```

## Feature Creation

### Techniques
1. **Polynomial Features**
   - Create interaction terms and powers of features
   
2. **Binning**
   - Convert continuous variables into categorical bins
   
3. **Domain-Specific Features**
   - Create features based on domain knowledge
   
4. **Date/Time Features**
   - Extract day, month, year, day of week, etc.

### Example: Feature Creation
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Sample data
df = pd.DataFrame({
    'x1': np.random.rand(100),
    'x2': np.random.rand(100),
    'y': np.random.rand(100)
})

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['x1', 'x2']])
poly_columns = poly.get_feature_names_out(['x1', 'x2'])
df_poly = pd.DataFrame(poly_features, columns=poly_columns)

# Binning (convert age to age groups)
age = pd.Series([15, 25, 35, 45, 55, 65, 75])
age_bins = [0, 18, 35, 60, 100]
age_labels = ['child', 'young_adult', 'adult', 'senior']
age_groups = pd.cut(age, bins=age_bins, labels=age_labels)

# Date features
df['date'] = pd.date_range(start='1/1/2023', periods=7, freq='D')
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
```

## Feature Selection

### Why Select Features?
- Reduces overfitting
- Improves model interpretability
- Reduces training time
- May improve model performance

### Selection Techniques
1. **Filter Methods**
   - Select features based on statistical tests
   - Examples: Correlation, Chi-square, ANOVA
   
2. **Wrapper Methods**
   - Use a model to score feature subsets
   - Examples: Forward selection, Backward elimination, RFE
   
3. **Embedded Methods**
   - Feature selection as part of the model training
   - Examples: Lasso, Random Forest feature importance

### Example: Feature Selection
```python
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Filter method: Select top k features
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Wrapper method: Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
selected_features_rfe = X.columns[rfe.support_]

# Embedded method: Feature importance with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

## Dimensionality Reduction

### Why Reduce Dimensionality?
- Reduces computational cost
- Helps with visualization
- Reduces noise and redundancy
- May improve model performance

### Techniques
1. **Principal Component Analysis (PCA)**
   - Linear dimensionality reduction
   - Projects data to lower dimensions
   
2. **t-SNE**
   - Non-linear dimensionality reduction
   - Good for visualization
   
3. **UMAP**
   - Preserves both local and global structure
   - Faster than t-SNE for large datasets

### Example: Dimensionality Reduction
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
axes[0].set_title('PCA')

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
axes[1].set_title('t-SNE')

scatter3 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
axes[2].set_title('UMAP')

plt.tight_layout()
plt.show()
```

## Working with Text Data

### Text Preprocessing
1. **Tokenization**
2. **Lowercasing**
3. **Removing stop words**
4. **Stemming/Lemmatization**
5. **Removing punctuation and numbers**

### Feature Extraction
1. **Bag of Words (CountVectorizer)**
2. **TF-IDF (TfidfVectorizer)**
3. **Word Embeddings (Word2Vec, GloVe, FastText)**
4. **BERT and Transformer-based embeddings**

### Example: Text Processing
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text data
texts = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
processed_texts = [preprocess_text(text) for text in texts]

# Bag of Words
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(processed_texts)
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(processed_texts)
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
```

## Working with Time Series Data

### Time Series Features
1. **Temporal Features**
   - Time since start
   - Time since last event
   - Time until next event
   
2. **Rolling Statistics**
   - Moving averages
   - Rolling standard deviation
   - Expanding windows
   
3. **Date/Time Features**
   - Hour of day
   - Day of week
   - Month
   - Season
   - Is weekend/holiday

### Example: Time Series Feature Engineering
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample time series data
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_data = pd.DataFrame(date_rng, columns=['date'])
ts_data['value'] = np.random.randn(len(date_rng)).cumsum()

# Basic time features
ts_data['day_of_week'] = ts_data['date'].dt.dayofweek
ts_data['month'] = ts_data['date'].dt.month
ts_data['quarter'] = ts_data['date'].dt.quarter
ts_data['is_weekend'] = ts_data['day_of_week'].isin([5, 6]).astype(int)

# Lag features
for lag in [1, 2, 3, 7, 14, 30]:
    ts_data[f'value_lag_{lag}'] = ts_data['value'].shift(lag)

# Rolling statistics
window_sizes = [7, 14, 30]
for window in window_sizes:
    ts_data[f'rolling_mean_{window}'] = ts_data['value'].rolling(window=window).mean()
    ts_data[f'rolling_std_{window}'] = ts_data['value'].rolling(window=window).std()

# Time since last event (example for events)
events = pd.DataFrame({
    'event_date': pd.to_datetime(['2023-03-15', '2023-06-20', '2023-09-10']),
    'event_type': ['promo', 'promo', 'holiday']
})

# Merge events with time series
ts_data = ts_data.merge(
    events, 
    left_on='date', 
    right_on='event_date', 
    how='left'
).fillna({'event_type': 'normal'})

# Time since last event
ts_data['days_since_last_event'] = ts_data['date'].diff().dt.days
```

## Best Practices

1. **Start Simple**
   - Begin with basic features
   - Add complexity gradually
   
2. **Avoid Data Leakage**
   - Fit transformers on training data only
   - Use pipelines to prevent leakage
   
3. **Document Your Features**
   - Keep track of how features were created
   - Document any domain knowledge used
   
4. **Automate Feature Engineering**
   - Create reusable pipelines
   - Use Feature Stores for production
   
5. **Monitor Feature Performance**
   - Track feature importance over time
   - Set up monitoring for data drift

## Practice Exercises
1. Load the Titanic dataset and perform feature engineering:
   - Handle missing values
   - Encode categorical variables
   - Create new features (e.g., family size from sibsp and parch)
   - Scale the features
   
2. For a time series dataset:
   - Create lag features
   - Add rolling statistics
   - Extract date/time features
   - Handle seasonality
   
3. For text data:
   - Clean and preprocess the text
   - Create a bag-of-words representation
   - Apply TF-IDF transformation
   - Visualize word frequencies
   
4. Feature selection:
   - Apply filter methods to select top features
   - Use recursive feature elimination
   - Compare model performance with and without feature selection

---
Next: [Model Training and Evaluation](./03_model_training_evaluation.md)
