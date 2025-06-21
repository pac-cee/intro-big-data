# Introduction to Machine Learning

## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Types of Machine Learning](#types-of-machine-learning)
3. [Supervised Learning](#supervised-learning)
4. [Unsupervised Learning](#unsupervised-learning)
5. [Model Evaluation](#model-evaluation)
6. [Machine Learning with Scikit-learn](#machine-learning-with-scikit-learn)
7. [Practical Example: House Price Prediction](#practical-example-house-price-prediction)

## What is Machine Learning?

Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

### Key Concepts
- **Features**: Input variables
- **Labels**: Output variables (in supervised learning)
- **Model**: Representation learned from data
- **Training**: Process of learning from data
- **Inference**: Making predictions on new data

## Types of Machine Learning

### 1. Supervised Learning
- Learn a mapping from inputs to outputs
- Requires labeled training data
- Examples: Classification, Regression

### 2. Unsupervised Learning
- Find patterns in unlabeled data
- No explicit output variables
- Examples: Clustering, Dimensionality Reduction

### 3. Reinforcement Learning
- Learn by interacting with an environment
- Rewards and penalties guide learning
- Examples: Game playing, Robotics

## Supervised Learning

### Classification
Predicting discrete class labels.

**Example Algorithms**:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks

### Regression
Predicting continuous values.

**Example Algorithms**:
- Linear Regression
- Polynomial Regression
- Decision Trees
- Neural Networks

## Unsupervised Learning

### Clustering
Grouping similar data points.

**Example Algorithms**:
- K-Means
- Hierarchical Clustering
- DBSCAN

### Dimensionality Reduction
Reducing the number of features.

**Example Algorithms**:
- PCA (Principal Component Analysis)
- t-SNE
- Autoencoders

## Model Evaluation

### Classification Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

### Cross-Validation
- K-Fold Cross-Validation
- Stratified K-Fold
- Train-Test Split

## Machine Learning with Scikit-learn

### Basic Workflow
1. Load and prepare data
2. Split data into training and test sets
3. Choose a model
4. Train the model
5. Make predictions
6. Evaluate the model

### Example: Classification with Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

## Practical Example: House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('house_prices.csv')

# Handle missing values
data = data.fillna(data.median())

# Select features and target
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']]
y = data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

## Practice Exercises
1. Load the Titanic dataset and predict survival
2. Implement K-means clustering on the Iris dataset
3. Build a sentiment analysis model using movie reviews
4. Tune hyperparameters using GridSearchCV
5. Deploy a simple ML model as a web service

---
Next: [Feature Engineering for Machine Learning](./02_feature_engineering.md)
