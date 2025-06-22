# Model Training and Evaluation

## Table of Contents
1. [Introduction to Model Training](#introduction-to-model-training)
2. [Splitting Data](#splitting-data)
3. [Model Training](#model-training)
4. [Cross-Validation](#cross-validation)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Persistence](#model-persistence)
8. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
9. [Handling Class Imbalance](#handling-class-imbalance)
10. [Best Practices](#best-practices)

## Introduction to Model Training

### The Model Training Process
1. **Data Preparation**
   - Clean and preprocess data
   - Feature engineering
   - Handle missing values and outliers

2. **Model Selection**
   - Choose appropriate algorithm(s)
   - Consider problem type (classification, regression, etc.)
   - Account for data size and complexity

3. **Training**
   - Split data into training and validation sets
   - Train model on training data
   - Tune hyperparameters

4. **Evaluation**
   - Assess model performance on test set
   - Compare with baseline models
   - Analyze errors

5. **Deployment**
   - Save the trained model
   - Create prediction API
   - Monitor in production

## Splitting Data

### Train-Validation-Test Split
- **Training Set**: Used to train the model (60-80%)
- **Validation Set**: Used for hyperparameter tuning (10-20%)
- **Test Set**: Used for final evaluation (10-20%)

### Time-Based Split
- For time series data, maintain temporal order
- Train on past, validate on more recent, test on most recent

### Stratified Sampling
- Maintains the same class distribution in splits
- Important for imbalanced datasets

### Example: Data Splitting
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Basic train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# For time series data
# X_train, X_test = X[:int(0.7*len(X))], X[int(0.7*len(X)):]
# y_train, y_test = y[:int(0.7*len(y))], y[int(0.7*len(y)):]

# For validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

## Model Training

### Common Algorithms
1. **Linear Models**
   - Linear/Logistic Regression
   - Ridge/Lasso Regression
   - Support Vector Machines (SVM)

2. **Tree-Based Models**
   - Decision Trees
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM, CatBoost)

3. **Neural Networks**
   - Multi-layer Perceptron (MLP)
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)

### Example: Training Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Initialize models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(random_state=42, n_estimators=100),
    'svm': SVC(probability=True, random_state=42)
}

# Train models
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} training complete!")
```

## Cross-Validation

### K-Fold Cross-Validation
- Split data into K folds
- Train on K-1 folds, validate on 1 fold
- Repeat K times with different validation fold
- Average results

### Stratified K-Fold
- Preserves class distribution in each fold
- Better for imbalanced datasets

### Time Series Cross-Validation
- Maintains temporal order
- Expanding or sliding window

### Example: Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Basic K-Fold CV
cv = 5
scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=cv, scoring='accuracy'
)
print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Stratified K-Fold
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=stratified_cv, scoring='accuracy'
)

# Time Series CV
tscv = TimeSeriesSplit(n_splits=5)
time_series_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=tscv, scoring='accuracy'
)
```

## Evaluation Metrics

### Classification Metrics
1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
2. **Precision**: TP / (TP + FP)
3. **Recall/Sensitivity**: TP / (TP + FN)
4. **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
5. **ROC-AUC**: Area under ROC curve
6. **Confusion Matrix**: TP, FP, TN, FN

### Regression Metrics
1. **Mean Absolute Error (MAE)**: Average absolute difference
2. **Mean Squared Error (MSE)**: Average squared difference
3. **Root Mean Squared Error (RMSE)**: sqrt(MSE)
4. **R² Score**: Variance explained by the model
5. **Mean Absolute Percentage Error (MAPE)**: Average percentage error

### Example: Model Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_classification(y_true, y_pred, y_proba=None):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_proba[:, 1]):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def evaluate_regression(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")

# Example usage for classification
y_pred = trained_models['random_forest'].predict(X_test)
y_proba = trained_models['random_forest'].predict_proba(X_test)
evaluate_classification(y_test, y_pred, y_proba)

# Example usage for regression
# y_pred = model.predict(X_test)
# evaluate_regression(y_test, y_pred)
```

## Hyperparameter Tuning

### Grid Search
- Exhaustive search over specified parameter values
- Guaranteed to find best combination in search space
- Computationally expensive

### Random Search
- Randomly samples parameter combinations
- More efficient than grid search for large parameter spaces
- May miss optimal combination

### Bayesian Optimization
- Builds probabilistic model of the objective function
- Focuses on promising hyperparameters
- More efficient than random search

### Example: Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Random Search
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(np.arange(5, 50, 5)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

## Model Persistence

### Saving and Loading Models
- **Pickle**: Python's native serialization format
- **Joblib**: More efficient for large NumPy arrays
- **ONNX**: Open standard for machine learning models
- **Model-specific formats**: e.g., XGBoost's .model, TensorFlow's SavedModel

### Example: Model Persistence
```python
import joblib
import pickle

# Save model using joblib (better for large models)
joblib.dump(trained_models['random_forest'], 'random_forest_model.joblib')

# Save model using pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(trained_models['random_forest'], f)

# Load models
loaded_model_joblib = joblib.load('random_forest_model.joblib')

with open('random_forest_model.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)

# Save model metadata
import json

model_metadata = {
    'model_name': 'random_forest',
    'version': '1.0',
    'features': list(X_train.columns) if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])],
    'target': 'target',
    'metrics': {
        'accuracy': accuracy_score(y_test, trained_models['random_forest'].predict(X_test)),
        'f1_score': f1_score(y_test, trained_models['random_forest'].predict(X_test), average='weighted')
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
```

## Bias-Variance Tradeoff

### Understanding the Tradeoff
- **Bias**: Error from overly-simplistic model
- **Variance**: Error from overly-complex model
- **Total Error = Bias² + Variance + Irreducible Error**

### Diagnosing
- **High Bias (Underfitting)**:
  - High training error
  - Validation error similar to training error
  - Model is too simple
  
- **High Variance (Overfitting)**:
  - Low training error
  - High validation error
  - Model is too complex

### Addressing Issues
- **High Bias**:
  - Add more features
  - Use more complex model
  - Reduce regularization
  
- **High Variance**:
  - Get more training data
  - Use simpler model
  - Increase regularization
  - Apply dropout (for neural networks)

### Learning Curves
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    
    plt.title('Learning Curves')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Example usage
plot_learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5
)
```

## Handling Class Imbalance

### Techniques
1. **Resampling**
   - **Oversampling**: Duplicate minority class samples (e.g., RandomOverSampler)
   - **Undersampling**: Remove majority class samples (e.g., RandomUnderSampler)
   - **SMOTE**: Synthetic Minority Over-sampling Technique

2. **Algorithm-Level**
   - Class weights
   - Cost-sensitive learning
   - Threshold moving

3. **Evaluation Metrics**
   - Precision-Recall curve
   - F1-Score
   - ROC-AUC

### Example: Handling Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Generate imbalanced data
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=2,
    n_redundant=10, n_classes=2, weights=[0.9, 0.1],
    random_state=42
)

# Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Model with class weights
weighted_model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)

# SMOTE + Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Undersampling + Random Forest
pipeline_under = Pipeline([
    ('undersampler', RandomUnderSampler(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Compare different approaches
from sklearn.model_selection import cross_val_score

print("Original:", cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, scoring='f1'
).mean())

print("With class weights:", cross_val_score(
    weighted_model, X, y, cv=5, scoring='f1'
).mean())

print("With SMOTE:", cross_val_score(
    pipeline, X, y, cv=5, scoring='f1'
).mean())

print("With undersampling:", cross_val_score(
    pipeline_under, X, y, cv=5, scoring='f1'
).mean())
```

## Best Practices

1. **Start Simple**
   - Begin with a simple baseline model
   - Gradually increase complexity
   
2. **Use Cross-Validation**
   - Prefer k-fold CV over single train-test split
   - Use stratified CV for imbalanced data
   
3. **Feature Importance**
   - Analyze which features are most important
   - Remove or combine less important features
   
4. **Regularization**
   - Use L1/L2 regularization to prevent overfitting
   - Tune regularization strength
   
5. **Ensemble Methods**
   - Combine multiple models for better performance
   - Use bagging, boosting, or stacking
   
6. **Track Experiments**
   - Log hyperparameters and metrics
   - Use tools like MLflow or Weights & Biases
   
7. **Monitor in Production**
   - Track model performance over time
   - Set up alerts for data drift
   - Plan for model retraining

## Practice Exercises
1. **Model Comparison**
   - Train multiple models on a dataset
   - Compare their performance using appropriate metrics
   - Visualize the results
   
2. **Hyperparameter Tuning**
   - Perform grid search and random search
   - Compare the results
   - Visualize the effect of different hyperparameters
   
3. **Bias-Variance Analysis**
   - Generate learning curves for different model complexities
   - Identify if the model is underfitting or overfitting
   - Apply appropriate techniques to address the issues
   
4. **Class Imbalance**
   - Work with an imbalanced dataset
   - Apply different techniques to handle the imbalance
   - Compare the results using appropriate metrics
   
5. **End-to-End Project**
   - Load and preprocess data
   - Perform feature engineering
   - Train and evaluate multiple models
   - Tune hyperparameters
   - Save the best model
   - Create a simple API for predictions

---
Next: [Ensemble Methods and Model Stacking](./04_ensemble_methods.md)
