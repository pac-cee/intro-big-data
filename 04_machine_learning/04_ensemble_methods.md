# Ensemble Methods and Model Stacking

## Table of Contents
1. [Introduction to Ensemble Learning](#introduction-to-ensemble-learning)
2. [Voting Classifiers/Regressors](#voting-classifiersregressors)
3. [Bagging](#bagging)
4. [Boosting](#boosting)
5. [Stacking](#stacking)
6. [Blending](#blending)
7. [Feature-Weighted Linear Stacking](#feature-weighted-linear-stacking)
8. [Ensemble Selection](#ensemble-selection)
9. [Advanced Ensemble Techniques](#advanced-ensemble-techniques)
10. [Best Practices](#best-practices)

## Introduction to Ensemble Learning

### What are Ensemble Methods?
Ensemble methods combine multiple machine learning models to create a more powerful model.

### Why Use Ensembles?
- **Reduced Variance**: Averages out biases
- **Improved Generalization**: Combines different model strengths
- **Better Performance**: Often outperforms individual models
- **Robustness**: Less sensitive to noise and outliers

### Types of Ensemble Methods
1. **Averaging Methods**
   - Bagging
   - Random Forests
   - Model Averaging

2. **Boosting Methods**
   - AdaBoost
   - Gradient Boosting
   - XGBoost, LightGBM, CatBoost

3. **Stacking/Blending**
   - Meta-learning approach
   - Combins multiple models with a meta-model

## Voting Classifiers/Regressors

### How It Works
- **Hard Voting**: Each model votes, majority wins (classification)
- **Soft Voting**: Averages predicted probabilities (classification/regression)

### Example: Voting Classifier
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
model1 = LogisticRegression(random_state=42, max_iter=1000)
model2 = DecisionTreeClassifier(random_state=42)
model3 = SVC(probability=True, random_state=42)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('dt', model2),
        ('svc', model3)
    ],
    voting='soft'  # 'hard' for majority voting
)

# Train and evaluate
voting_clf.fit(X_train, y_train)
print(f"Voting Classifier Accuracy: {voting_clf.score(X_test, y_test):.4f}")

# Compare with individual models
for model in (model1, model2, model3, voting_clf):
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__} Accuracy: {model.score(X_test, y_test):.4f}")
```

## Bagging

### Bootstrap Aggregating (Bagging)
- Trains multiple instances of the same model on different subsets of the data
- Reduces variance and helps avoid overfitting
- Works especially well with high-variance models (e.g., decision trees)

### Example: Bagging Classifier
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create base model
base_model = DecisionTreeClassifier(random_state=42)

# Create bagging classifier
bagging_clf = BaggingClassifier(
    base_estimator=base_model,
    n_estimators=100,
    max_samples=0.8,  # 80% of samples for each estimator
    max_features=0.8,  # 80% of features for each estimator
    bootstrap=True,    # Sample with replacement
    bootstrap_features=False,  # Don't sample features with replacement
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Train and evaluate
bagging_clf.fit(X_train, y_train)
print(f"Bagging Classifier Accuracy: {bagging_clf.score(X_test, y_test):.4f}")
```

### Random Forest
- Special case of bagging with decision trees
- Additional randomness in feature selection
- Built-in feature importance

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Number of features to consider at each split
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_clf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf_clf.score(X_test, y_test):.4f}")

# Feature importance
feature_importances = pd.DataFrame(
    rf_clf.feature_importances_,
    index=feature_names if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])],
    columns=['importance']
).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)
```

## Boosting

### How Boosting Works
1. Train a sequence of weak learners
2. Each new model focuses on the mistakes of the previous ones
3. Combine all predictions with weights

### AdaBoost (Adaptive Boosting)
```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Decision Stump
    n_estimators=100,
    learning_rate=0.1,
    algorithm='SAMME.R',
    random_state=42
)

adaboost.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {adaboost.score(X_test, y_test):.4f}")
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,  # Fraction of samples to use for fitting
    max_features=None,  # Number of features to consider for each split
    random_state=42,
    verbose=1
)

gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {gb_clf.score(X_test, y_test):.4f}")
```

### XGBoost (Extreme Gradient Boosting)
```python
import xgboost as xgb

# Create DMatrix (optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': len(np.unique(y)),
    'max_depth': 6,
    'eta': 0.1,  # learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'seed': 42
}

# Train model
num_round = 100
xgb_model = xgb.train(
    params,
    dtrain,
    num_round,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Make predictions
y_pred = xgb_model.predict(dtest)
accuracy = (y_pred == y_test).mean()
print(f"XGBoost Accuracy: {accuracy:.4f}")
```

### LightGBM (Light Gradient Boosting Machine)
```python
import lightgbm as lgb

# Create Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 42
}

# Train
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(10)
    ]
)

# Predict
y_pred = np.argmax(lgb_model.predict(X_test), axis=1)
accuracy = (y_pred == y_test).mean()
print(f"LightGBM Accuracy: {accuracy:.4f}")
```

### CatBoost (Categorical Boosting)
```python
from catboost import CatBoostClassifier, Pool

# Create Pool (optimized data structure for CatBoost)
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# Initialize CatBoost
catboost = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=10
)

# Train model
catboost.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=10
)

# Evaluate
accuracy = catboost.score(X_test, y_test)
print(f"CatBoost Accuracy: {accuracy:.4f}")
```

## Stacking

### What is Stacking?
- Combines multiple models via a meta-model
- Base models make predictions
- Meta-model learns to combine these predictions

### Example: Stacking Classifier
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', lgb.LGBMClassifier(random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='auto',
    n_jobs=-1,
    passthrough=False,  # Whether to use original features along with predictions
    cv=5  # Number of cross-validation folds
)

# Train and evaluate
stacking_clf.fit(X_train, y_train)
print(f"Stacking Classifier Accuracy: {stacking_clf.score(X_test, y_test):.4f}")

# Access base models and meta-model
print("\nBase models:", [name for name, _ in stacking_clf.named_estimators_.items()])
print("Meta-model:", stacking_clf.final_estimator_.__class__.__name__)
```

## Blending

### What is Blending?
- Similar to stacking but with a holdout set
- Split training data into training and holdout sets
- Train base models on training set
- Make predictions on holdout set
- Train meta-model on these predictions

### Example: Blending Classifier
```python
def train_blending_ensemble(X_train, y_train, X_val, y_val, X_test, base_models, meta_model):
    # Train base models on training data
    base_predictions_train = []
    base_predictions_val = []
    
    for name, model in base_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Get predictions for validation set (for meta-model training)
        val_preds = model.predict_proba(X_val)
        base_predictions_val.append(val_preds)
        
        # Get predictions for test set (for final prediction)
        test_preds = model.predict_proba(X_test)
        base_predictions_train.append(test_preds)
    
    # Stack predictions for meta-model
    X_meta_val = np.hstack(base_predictions_val)
    X_meta_test = np.hstack(base_predictions_train)
    
    # Train meta-model on validation predictions
    print("\nTraining meta-model...")
    meta_model.fit(X_meta_val, y_val)
    
    # Make final predictions
    final_predictions = meta_model.predict(X_meta_test)
    
    return final_predictions, meta_model, base_models

# Example usage
from sklearn.model_selection import train_test_split

# Split data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Define base models
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgb': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'lgbm': lgb.LGBMClassifier(random_state=42)
}

# Define meta-model
meta_model = LogisticRegression()

# Train blending ensemble
y_pred, meta_model, base_models = train_blending_ensemble(
    X_train, y_train, X_val, y_val, X_test,
    base_models, meta_model
)

# Evaluate
accuracy = (y_pred == y_test).mean()
print(f"Blending Ensemble Accuracy: {accuracy:.4f}")
```

## Feature-Weighted Linear Stacking (FWLS)

### What is FWLS?
- Extension of stacking that includes original features
- Meta-model uses both base model predictions and original features
- Can capture interactions between features and model predictions

### Example: Custom FWLS Implementation
```python
from sklearn.base import BaseEstimator, ClassifierMixin

class FeatureWeightedLinearStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model, use_original_features=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_original_features = use_original_features
    
    def fit(self, X, y):
        # Split data for meta-model training
        X_train_meta = []
        
        # Get predictions from each base model
        for name, model in self.base_models.items():
            # Clone the model to ensure a fresh fit
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.base_models[name] = model_clone
            
            # Get predictions (probabilities for each class)
            preds = model_clone.predict_proba(X)
            X_train_meta.append(preds)
        
        # Stack predictions
        X_meta = np.hstack(X_train_meta)
        
        # Optionally include original features
        if self.use_original_features:
            X_meta = np.hstack([X_meta, X])
        
        # Train meta-model
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(X_meta, y)
        
        return self
    
    def predict(self, X):
        # Get predictions from base models
        X_meta = []
        
        for name, model in self.base_models.items():
            preds = model.predict_proba(X)
            X_meta.append(preds)
        
        # Stack predictions
        X_meta = np.hstack(X_meta)
        
        # Optionally include original features
        if self.use_original_features:
            X_meta = np.hstack([X_meta, X])
        
        # Make final prediction
        return self.meta_model_.predict(X_meta)
    
    def predict_proba(self, X):
        # Similar to predict but returns probabilities
        X_meta = []
        
        for name, model in self.base_models.items():
            preds = model.predict_proba(X)
            X_meta.append(preds)
        
        X_meta = np.hstack(X_meta)
        
        if self.use_original_features:
            X_meta = np.hstack([X_meta, X])
        
        return self.meta_model_.predict_proba(X_meta)

# Example usage
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgb': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'lgbm': lgb.LGBMClassifier(random_state=42)
}

# Define meta-model
meta_model = LogisticRegression()

# Create and train FWLS
fwls = FeatureWeightedLinearStacking(
    base_models=base_models,
    meta_model=meta_model,
    use_original_features=True
)

fwls.fit(X_train, y_train)

# Evaluate
accuracy = fwls.score(X_test, y_test)
print(f"FWLS Accuracy: {accuracy:.4f}")
```

## Ensemble Selection

### What is Ensemble Selection?
- Greedy algorithm to select the best subset of models
- Iteratively adds models that improve ensemble performance
- Can use replacement (same model can be added multiple times)

### Example: Ensemble Selection
```python
from sklearn.metrics import accuracy_score
import numpy as np

def ensemble_selection(models, X_val, y_val, X_test, metric=accuracy_score, max_models=10):
    """
    Greedy ensemble selection algorithm
    """
    # Get predictions from all models on validation set
    val_preds = []
    test_preds = []
    
    for name, model in models.items():
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_preds.append(val_pred)
        test_preds.append(test_pred)
    
    val_preds = np.array(val_preds)
    test_preds = np.array(test_preds)
    
    # Initialize ensemble
    ensemble_indices = []
    best_score = -np.inf
    best_ensemble = None
    
    # Greedy selection
    for _ in range(max_models):
        scores = []
        
        # Try adding each model
        for i in range(len(models)):
            # Create new ensemble with this model added
            current_indices = ensemble_indices + [i]
            ensemble_pred = np.mean(val_preds[current_indices], axis=0)
            
            # For classification, take the majority vote
            if len(ensemble_pred.shape) > 1:  # Probabilities
                ensemble_pred = np.argmax(ensemble_pred, axis=1)
            else:  # Class labels
                ensemble_pred = np.round(ensemble_pred).astype(int)
            
            # Calculate score
            score = metric(y_val, ensemble_pred)
            scores.append(score)
        
        # Find best model to add
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Add to ensemble
        ensemble_indices.append(best_idx)
        
        # Early stopping if no improvement
        if len(ensemble_indices) > 1 and scores[best_idx] <= best_score:
            break
    
    # Create final ensemble predictions
    final_ensemble_pred = np.mean(test_preds[ensemble_indices], axis=0)
    
    if len(final_ensemble_pred.shape) > 1:  # Probabilities
        final_ensemble_pred = np.argmax(final_ensemble_pred, axis=1)
    else:  # Class labels
        final_ensemble_pred = np.round(final_ensemble_pred).astype(int)
    
    return final_ensemble_pred, ensemble_indices

# Example usage
models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
    'xgb': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train),
    'lgbm': lgb.LGBMClassifier(random_state=42).fit(X_train, y_train),
    'logreg': LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train),
    'svm': SVC(probability=True, random_state=42).fit(X_train, y_train)
}

# Split into train and validation sets
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Retrain models on training set
for name, model in models.items():
    model.fit(X_train_es, y_train_es)

# Run ensemble selection
y_pred_es, selected_indices = ensemble_selection(
    models, X_val_es, y_val_es, X_test,
    metric=accuracy_score, max_models=5
)

# Evaluate
accuracy = accuracy_score(y_test, y_pred_es)
print(f"Ensemble Selection Accuracy: {accuracy:.4f}")
print(f"Selected models: {[list(models.keys())[i] for i in selected_indices]}")
```

## Advanced Ensemble Techniques

### Super Learner
- Generalization of stacking
- Uses cross-validation to create out-of-fold predictions
- More robust than simple stacking

### Bayesian Model Combination
- Uses Bayesian methods to combine models
- Accounts for model uncertainty
- Can provide better uncertainty estimates

### Stacked Generalization with Feature Engineering
- Create new features from model predictions
- Combine with original features
- Can capture complex interactions

### Example: Super Learner
```python
from sklearn.model_selection import KFold

def super_learner(X, y, base_models, meta_model, n_splits=5):
    """
    Implements the Super Learner algorithm
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize out-of-fold predictions
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    n_models = len(base_models)
    
    # For classification, we'll use class probabilities
    X_meta = np.zeros((n_samples, n_models * n_classes))
    
    # For each base model
    for i, (name, model) in enumerate(base_models.items()):
        print(f"Training {name} with {n_splits}-fold CV...")
        
        # For each fold
        for train_idx, val_idx in kf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            # Train model on training fold
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Get predictions on validation fold
            preds = model_clone.predict_proba(X_val_fold)
            X_meta[val_idx, i*n_classes:(i+1)*n_classes] = preds
    
    # Train meta-model on out-of-fold predictions
    print("\nTraining meta-model...")
    meta_model.fit(X_meta, y)
    
    # Train all base models on full training data
    print("\nTraining base models on full data...")
    trained_base_models = {}
    for name, model in base_models.items():
        model_clone = clone(model)
        model_clone.fit(X, y)
        trained_base_models[name] = model_clone
    
    return trained_base_models, meta_model

def super_learner_predict(X, base_models, meta_model):
    """
    Make predictions using the Super Learner
    """
    # Get predictions from all base models
    base_preds = []
    
    for name, model in base_models.items():
        preds = model.predict_proba(X)
        base_preds.append(preds)
    
    # Stack predictions
    X_meta = np.hstack(base_preds)
    
    # Get final predictions from meta-model
    return meta_model.predict(X_meta)

# Example usage
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgb': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'lgbm': lgb.LGBMClassifier(random_state=42),
    'logreg': LogisticRegression(max_iter=1000, random_state=42)
}

# Define meta-model
meta_model = LogisticRegression()

# Train Super Learner
trained_base_models, trained_meta_model = super_learner(
    X_train, y_train, base_models, meta_model, n_splits=5
)

# Make predictions
y_pred_sl = super_learner_predict(X_test, trained_base_models, trained_meta_model)

# Evaluate
accuracy = accuracy_score(y_test, y_pred_sl)
print(f"Super Learner Accuracy: {accuracy:.4f}")
```

## Best Practices

1. **Diversity is Key**
   - Use models with different learning algorithms
   - Train on different subsets of features or samples
   - Use different hyperparameter settings

2. **Start Simple**
   - Begin with simple averaging or voting
   - Progress to more complex methods if needed
   - Always compare against individual models

3. **Avoid Overfitting**
   - Use cross-validation for meta-model training
   - Keep the meta-model simple
   - Use a holdout set for final evaluation

4. **Consider Computational Cost**
   - More complex ensembles require more resources
   - Balance performance gains with computational cost
   - Consider model inference time in production

5. **Monitor Performance**
   - Track performance of individual models and ensemble
   - Watch for model decay over time
   - Set up retraining pipelines

6. **Interpretability**
   - Simple ensembles are more interpretable
   - Consider model-agnostic interpretation methods
   - Document the ensemble construction process

## Practice Exercises
1. **Basic Ensembles**
   - Implement a voting classifier with 3 different models
   - Compare performance against individual models
   - Experiment with hard vs. soft voting

2. **Bagging and Boosting**
   - Compare Random Forest with Gradient Boosting
   - Tune hyperparameters for both approaches
   - Analyze feature importances

3. **Stacking Implementation**
   - Implement a simple stacking classifier from scratch
   - Compare with scikit-learn's StackingClassifier
   - Experiment with different meta-models

4. **Advanced Techniques**
   - Implement the Super Learner algorithm
   - Compare with simpler ensemble methods
   - Analyze the diversity of base models

5. **Real-world Application**
   - Apply ensemble methods to a Kaggle competition
   - Experiment with different ensemble strategies
   - Document performance improvements

---
Next: [Deep Learning Fundamentals](./05_deep_learning_fundamentals.md)
