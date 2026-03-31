"""Module for model training and evaluation"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    return xgb

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    logr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    logr.fit(X_train, y_train)
    return logr

def train_svm(X_train, y_train):
    """Train SVM model"""
    svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    return svm

def compare_models(models, X_test, y_test):
    """Compare multiple models"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    
    print("\nModel Accuracy Comparison:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")
    
    return results

def cross_validate_models(models, X, y, cv=5):
    """Perform k-fold cross-validation on models"""
    print("\n5-Fold Cross-Validation Results:")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"{name}: Mean Accuracy = {scores.mean():.4f} | Std = {scores.std():.4f}")

def get_feature_importance(model, feature_names):
    """Get feature importance from Logistic Regression"""
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Abs(Coefficient)': np.abs(model.coef_[0])
    }).sort_values(by='Abs(Coefficient)', ascending=False)
    
    return coefficients