"""Main training script with model persistence"""
import encodings
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from data_loader import load_data
from preprocessing import preprocess_data
from models import (
    train_random_forest, train_xgboost, train_logistic_regression, 
    train_svm, evaluate_model, compare_models, cross_validate_models, 
    get_feature_importance
)

# Create models directory if it doesn't exist
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def save_model(model, model_name):
    """Save model using joblib"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Saved {model_name} to {model_path}")
    return model_path


def save_scaler(scaler):
    """Save the StandardScaler"""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")
    return scaler_path


def save_feature_names(feature_names):
    """Save feature names for prediction"""
    features_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(list(feature_names), f)
    print(f"✓ Saved feature names to {features_path}")
    return features_path


def save_encodings(bmi_map, disorder_map):
    """Save label encodings"""
    encodings_path = os.path.join(MODELS_DIR, "encodings.json")
    encodings = {
        'bmi_category': {k: int(v) for k, v in bmi_map.items()},
        'sleep_disorder': {k: int(v) for k, v in disorder_map.items()}
    }
    with open(encodings_path, 'w') as f:
        json.dump(encodings, f)
    print(f"✓ Saved encodings to {encodings_path}")
    return encodings_path


def main():
    """Main training pipeline with model saving"""
    # Load data
    print("Loading data...")
    url = "https://raw.githubusercontent.com/monishm-2004/sleep-disorder-predicition/refs/heads/main/Sleep_health_and_lifestyle_dataset.csv"
    df = load_data(url=url)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Preprocessing
    print("Preprocessing data...")
    df, bmi_map, disorder_map = preprocess_data(df)
    print(f"Data preprocessed: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Save encodings
    print("\n" + "="*50)
    print("Saving Encodings and Metadata")
    print("="*50)
    save_encodings(bmi_map, disorder_map)
    
    # Basic statistics
    print("\n" + "="*50)
    print("Basic Statistics")
    print("="*50)

    
    # Train-test split
    print("\n" + "="*50)
    print("Splitting data...")
    print("="*50)
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    save_scaler(scaler)
    save_feature_names(X.columns)
    
    # Train models
    print("\n" + "="*50)
    print("Training Models")
    print("="*50)
    
    print("\n---- Random Forest ----")
    rf = train_random_forest(X_train_scaled, y_train)
    rf_acc = evaluate_model(rf, X_test_scaled, y_test)
    
    print("\n---- XGBoost ----")
    xgb = train_xgboost(X_train_scaled, y_train)
    xgb_acc = evaluate_model(xgb, X_test_scaled, y_test)
    
    print("\n---- Logistic Regression ----")
    logr = train_logistic_regression(X_train_scaled, y_train)
    logr_acc = evaluate_model(logr, X_test_scaled, y_test)
    
    print("\n---- SVM ----")
    svm = train_svm(X_train_scaled, y_train)
    svm_acc = evaluate_model(svm, X_test_scaled, y_test)
    
    # Compare models
    models = {
        'Random Forest': rf,
        'XGBoost': xgb,
        'Logistic Regression': logr,
        'SVM': svm
    }
    compare_models(models, X_test_scaled, y_test)
    
    # Cross-validation
    models_for_cv = {
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
        "Random Forest": train_random_forest(X_train_scaled, y_train),
        "Logistic Regression": make_pipeline(StandardScaler(), train_logistic_regression(X_train_scaled, y_train)),
        "SVM": make_pipeline(StandardScaler(), train_svm(X_train_scaled, y_train))
    }
    cross_validate_models(models_for_cv, X, y)
    
    # Feature importance
    print("\nFeature Importance (Logistic Regression):")
    feature_imp = get_feature_importance(logr, X.columns)
    print(feature_imp)
    
    # Save all models
    print("\n" + "="*50)
    print("Saving Models")
    print("="*50)
    save_model(rf, "random_forest")
    save_model(xgb, "xgboost")
    save_model(logr, "logistic_regression")
    save_model(svm, "svm")
    
    # Save model metadata
    metadata = {
        "random_forest_accuracy": round(rf_acc, 4),
        "xgboost_accuracy": round(xgb_acc, 4),
        "logistic_regression_accuracy": round(logr_acc, 4),
        "svm_accuracy": round(svm_acc, 4),
        "best_model": max(models.items(), key=lambda x: evaluate_model(x[1], X_test_scaled, y_test))[0],
        "feature_count": X.shape[1],
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0]
    }
    
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nModels saved in '{MODELS_DIR}/' directory:")
    print("  - random_forest.pkl")
    print("  - xgboost.pkl")
    print("  - logistic_regression.pkl")
    print("  - svm.pkl")
    print("  - scaler.pkl")
    print("  - feature_names.json")
    print("  - encodings.json")
    print("  - metadata.json")


if __name__ == "__main__":
    main()