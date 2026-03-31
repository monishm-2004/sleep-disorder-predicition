"""Example script for using trained models"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prediction import initialize_models

# Example 1: Initialize and load models
print("=" * 60)
print("Loading Models")
print("=" * 60)

models = initialize_models("models")

# Example 2: Make a prediction with a single model
print("\n" + "=" * 60)
print("Making Predictions")
print("=" * 60)

# Sample person's data
sample_person = {
    "age": 42,
    "sleep_duration": 7.5,
    "quality_of_sleep": 8,
    "physical_activity_level": 60,
    "stress_level": 5,
    "heart_rate": 70,
    "daily_steps": 8000,
    "bp_high": 120,
    "bp_low": 80,
    "bmi_category": 0,  # Normal weight
    "gender_male": 1,
    "gender_female": 0,
    # Occupations (one-hot encoded - only engineer is 1)
    "occupation_accountant": 0,
    "occupation_doctor": 0,
    "occupation_engineer": 1,
    "occupation_lawyer": 0,
    "occupation_manager": 0,
    "occupation_nurse": 0,
    "occupation_sales_person": 0,
    "occupation_teacher": 0,
}

# Predict with XGBoost
print("\n1. XGBoost Prediction:")
result = models.predict(sample_person, model_name="xgboost")
print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.2%}")
print(f"   Probabilities: {result['probabilities']}")

# Predict with Random Forest
print("\n2. Random Forest Prediction:")
result = models.predict(sample_person, model_name="random_forest")
print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.2%}")

# Predict with Logistic Regression
print("\n3. Logistic Regression Prediction:")
result = models.predict(sample_person, model_name="logistic_regression")
print(f"   Prediction: {result['prediction']}")
print(f"   Confidence: {result['confidence']:.2%}")

# Example 3: Get predictions from all models
print("\n" + "=" * 60)
print("All Models Predictions")
print("=" * 60)

all_results = models.predict_all_models(sample_person)
for model_name, result in all_results.items():
    if "error" not in result:
        print(f"\n{model_name.upper()}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")

# Example 4: Get detailed explanation
print("\n" + "=" * 60)
print("Model Information")
print("=" * 60)

print(f"\nAvailable Models: {list(models.models.keys())}")
print(f"Number of Features: {len(models.feature_names)}")
print(f"Feature Names: {models.feature_names}")

print(f"\nModel Accuracies:")
print(f"  XGBoost: {models.metadata['xgboost_accuracy']:.2%}")
print(f"  Random Forest: {models.metadata['random_forest_accuracy']:.2%}")
print(f"  Logistic Regression: {models.metadata['logistic_regression_accuracy']:.2%}")
print(f"  SVM: {models.metadata['svm_accuracy']:.2%}")

print(f"\nBest Model: {models.metadata['best_model']}")

# Example 5: Test with different scenarios
print("\n" + "=" * 60)
print("Testing Different Scenarios")
print("=" * 60)

scenarios = {
    "Healthy Person": {
        "age": 30,
        "sleep_duration": 8.0,
        "quality_of_sleep": 9,
        "physical_activity_level": 100,
        "stress_level": 2,
        "heart_rate": 60,
        "daily_steps": 10000,
        "bp_high": 110,
        "bp_low": 70,
        "bmi_category": 0,
        "gender_male": 1,
        "gender_female": 0,
        "occupation_teacher": 1,
        "occupation_accountant": 0,
        "occupation_doctor": 0,
        "occupation_engineer": 0,
        "occupation_lawyer": 0,
        "occupation_manager": 0,
        "occupation_nurse": 0,
        "occupation_sales_person": 0,
    },
    "Stressed Person": {
        "age": 50,
        "sleep_duration": 5.5,
        "quality_of_sleep": 4,
        "physical_activity_level": 20,
        "stress_level": 9,
        "heart_rate": 95,
        "daily_steps": 3000,
        "bp_high": 145,
        "bp_low": 95,
        "bmi_category": 1,
        "gender_male": 0,
        "gender_female": 1,
        "occupation_nurse": 1,
        "occupation_accountant": 0,
        "occupation_doctor": 0,
        "occupation_engineer": 0,
        "occupation_lawyer": 0,
        "occupation_manager": 0,
        "occupation_sales_person": 0,
        "occupation_teacher": 0,
    },
    "Overweight Person": {
        "age": 55,
        "sleep_duration": 6.0,
        "quality_of_sleep": 5,
        "physical_activity_level": 30,
        "stress_level": 7,
        "heart_rate": 85,
        "daily_steps": 5000,
        "bp_high": 135,
        "bp_low": 85,
        "bmi_category": 2,
        "gender_male": 1,
        "gender_female": 0,
        "occupation_manager": 1,
        "occupation_accountant": 0,
        "occupation_doctor": 0,
        "occupation_engineer": 0,
        "occupation_lawyer": 0,
        "occupation_nurse": 0,
        "occupation_sales_person": 0,
        "occupation_teacher": 0,
    }
}

best_model = models.metadata.get('best_model', 'xgboost')
for scenario_name, scenario_data in scenarios.items():
    result = models.predict(scenario_data, model_name=best_model)
    if "error" not in result:
        print(f"\n{scenario_name}: {result['prediction'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")

print("\n" + "=" * 60)
print("✓ Example script completed successfully!")
print("=" * 60)
