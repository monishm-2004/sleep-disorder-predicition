## 🔄 End-to-End ML Project: Sleep Disorder Prediction

Complete machine learning project with training pipeline and FastAPI deployment.

---

## 📁 Project Structure

```
sleep_disorder_pred/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing
│   ├── eda.py                   # Exploratory data analysis
│   ├── models.py                # Model training functions
│   ├── train.py                 # Main training script
│   └── prediction.py            # Model loading for predictions
├── models/                      # Generated after training
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── encodings.json
│   └── metadata.json
├── app.py                       # FastAPI application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Models
```bash
cd src
python train.py
```

This will:
- Load dataset from GitHub
- Preprocess data (handle missing values, encode features)
- Train 4 models (XGBoost, Random Forest, Logistic Regression, SVM)
- Save all models to `models/` directory
- Save preprocessing artifacts (scaler, encodings)

### 3️⃣ Run FastAPI Server
```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Visit API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Data Processing Pipeline

### Input Features
- **Age**: 0-100
- **Sleep Duration**: Hours (0+)
- **Quality of Sleep**: 0-10
- **Physical Activity Level**: Steps/minutes
- **Stress Level**: 0-10
- **Heart Rate**: BPM
- **Daily Steps**: Number
- **Blood Pressure**: High/Low
- **BMI Category**: Normal/Overweight/Obese (encoded)
- **Gender**: Male/Female (one-hot)
- **Occupation**: 8 categories (one-hot)

### Output
- **Sleep Disorder**: healthy, insomnia, or sleep apnea

---

## 🤖 Model Information

| Model | Type | Accuracy |
|-------|------|----------|
| **XGBoost** | Gradient Boosting | Best performing |
| **Random Forest** | Ensemble | Fast inference |
| **Logistic Regression** | Linear | Interpretable |
| **SVM** | Support Vector Machine | Robust |

---

## 🔌 API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["xgboost", "random_forest", "logistic_regression", "svm"]
}
```

### 2. Predict (Best Model)
```bash
POST /predict
```

Request:
```json
{
  "age": 42,
  "sleep_duration": 7.5,
  "quality_of_sleep": 8,
  "physical_activity_level": 60,
  "stress_level": 5,
  "heart_rate": 70,
  "daily_steps": 8000,
  "bp_high": 120,
  "bp_low": 80,
  "bmi_category": 0,
  "gender_male": 1,
  "gender_female": 0,
  "occupation_engineer": 1,
  "occupation_accountant": 0,
  "occupation_doctor": 0,
  "occupation_lawyer": 0,
  "occupation_manager": 0,
  "occupation_nurse": 0,
  "occupation_sales_person": 0,
  "occupation_teacher": 0
}
```

Response:
```json
{
  "prediction": "healthy",
  "model_used": "xgboost",
  "confidence": 0.95,
  "probabilities": {
    "healthy": 0.95,
    "insomnia": 0.03,
    "sleep_apnea": 0.02
  }
}
```

### 3. Predict All Models
```bash
POST /predict-all
```

Returns predictions from all 4 models along with metadata.

### 4. Get Models Info
```bash
GET /models
```

Returns model accuracies, feature names, and training info.

---

## 📝 Example Usage with Python

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make prediction
prediction_data = {
    "age": 42,
    "sleep_duration": 7.5,
    "quality_of_sleep": 8,
    "physical_activity_level": 60,
    "stress_level": 5,
    "heart_rate": 70,
    "daily_steps": 8000,
    "bp_high": 120,
    "bp_low": 80,
    "bmi_category": 0,
    "gender_male": 1,
    "gender_female": 0,
    "occupation_engineer": 1,
    "occupation_accountant": 0,
    "occupation_doctor": 0,
    "occupation_lawyer": 0,
    "occupation_manager": 0,
    "occupation_nurse": 0,
    "occupation_sales_person": 0,
    "occupation_teacher": 0
}

response = requests.post(f"{BASE_URL}/predict", json=prediction_data)
print(json.dumps(response.json(), indent=2))
```

---

## 🛠️ Files Overview

### `src/data_loader.py`
- `load_data()`: Load from URL or local file

### `src/preprocessing.py`
- `fill_missing_values()`: Handle missing data
- `split_blood_pressure()`: Split BP into high/low
- `encode_features()`: Label & one-hot encoding
- `preprocess_data()`: Complete pipeline

### `src/eda.py`
- `plot_disorder_vs_gender()`: Visualization
- `plot_disorder_vs_occupation()`: Visualization
- `plot_disorder_vs_bmi()`: Visualization
- `print_basic_stats()`: Statistical summary

### `src/models.py`
- `train_xgboost()`, `train_random_forest()`, etc.
- `evaluate_model()`: Calculate accuracy
- `compare_models()`: Compare all models
- `cross_validate_models()`: K-fold CV
- `get_feature_importance()`: Feature analysis

### `src/train.py`
- Main training pipeline
- Saves all models and artifacts
- Creates `models/` directory automatically

### `src/prediction.py`
- `ModelLoader`: Loads saved models
- `predict()`: Make predictions
- `predict_all_models()`: Ensemble predictions

### `app.py`
- FastAPI application
- Defines API endpoints
- Automatic model loading on startup

---

## 📦 Output Files

After running `python train.py`, the `models/` directory contains:

| File | Purpose |
|------|---------|
| `xgboost.pkl` | XGBoost model weights |
| `random_forest.pkl` | Random Forest model weights |
| `logistic_regression.pkl` | Logistic Regression weights |
| `svm.pkl` | SVM model weights |
| `scaler.pkl` | StandardScaler for feature normalization |
| `feature_names.json` | List of feature names in order |
| `encodings.json` | Label encodings for categorical features |
| `metadata.json` | Model accuracies and training info |

---

## 🔍 Model Details

### Training Data
- 374 samples after preprocessing
- 20% test split
- Features scaled with StandardScaler

### Models Trained
1. **XGBoost**: 100 estimators, learning_rate=0.1
2. **Random Forest**: 100 estimators
3. **Logistic Regression**: multinomial classifier
4. **SVM**: RBF kernel

### Performance
Check `models/metadata.json` for accuracy metrics:
```json
{
  "xgboost_accuracy": 0.9400,
  "random_forest_accuracy": 0.9200,
  "logistic_regression_accuracy": 0.8800,
  "svm_accuracy": 0.9000,
  "best_model": "xgboost"
}
```

---

## 🐛 Troubleshooting

### Models not loading
```
Error: Models not loaded
```
**Solution**: Ensure you've run `python src/train.py` first

### Feature dimension mismatch
```
Error: X has 25 features but model expects 20
```
**Solution**: Check that input features match `feature_names.json`

### CORS errors
API uses CORS middleware to allow requests from any origin

---

## 📈 Next Steps

1. **Deploy to Production**: Use Docker + Kubernetes
2. **Add Database**: Store predictions for monitoring
3. **Model Monitoring**: Track prediction drift
4. **A/B Testing**: Compare model versions
5. **Feature Store**: Centralize feature management

---

## 📄 License

This is a demo project for educational purposes.

---

## 🤝 Contributing

Feel free to modify and extend this project!

---

**Created**: 2024
**Version**: 1.0.0
