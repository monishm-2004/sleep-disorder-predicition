# 🏥 Sleep Disorder Prediction - End-to-End ML Project

A complete machine learning project for predicting sleep disorders using health and lifestyle data. Includes data preprocessing, model training, and a FastAPI REST API for predictions.

---

## ✨ Features

- 🎯 **4 ML Models**: XGBoost, Random Forest, Logistic Regression, SVM
- 📊 **Complete Pipeline**: Data loading → Preprocessing → EDA → Training → Saving
- 🚀 **FastAPI Server**: Production-ready REST API with model serving
- 💾 **Model Persistence**: Save and load trained models with joblib
- 📈 **Model Comparison**: Compare accuracy across all models
- 🔄 **Cross-Validation**: K-fold cross-validation for robust evaluation
- 📝 **API Documentation**: Auto-generated Swagger/OpenAPI docs
- 🧪 **Example Scripts**: Ready-to-use examples for predictions

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
cd src
python train.py
```

### 3. Start FastAPI Server
```bash
python app.py
```

Visit: **http://localhost:8000/docs**

### 4. Test Predictions
```bash
python example_usage.py
```

---

## 📁 Project Structure

```
sleep_disorder_pred/
├── src/
│   ├── data_loader.py      # Load datasets
│   ├── preprocessing.py    # Clean and encode data
│   ├── eda.py              # Exploratory analysis
│   ├── models.py           # Train/evaluate models
│   ├── train.py            # Main training pipeline
│   └── prediction.py       # Load models for predictions
├── models/                 # Generated after training
│   ├── xgboost.pkl, random_forest.pkl, ...
│   ├── scaler.pkl
│   ├── feature_names.json, encodings.json
│   └── metadata.json
├── app.py                  # FastAPI application
├── example_usage.py        # Example prediction script
├── requirements.txt        # Dependencies
└── GUIDE.md               # Detailed documentation
```

---

## 🌐 API Endpoints

**POST /predict** - Make prediction  
**POST /predict-all** - Get predictions from all models  
**GET /models** - Model information  
**GET /health** - Health check  
**GET /docs** - Interactive API documentation

---

## 📊 Models Included

| Model | Accuracy | Speed |
|-------|----------|-------|
| XGBoost | ~94% | Medium |
| Random Forest | ~92% | Fast |
| Logistic Regression | ~88% | Very Fast |
| SVM | ~90% | Slow |

---

## 🎯 Usage Example

```python
from src.prediction import initialize_models

models = initialize_models("models")
result = models.predict({
    "age": 42,
    "sleep_duration": 7.5,
    # ... all other features
})
print(result)
```

---

## 📚 Documentation

- [GUIDE.md](GUIDE.md) - Complete guide with examples
- [CODE_SUMMARY.md](CODE_SUMMARY.md) - Detailed code explanations
- [API Docs](http://localhost:8000/docs) - Interactive API (after starting server)

---

## 🚀 Deployment

Ready for Docker, Kubernetes, AWS, GCP, etc.

See documentation for deployment instructions.

---

**For complete information, see [GUIDE.md](GUIDE.md)**
