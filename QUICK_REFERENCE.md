**# 🚀 QUICK REFERENCE - Sleep Disorder Prediction Project**

---

## ⚡ Essential Commands

### Setup
```bash
pip install -r requirements.txt
```

### Train Models (saves to models/)
```bash
cd src
python train.py
```

### Run API Server
```bash
python app.py
```

### Test Script
```bash
python example_usage.py
```

---

## 📝 API Quick Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Test in Browser
Visit: **http://localhost:8000/docs**

---

## 📂 File Locations

| File | Location |
|------|----------|
| Training Script | `src/train.py` |
| API Server | `app.py` |
| Model Loader | `src/prediction.py` |
| Example Usage | `example_usage.py` |
| Models (generated) | `models/` |
| Documentation | `GUIDE.md` |

---

## 🤖 Models Available

```
- xgboost (best ~94% accuracy)
- random_forest (~92%)
- logistic_regression (~88%)
- svm (~90%)
```

---

## 📊 Input Features (required for prediction)

**Personal Info:**
- age (0-100)
- gender_male (0/1)
- gender_female (0/1)

**Sleep Metrics:**
- sleep_duration (hours)
- quality_of_sleep (0-10)

**Activity & Health:**
- physical_activity_level (int)
- stress_level (0-10)
- heart_rate (bpm)
- daily_steps (int)
- bp_high (systolic)
- bp_low (diastolic)
- bmi_category (0/1/2)

**Occupation (one-hot, exactly one = 1):**
- occupation_accountant
- occupation_doctor
- occupation_engineer
- occupation_lawyer
- occupation_manager
- occupation_nurse
- occupation_sales_person
- occupation_teacher

---

## 💾 Files Generated After Training

```
models/
├── xgboost.pkl                          (trained model)
├── random_forest.pkl                    (trained model)
├── logistic_regression.pkl              (trained model)
├── svm.pkl                              (trained model)
├── scaler.pkl                           (feature normalization)
├── feature_names.json                   (ordered feature list)
├── encodings.json                       (category mappings)
└── metadata.json                        (model accuracies)
```

---

## 🔌 Load Models in Python

```python
from src.prediction import initialize_models

models = initialize_models("models")
result = models.predict(your_data_dict, model_name="xgboost")
print(result['prediction'])
```

---

## 🌐 API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /health | Check API status |
| POST | /predict | Get prediction |
| POST | /predict-all | All models prediction |
| GET | /models | Model info |
| GET | /docs | API documentation |

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8000 in use | `uvicorn app:app --port 8001` |
| Models not found | Run `python src/train.py` first |
| Import error | `pip install -r requirements.txt --upgrade` |
| Wrong predictions | Check feature names match `feature_names.json` |

---

## 📊 Expected Accuracy by Model

```
Best (XGBoost):         🥇 94%
Good (Random Forest):   🥈 92%  
Good (SVM):             🥉 90%
Fair (Log Regression):  ℹ️  88%
```

---

## 🎯 Complete Workflow

```
1. Install dependencies
   └─ pip install -r requirements.txt

2. Train models (creates models/ dir)
   └─ cd src && python train.py

3. Start API server
   └─ python app.py

4. Make predictions
   └─ curl or visit /docs

5. Deploy (optional)
   └─ Docker / Kubernetes
```

---

## 📱 Make Prediction from Python

```python
from src.prediction import initialize_models

models = initialize_models("models")

person = {
    "age": 35,
    "sleep_duration": 7.0,
    "quality_of_sleep": 8,
    "physical_activity_level": 70,
    "stress_level": 4,
    "heart_rate": 65,
    "daily_steps": 9000,
    "bp_high": 115,
    "bp_low": 75,
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
    "occupation_teacher": 0,
}

result = models.predict(person, model_name="xgboost")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.0%}")
```

---

## 📚 Documentation Files

- **README.md** - Overview (this file's parent)
- **GUIDE.md** - Complete guide with examples
- **CODE_SUMMARY.md** - Detailed code explanations
- **example_usage.py** - Full working examples

---

## ✅ Project Status

✓ Modularized code in src/  
✓ Model weights saved and loaded  
✓ FastAPI server ready  
✓ Complete documentation  
✓ Example scripts included  
✓ Production ready

---

**Save this file for quick reference!**
