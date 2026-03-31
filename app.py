"""FastAPI application for Sleep Disorder Prediction"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prediction import initialize_models, get_models

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Disorder Prediction API",
    description="API for predicting sleep disorders based on health and lifestyle data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    age: int = Field(..., ge=0, le=100, description="Age of person")
    sleep_duration: float = Field(..., ge=0, description="Sleep duration in hours")
    quality_of_sleep: int = Field(..., ge=0, le=10, description="Sleep quality (0-10)")
    physical_activity_level: int = Field(..., ge=0, description="Physical activity level")
    stress_level: int = Field(..., ge=0, le=10, description="Stress level (0-10)")
    heart_rate: int = Field(..., ge=0, description="Heart rate (bpm)")
    daily_steps: int = Field(..., ge=0, description="Daily steps")
    bp_high: int = Field(..., ge=0, description="Systolic blood pressure")
    bp_low: int = Field(..., ge=0, description="Diastolic blood pressure")
    bmi_category: int = Field(..., ge=0, description="BMI category (0=Normal, 1=Overweight, 2=Obese)")
    gender_female: int = Field(default=0, ge=0, le=1, description="Gender: Female (1) or Male (0)")
    gender_male: int = Field(default=0, ge=0, le=1, description="Gender: Male (1) or Female (0)")
    
    # Occupations (one-hot encoded)
    occupation_accountant: int = Field(default=0, ge=0, le=1)
    occupation_doctor: int = Field(default=0, ge=0, le=1)
    occupation_engineer: int = Field(default=0, ge=0, le=1)
    occupation_lawyer: int = Field(default=0, ge=0, le=1)
    occupation_manager: int = Field(default=0, ge=0, le=1)
    occupation_nurse: int = Field(default=0, ge=0, le=1)
    occupation_sales_person: int = Field(default=0, ge=0, le=1)
    occupation_teacher: int = Field(default=0, ge=0, le=1)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "occupation_engineer": 1
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    model_used: str
    confidence: Optional[float]
    probabilities: Dict[str, float]


class HealthCheckResponse(BaseModel):
    """Response for health check"""
    status: str
    models_loaded: bool
    available_models: list


# Events
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    initialize_models(models_dir)
    print("✓ Models initialized on startup")


# Routes
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check if API is running and models are loaded"""
    model_loader = get_models()
    models_loaded = model_loader is not None and len(model_loader.models) > 0
    
    return HealthCheckResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        available_models=list(model_loader.models.keys()) if model_loader else []
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the best model (XGBoost)"""
    model_loader = get_models()
    
    if not model_loader or len(model_loader.models) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Convert request to features dict
    features = request.model_dump()
    
    # Make prediction
    result = model_loader.predict(features, model_name="xgboost")
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return PredictionResponse(
        prediction=result["prediction"],
        model_used=result["model"],
        confidence=result.get("confidence"),
        probabilities=result.get("probabilities", {})
    )


@app.post("/predict-all")
async def predict_all(request: PredictionRequest):
    """Make predictions using all available models"""
    model_loader = get_models()
    
    if not model_loader or len(model_loader.models) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Convert request to features dict
    features = request.model_dump()
    
    # Get predictions from all models
    results = model_loader.predict_all_models(features)
    
    return {
        "predictions": results,
        "model_metadata": model_loader.metadata
    }


@app.get("/models")
async def get_model_info():
    """Get information about available models"""
    model_loader = get_models()
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "available_models": list(model_loader.models.keys()),
        "best_model": model_loader.metadata.get("best_model"),
        "model_accuracies": {
            "xgboost": model_loader.metadata.get("xgboost_accuracy"),
            "random_forest": model_loader.metadata.get("random_forest_accuracy"),
            "logistic_regression": model_loader.metadata.get("logistic_regression_accuracy"),
            "svm": model_loader.metadata.get("svm_accuracy")
        },
        "feature_names": model_loader.feature_names,
        "training_samples": model_loader.metadata.get("training_samples"),
        "test_samples": model_loader.metadata.get("test_samples")
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sleep Disorder Prediction API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
