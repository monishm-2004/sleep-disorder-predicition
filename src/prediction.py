"""Module for loading models and making predictions"""
import os
import json
import joblib
import numpy as np
import pandas as pd


class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.encodings = None
        self.metadata = None
        
    def load_all_models(self):
        """Load all saved models and artifacts"""
        try:
            # Load models
            self.models['xgboost'] = joblib.load(os.path.join(self.models_dir, "xgboost.pkl"))
            self.models['random_forest'] = joblib.load(os.path.join(self.models_dir, "random_forest.pkl"))
            self.models['logistic_regression'] = joblib.load(os.path.join(self.models_dir, "logistic_regression.pkl"))
            self.models['svm'] = joblib.load(os.path.join(self.models_dir, "svm.pkl"))
            
            # Load scaler
            self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
            
            # Load feature names
            with open(os.path.join(self.models_dir, "feature_names.json"), 'r') as f:
                self.feature_names = json.load(f)
            
            # Load encodings
            with open(os.path.join(self.models_dir, "encodings.json"), 'r') as f:
                self.encodings = json.load(f)
            
            # Load metadata
            with open(os.path.join(self.models_dir, "metadata.json"), 'r') as f:
                self.metadata = json.load(f)
            
            print("✓ All models loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def predict(self, features, model_name="xgboost"):
        """
        Make prediction using specified model
        
        Args:
            features (dict): Dictionary with feature values
            model_name (str): Name of model to use
        
        Returns:
            dict: Prediction result with confidence
        """
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        try:
            # Convert to DataFrame with correct feature order
            X = pd.DataFrame([features], columns=self.feature_names)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_scaled)[0]
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_scaled)[0]
                # Create reverse mapping for disorder classes
                disorder_key = 'sleep_disorder'
                if disorder_key not in self.encodings:
                    raise ValueError(f"Missing '{disorder_key}' key in encodings. Available keys: {list(self.encodings.keys())}")
                disorder_reverse = {v: k for k, v in self.encodings[disorder_key].items()}
                for idx, prob in enumerate(probs):
                    probabilities[disorder_reverse.get(idx, str(idx))] = float(prob)
            
            # Decode prediction
            disorder_key = 'sleep_disorder'
            if disorder_key not in self.encodings:
                raise ValueError(f"Missing '{disorder_key}' key in encodings. Available keys: {list(self.encodings.keys())}")
            disorder_reverse = {v: k for k, v in self.encodings[disorder_key].items()}
            predicted_disorder = disorder_reverse.get(prediction, str(prediction))
            
            return {
                "prediction": predicted_disorder,
                "model": model_name,
                "probabilities": probabilities,
                "confidence": float(max(probabilities.values())) if probabilities else None
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def predict_all_models(self, features):
        """Make predictions using all models"""
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.predict(features, model_name)
        return results
    
    def get_prediction_explanation(self, features):
        """Get detailed prediction explanation"""
        results = self.predict_all_models(features)
        
        # Get the best model's prediction
        best_model = self.metadata.get('best_model', 'xgboost')
        best_prediction = results.get(best_model, {})
        
        return {
            "best_model_prediction": best_prediction,
            "all_models": results,
            "model_accuracies": {
                "xgboost": self.metadata.get('xgboost_accuracy'),
                "random_forest": self.metadata.get('random_forest_accuracy'),
                "logistic_regression": self.metadata.get('logistic_regression_accuracy'),
                "svm": self.metadata.get('svm_accuracy')
            }
        }


# Global model loader instance
model_loader = None


def initialize_models(models_dir="models"):
    """Initialize global model loader"""
    global model_loader
    model_loader = ModelLoader(models_dir)
    model_loader.load_all_models()
    return model_loader


def get_models():
    """Get the global model loader instance"""
    return model_loader
