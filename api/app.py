"""
FastAPI Application for Churn Prediction
Simulates production deployment with REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn probability and provide retention recommendations",
    version="1.0.0"
)

# Global variables for model and config
model = None
feature_engineer = None
config = None


# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Input features for a single customer"""
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., gt=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=840.20)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.85,
                "TotalCharges": 2156.40
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    churn_probability: float
    churn_prediction: str
    risk_level: str
    confidence: float
    recommendation: str
    key_factors: List[str]


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[CustomerFeatures]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str]


def load_model_and_config():
    """Load trained model and configuration"""
    global model, feature_engineer, config
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_name = config['deployment']['champion_model']
    model_path = Path(__file__).parent.parent / 'models' / f'{model_name}_model.joblib'
    
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✓ Loaded model: {model_name}")
    else:
        print(f"⚠️  Model not found: {model_path}")
        model = None
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_and_config()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": config['deployment']['champion_model'] if config else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer
    
    Args:
        customer: Customer features
        
    Returns:
        Prediction with probability and recommendations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Engineer features
        df_eng = feature_engineer.engineer_features(df, fit=False)
        
        # Make prediction
        probability = model.predict_proba(df_eng)[0, 1]
        prediction = "Churn" if probability >= 0.5 else "No Churn"
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate confidence
        confidence = max(probability, 1 - probability)
        
        # Generate recommendation
        recommendation = generate_recommendation(customer_dict, probability)
        
        # Get key factors
        key_factors = identify_key_factors(customer_dict, probability)
        
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "recommendation": recommendation,
            "key_factors": key_factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers
    
    Args:
        request: Batch of customers
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for customer in request.customers:
        pred = await predict_churn(customer)
        predictions.append(pred)
    
    return {
        "predictions": predictions,
        "total_customers": len(predictions),
        "high_risk_count": sum(1 for p in predictions if p.risk_level == "High")
    }


def generate_recommendation(customer_data: Dict, probability: float) -> str:
    """
    Generate retention recommendation based on customer profile
    
    Args:
        customer_data: Customer features
        probability: Churn probability
        
    Returns:
        Recommendation string
    """
    if probability < 0.3:
        return "Customer is low risk. Continue standard engagement."
    
    recommendations = []
    
    # Contract-based
    if customer_data['Contract'] == 'Month-to-month':
        recommendations.append("Offer long-term contract discount")
    
    # Payment method
    if customer_data['PaymentMethod'] == 'Electronic check':
        recommendations.append("Encourage automatic payment setup")
    
    # Service-based
    if customer_data['OnlineSecurity'] == 'No':
        recommendations.append("Promote security services bundle")
    
    if customer_data['TechSupport'] == 'No':
        recommendations.append("Offer premium support trial")
    
    # Tenure-based
    if customer_data['tenure'] < 12:
        recommendations.append("Early customer - focus on onboarding support")
    
    # Price sensitivity
    if customer_data['MonthlyCharges'] > 70:
        recommendations.append("Review pricing tier and offer loyalty discount")
    
    if recommendations:
        return " | ".join(recommendations[:3])  # Top 3
    else:
        return "Proactive outreach recommended"


def identify_key_factors(customer_data: Dict, probability: float) -> List[str]:
    """
    Identify key churn factors for this customer
    
    Args:
        customer_data: Customer features
        probability: Churn probability
        
    Returns:
        List of key factors
    """
    factors = []
    
    if customer_data['Contract'] == 'Month-to-month':
        factors.append("Month-to-month contract")
    
    if customer_data['tenure'] < 12:
        factors.append("New customer (<12 months)")
    
    if customer_data['PaymentMethod'] == 'Electronic check':
        factors.append("Electronic check payment")
    
    if customer_data['InternetService'] == 'Fiber optic':
        factors.append("Fiber optic service")
    
    if customer_data['MonthlyCharges'] > 70:
        factors.append("High monthly charges")
    
    # Service adoption
    services = sum([
        customer_data.get('OnlineSecurity', 'No') == 'Yes',
        customer_data.get('OnlineBackup', 'No') == 'Yes',
        customer_data.get('DeviceProtection', 'No') == 'Yes',
        customer_data.get('TechSupport', 'No') == 'Yes'
    ])
    
    if services == 0:
        factors.append("No additional services")
    
    return factors[:5]  # Top 5 factors


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_name": config['deployment']['champion_model'],
        "features_count": len(feature_engineer.feature_names) if feature_engineer.feature_names else "Unknown",
        "version": "1.0.0"
    }


# Example request for testing
"""
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 2156.40
  }'
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
