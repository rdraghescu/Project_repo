"""
API REST pentru predicții Web Traffic Time on Site
Model: Linear Regression ENHANCED cu 20 features
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Inițializare FastAPI
app = FastAPI(
    title="Web Traffic Prediction API",
    description="API pentru predicții Time on Site folosind Linear Regression ENHANCED (20 features)",
    version="2.0.0"
)

# Încărcare model, scaler și metadata
with open("models/linear_regression_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/web_traffic_scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/model_metadata_v2.json", "r") as f:
    metadata = json.load(f)


# Modele Pydantic pentru request/response
class PredictionInput(BaseModel):
    """Input pentru predicție - features base"""
    pageviews: int = Field(..., ge=0, description="Număr pagini vizualizate")
    visitNumber: int = Field(..., ge=1, description="Număr vizită")
    hits: int = Field(..., ge=0, description="Număr total hit-uri")
    device_category: str = Field(..., description="Categorie dispozitiv: desktop/mobile/tablet")
    country: str = Field(..., description="Țara vizitatorului (ex: United States, India)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pageviews": 5,
                "visitNumber": 3,
                "hits": 12,
                "device_category": "desktop",
                "country": "United States"
            }
        }


class PredictionOutput(BaseModel):
    """Output predicție"""
    predicted_time_on_site_seconds: float = Field(..., description="Timp prezis pe site (secunde)")
    predicted_time_on_site_minutes: float = Field(..., description="Timp prezis pe site (minute)")
    prediction_timestamp: str = Field(..., description="Timestamp predicție")
    model_version: str = Field(..., description="Versiune model")
    features_used: List[str] = Field(..., description="Features folosite în predicție")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_time_on_site_seconds": 245.67,
                "predicted_time_on_site_minutes": 4.09,
                "prediction_timestamp": "2026-05-30T12:30:45",
                "model_version": "v2.0_ENHANCED",
                "features_used": ["pageviews", "visitNumber", "hits", "..."]
            }
        }


class BatchPredictionInput(BaseModel):
    """Input pentru predicții batch"""
    samples: List[PredictionInput] = Field(..., description="Lista de sample-uri pentru predicție")


class BatchPredictionOutput(BaseModel):
    """Output predicții batch"""
    predictions: List[float] = Field(..., description="Lista predicții (secunde)")
    count: int = Field(..., description="Număr predicții")
    average_time: float = Field(..., description="Media timpului prezis")


# Funcție pentru feature engineering
def engineer_features(data: dict) -> pd.DataFrame:
    """
    Recreează toate 20 features din input base
    Formule conform mlflow_ml_model_web_traffic.py
    """
    # Features base (3)
    pageviews = data["pageviews"]
    visitNumber = data["visitNumber"]
    hits = data["hits"]
    
    # Categorical encoding (12)
    device_category = data["device_category"].lower()
    country = data["country"]
    
    # Device encoding (drop_first=True => 2 coloane: mobile, tablet)
    device_mobile = 1 if device_category == "mobile" else 0
    device_tablet = 1 if device_category == "tablet" else 0
    
    # Country encoding (top 10, drop_first=True => 9 coloane active)
    # Ordinea EXACTĂ din model_metadata_v2.json:
    top_countries_encoded = [
        "Brazil", "Canada", "France", "Germany", "India", 
        "Japan", "Other", "United Kingdom", "United States", "Vietnam"
    ]
    country_encoded = {f"country_grouped_{c.replace(' ', '_')}": (1 if country == c else 0) 
                      for c in top_countries_encoded}
    
    # Engineered features (5) - FORMULE CORECTE
    pageviews_per_visit = pageviews / (visitNumber + 1)  # +1 pentru evitare împărțire la 0
    engagement_score = pageviews * hits  # Formula CORECTĂ: pageviews × hits
    hits_per_pageview = hits / (pageviews + 1)  # +1 pentru evitare împărțire la 0
    high_pageviews = 1 if pageviews > 4.0 else 0  # Median pageviews = 4.0
    is_returning = 1 if visitNumber > 1 else 0
    
    # Construire DataFrame cu toate 20 features în ordinea corectă
    features = {
        "pageviews": pageviews,
        "visitNumber": visitNumber,
        "hits": hits,
        "device_category_mobile": device_mobile,
        "device_category_tablet": device_tablet,
        **country_encoded,
        "pageviews_per_visit": pageviews_per_visit,
        "engagement_score": engagement_score,
        "hits_per_pageview": hits_per_pageview,
        "high_pageviews": high_pageviews,
        "is_returning": is_returning
    }
    
    return pd.DataFrame([features])


# Endpoints
@app.get("/")
def root():
    """Health check și info API"""
    return {
        "status": "online",
        "api": "Web Traffic Prediction API",
        "version": metadata.get("version", "2.0"),
        "model": metadata.get("model_type", "LinearRegression"),
        "features_count": len(metadata.get("features_all", [])),
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info"
        }
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predicție pentru un singur sample
    """
    try:
        # Feature engineering
        features_df = engineer_features(input_data.dict())
        
        # Scaling
        features_scaled = scaler.transform(features_df)
        
        # Predicție
        prediction = model.predict(features_scaled)[0]
        
        # Asigură-te că predicția e pozitivă
        prediction = max(0, prediction)
        
        return PredictionOutput(
            predicted_time_on_site_seconds=round(prediction, 2),
            predicted_time_on_site_minutes=round(prediction / 60, 2),
            prediction_timestamp=datetime.now().isoformat(),
            model_version=metadata.get("version", "v2.0_ENHANCED"),
            features_used=metadata.get("features_all", [])[:5] + ["..."]  # Primele 5 + ...
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare predicție: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionOutput)
def batch_predict(input_data: BatchPredictionInput):
    """
    Predicții pentru multiple samples
    """
    try:
        predictions = []
        
        for sample in input_data.samples:
            # Feature engineering
            features_df = engineer_features(sample.dict())
            
            # Scaling
            features_scaled = scaler.transform(features_df)
            
            # Predicție
            prediction = model.predict(features_scaled)[0]
            predictions.append(max(0, prediction))
        
        return BatchPredictionOutput(
            predictions=[round(p, 2) for p in predictions],
            count=len(predictions),
            average_time=round(np.mean(predictions), 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare batch predicție: {str(e)}")


@app.get("/model-info")
def model_info():
    """
    Informații despre model și performanță
    """
    return {
        "model_type": metadata.get("model_type", "LinearRegression"),
        "version": metadata.get("version", "v2.0_ENHANCED"),
        "features_count": len(metadata.get("features_all", [])),
        "features": metadata.get("features_all", []),
        "training_date": metadata.get("training_date", "N/A"),
        "training_samples": metadata.get("training_samples", "N/A"),
        "test_samples": metadata.get("test_samples", "N/A"),
        "performance": {
            "r2_score": metadata.get("r2_score", "N/A"),
            "mae": metadata.get("mae", "N/A"),
            "rmse": metadata.get("rmse", "N/A"),
            "interpretation": metadata.get("interpretation", {})
        },
        "top_features": metadata.get("coefficients_top10", [])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
