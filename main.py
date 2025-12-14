import os
import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

app = FastAPI(title="CI/CD Project (homework 3)", version=MODEL_VERSION)
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}

# @app.get("/predict")
# def predict():
#     return {"prediction": "ok", "version": MODEL_VERSION}

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    version: str

@app.get("/predict", response_model=PredictResponse)
def predict(pclass: int = Query(..., ge=1, le=3)):
    X = pd.DataFrame([{"Pclass": pclass}])
    pred_raw = model.predict(X)
    pred = int(pred_raw[0])
    probs = model.predict_proba(X)[0]
    confidence = float(probs.max())

    return PredictResponse(prediction=pred, confidence=confidence, version=MODEL_VERSION)

