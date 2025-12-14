import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

app = FastAPI(title="CI/CD Project (homework 3)", version=MODEL_VERSION)
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}

@app.get("/predict")
def predict():
    return {"prediction": "ok", "version": MODEL_VERSION}


# class PredictRequest(BaseModel):
#     pclass: int


# class PredictResponse(BaseModel):
#     prediction: int
#     confidence: float
#     version: str

# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "version": MODEL_VERSION,
#     }


# @app.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):
#     df = pd.DataFrame([{"Pclass": request.pclass}])

#     prediction = int(model.predict(df)[0])
#     confidence = float(model.predict_proba(df)[0].max())

#     return PredictResponse(
#         prediction=prediction,
#         confidence=confidence,
#         version=MODEL_VERSION,
    # )