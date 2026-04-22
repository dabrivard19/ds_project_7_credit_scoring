
from fastapi import FastAPI, HTTPException

from app.config import SELECTED_FEATURES
from app.model_loader import load_model
from app.predictor import run_prediction
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(title="Credit Scoring API - 10 variables", version="1.0.0")


@app.get("/")
def root():
    return {"message": "API OK"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/form-config")
def form_config():
    return {"features": SELECTED_FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = load_model()
        prediction, probability, used_features = run_prediction(model, request.features)
        print("Prédiction effectuée avec succès.")
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            used_features=used_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
