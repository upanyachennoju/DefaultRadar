from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap

MODEL_PATH = Path("artifacts/models/best_model.pkl")
PREPROCESSOR_PATH = Path("artifacts/preprocessors/preprocessing.pkl")
THRESHOLD = 0.8

app = FastAPI(title="Credit Risk Scorer")


class CreditRiskRequest(BaseModel):
    age: int
    monthly_income: float
    debt_ratio: float
    credit_utilization: float
    transaction_count_30d: int
    avg_transaction_amount: float
    employment_type: str
    education_level: str
    region: str
    device_type: str
    last_payment_delay_days: int
    internal_score_v2: float



class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int
    risk_tier: Literal["low", "medium", "high"]
    explainability: dict


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open("rb") as f:
        artifact = pickle.load(f)

    return artifact


try:
    model = _load_pickle(MODEL_PATH)
    preprocessor = _load_pickle(PREPROCESSOR_PATH)
except Exception as exc:
    model = None
    preprocessor = None
    startup_error = exc
else:
    startup_error = None

X_background = np.load("artifacts/transformed_data/X_train.npy")[:100]
explainer = shap.LinearExplainer(model, X_background)

def _map_risk_tier(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.7:
        return "medium"
    return "high"


@app.get("/health")
def health_check():
    if startup_error:
        return {"status": "failed", "error": str(startup_error)}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CreditRiskRequest):
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {startup_error}")

    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor is not available")

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        features = preprocessor.transform(input_df)
        shap_values = explainer(features)

        feature_names = input_df.columns.tolist()
        explainability = {
            feature_names[i]: float(shap_values.values[0][i]) for i in range(len(feature_names))
        }

        if not hasattr(model, "predict_proba"):
            raise AttributeError("Loaded model does not implement predict_proba")

        probability = float(model.predict_proba(features)[:, 1][0])
        prediction = int(probability >= THRESHOLD)
        risk_tier = _map_risk_tier(probability)

        return PredictionResponse(
            default_probability=probability,
            prediction=prediction,
            risk_tier=risk_tier,
            explainability=explainability
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
