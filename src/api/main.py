from fastapi import FastAPI               # Web framework for APIs
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd
from joblib import load
from src.inference_pipeline.inference import (
    DEFAULT_MODEL,
    TRAIN_FEATURE_COLUMNS,
    predict,
)

MODEL_PATH = DEFAULT_MODEL
model = None
model_load_error = None

try:
    model = load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


# ---------------------------------
# App
# ---------------------------------
app = FastAPI(title="LOL prediction API")

@app.get("/")
def root():
    return {"message": "LOL prediction API is running 🎮" }

# /health -> checks if model exists, returns status info
@app.get("/health")
def health():
    status: Dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
        "n_features_expected": (
            len(TRAIN_FEATURE_COLUMNS)
            if TRAIN_FEATURE_COLUMNS is not None
            else None
        ),
    }
    if model is not None:
        status["status"] = "healthy"
    else:
        status["status"] = "unhealthy"
        status["error"] = model_load_error or "Model not loaded"
    return status

# Prediction Endpoint
@app.post("/predict")
def predict_endpoint(data: List[Dict]):
    df = pd.DataFrame(data)
    if df.empty:
        return { "error": "No data found" }

    preds_df = predict(df)

    results = []
    for i, row in preds_df.iterrows():
        result = {
            "features": row.drop(["predicted_winner", "actual_winner"], errors="ignore").to_dict(),
            "predicted_winner": int(row["predicted_winner"])
        }

        if "actual_winner" in row and pd.notnull(row["actual_winner"]):
            result["actual_winner"] = int(row["actual_winner"])
        results.append(result)

    return {"results": results}
