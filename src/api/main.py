from fastapi import FastAPI               # Web framework for APIs
from pathlib import Path
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd
import joblib
from src.inference_pipeline.inference import predict

# Load model
MODEL_PATH = Path("./models/xgb_best_model.pkl")

model = joblib.load(MODEL_PATH)

TRAIN_FE_PATH = Path("./data/cleaned/LOL_limpo.csv")

# Expected columns for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "blueWins"]
else:
    TRAIN_FEATURE_COLUMNS = None


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
    status: Dict[str, Any] = {"model_path": MODEL_PATH}
    if model is not None:
        status["status"] = "healthy"
        # opcional: mostrar quantas features são esperadas
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    else:
        status["status"] = "unhealthy"
        status["error"] = "Model not loaded"
    return status

# Prediction Endpoint
@app.post("/predict")
def predict_endpoint(data: List[Dict]):
    df = pd.DataFrame(data)
    if df.empty:
        return { "error": "No data found" }

    preds_df = predict(df, model_path=MODEL_PATH)

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
