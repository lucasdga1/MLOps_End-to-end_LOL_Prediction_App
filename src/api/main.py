from fastapi import FastAPI               # Web framework for APIs
from pathlib import Path
import time
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd
import os
import mlflow.pyfunc
import mlflow


# Load model from mlflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = os.getenv("MODEL_URI", "models:/xgb_lol_model/latest")

model = None

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
@app.on_event("startup")
def load_model_on_startup():
    global model, TRAIN_FEATURE_COLUMNS
    max_tries = 10
    for i in range(max_tries):
        try:
            model = mlflow.pyfunc.load_model(MODEL_URI)
            print("Model loaded from", MODEL_URI)
            break
        except Exception as e:
            print(f"Attempt {i+1}/{max_tries} - model not ready: {e}")
            time.sleep(2)
    if model is None:
        raise RuntimeError("Failed to load model at startup")

@app.get("/")
def root():
    return {"message": "LOL prediction API is running 🎮" }

# /health -> checks if model exists, returns status info
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": MODEL_URI}
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
def predict(data: List[Dict]):
    df = pd.DataFrame(data)
    if df.empty:
        return { "error": "No data found" }

    y_true = None
    if "blueWins" in df.columns:
        y_true = df["blueWins"].tolist()
        df = df.drop(columns=["blueWins"])

    EXPECTED_FEATURE_ORDER = [
        "redExperienceDiff", "blueGoldPerMin", "blueExperienceDiff",
        "redEliteMonsters", "blueFirstBlood", "blueCSPerMin",
        "redAvgLevel", "blueWardsPlaced", "blueAvgLevel", "blueDragons",
        "redTotalJungleMinionsKilled", "redDeaths", "redKills", "redAssists",
        "redTowersDestroyed", "blueDeaths", "redDragons", "blueTotalExperience",
        "blueTowersDestroyed", "redCSPerMin", "blueKills", "redGoldDiff",
        "redWardsPlaced", "redWardsDestroyed", "blueAssists",
        "redTotalMinionsKilled", "redGoldPerMin", "redHeralds",
        "blueTotalJungleMinionsKilled", "blueTotalMinionsKilled",
        "blueWardsDestroyed", "blueTotalGold", "blueEliteMonsters",
        "redFirstBlood", "redTotalExperience", "blueHeralds", "blueGoldDiff",
        "redTotalGold"
    ]
    df = df.reindex(columns=EXPECTED_FEATURE_ORDER, fill_value=0)

    df = df.dropna()


    preds = model.predict(df)

    results = []
    for i, row in df.iterrows():
        result = {
            "features": row.to_dict(),
            "predicted_winner": int(preds[i])
        }

        if y_true is not None:
            result["actual_winner"] = int(y_true[i])
        results.append(result)

    return {"results": results}
