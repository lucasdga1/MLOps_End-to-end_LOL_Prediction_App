"""
Inference pipeline for LOL classification

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing.
- Aligns features with training.
- Returns predictions.
"""

import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing
from src.feature_pipeline.load_preprocess import load_and_preprocess

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_TRAIN = PROJECT_ROOT / "src" / "data" / "raw" / "Base_LOL.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)

if DEFAULT_TRAIN.exists():
    _train_cols = pd.read_csv(DEFAULT_TRAIN, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "blueWins"]  # excluding target column
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: Path | str | pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
) -> pd.DataFrame:
    # Preprocess and clean
    df = load_and_preprocess(input_df)

    # Separate actuals if present
    y_true = None
    if "blueWins" in df.columns:
        y_true = df["blueWins"].tolist()
        df = df.drop(columns=["blueWins"])

    # Remove gameId if exists
    if "gameId" in df.columns:
        df = df.drop(columns=["gameId"])


    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Load model and predict
    model = load(model_path)
    preds = model.predict(df)

    # Build output
    out = df.copy()
    out["predicted_winner"] = preds
    if y_true is not None:
        out["actual_winner"] = y_true

    return out

# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on LOL match data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN), help="Path to raw csv file")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")