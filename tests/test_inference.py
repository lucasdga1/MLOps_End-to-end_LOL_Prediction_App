# Test inference
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_pipeline.inference import predict

@pytest.fixture(scope="session")
def sample_df():
    # Load a small sample from cleaning folder
    sample_path = ROOT / "data/cleaned/LOL_limpo.csv"
    df = pd.read_csv(sample_path).sample(n=5, random_state=42).reset_index(drop=True)
    return df

def test_inference_runs_and_returns_predictions(sample_df):
    # Ensure inference pipeline runs and returns predicted winner
    if "gameId" in sample_df.columns:
        sample_df = sample_df.drop(columns=["gameId"])



    preds_df = predict(sample_df)

    # Check output is not empty
    assert not preds_df.empty

    # Must include prediction column
    assert "predicted_winner" in preds_df.columns
    assert "actual_winner" in preds_df.columns

    # Predictions should be numeric
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_winner"])

    print("✅ Inference pipeline test passed. Predictions:")
    print(preds_df[["predicted_winner"]].head())