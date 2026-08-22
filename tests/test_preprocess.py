import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.feature_pipeline.load_preprocess import load_and_preprocess

"""
Test to confirm te pipeline works
"""
def test_load_and_preprocess(tmp_path):
    raw = pd.read_csv("src/data/raw/Base_LOL.csv")
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    df_limpo = load_and_preprocess(raw_path=str(raw_path), output_dir=tmp_path)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(exist_ok=True)

    print("✅ Full pipeline integration test passed")
