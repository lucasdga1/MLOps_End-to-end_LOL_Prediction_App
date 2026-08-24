import math
import numpy as np
import pandas as pd
import pytest
from joblib import load
from pathlib import Path


from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import tune_model

TRAIN_PATH = Path("D:/PyCharm 2026.1.4/Projeto_LOL/data/cleaned/LOL_limpo.csv")

# Ensure we have the same keys in metrics dict
def _assert_metrics(m):
    assert set(m.keys()) == {"recall"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())

# TRAIN: trains a quick and simple model
def test_train_creates_model_and_metrics(tmp_path):
    out_path = tmp_path / "xgb_model.pkl"

    # small params + sample
    _, metrics = train_model(
        train_path = TRAIN_PATH,
        model_output= out_path,
        model_params={'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 5,
    'reg_alpha': 5,
    'reg_lambda': 5},
        sample_frac=0.02
    )
    assert out_path.exists()
    _assert_metrics(metrics)
    model = load(out_path)
    assert model is not None
    print("✅ train_model test passed")

def test_eval_works_with_saved_model(tmp_path):
    # train quick model
    model_path = tmp_path / "xgb_model.pkl"
    train_model(
        train_path=TRAIN_PATH,
        model_output=model_path,
        model_params={"n_estimators": 20},
        sample_frac=0.02,
    )
    metrics = evaluate_model(model_path=model_path, sample_frac=0.02 )
    _assert_metrics(metrics)
    print("✅ evaluate_model test passed")

def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / "xgb_best.pkl"
    tracking_dir = tmp_path / "mlruns"
    best_params, best_metrics = tune_model(
        train_path=TRAIN_PATH,
        model_output=model_out,
        n_trials=2,
        sample_frac=0.02,
        experiment_name="test_xgb_optuna",
    )
    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)
    print("✅ tune_model test passed")