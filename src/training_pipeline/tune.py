"""
Hiperparameter tunin with Optuna + MLflow

- Optimizes XGB params on eval set recall;
- Logs trials to MLflow
- Retrains best model and saves to model_output
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

DEFAULT_PATH = Path ("/opt/airflow/dags/data/cleaned/LOL_limpo.csv")
DEFAULT_OUT = Path("/opt/airflow/dags/models/xgb_best_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def _load_data(train_path: Path | str, sample_frac: Optional[float], random_state: int, test_size: float = 0.2):
    df = pd.read_csv(train_path)
    df = _maybe_sample(df, sample_frac, random_state)

    target = "blueWins"
    x = df.drop(columns=[target])
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def tune_model(
        train_path: Path | str = DEFAULT_PATH,
        model_output: Path | str = DEFAULT_OUT,
        n_trials: int = 15,
        sample_frac: Optional[float] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "xgboost_optuna_lol",
        random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Run Optuna tuning; save best model; return (best_params, best_metrics).
    """
    os.environ["MLFLOW_TMP_DIR"] = "/tmp/mlflow_artifacts"
    # Se não foi passado tracking_uri, usa o do ambiente
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    x_train, x_test, y_train, y_test = _load_data(train_path, sample_frac, random_state, test_size=0.2)

    def objective(trial: optuna.Trial):
        params = {
            'n_estimators': trial.suggest_int("n_estimators",100, 500),
            'max_depth': trial.suggest_int("max_depth",5, 10),
            'learning_rate': trial.suggest_float("learning_rate",0.01, 0.2),
            'subsample': trial.suggest_float("subsample",0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree",0.5, 1.0),
            'gamma': trial.suggest_int("gamma",0, 7),
            'reg_alpha': trial.suggest_int("reg_alpha",0, 7),
            'reg_lambda': trial.suggest_int("reg_lambda",0, 7),
            'random_state': random_state,
            'n_jobs': -1,
        }

        with mlflow.start_run(nested=True):
            model = XGBClassifier(**params)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            recall = round(recall_score(y_test, y_pred, average='macro'), 2)

            mlflow.log_params(params)
            mlflow.log_metrics({"recall": recall})

        return recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("✅ Best params from Optuna:", best_params)

    # Retrain best model
    best_model = XGBClassifier(**{**best_params, "random_state": random_state, "n_jobs": -1})
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    best_metrics = {
        "recall": round(recall_score(y_test, y_pred, average='macro'), 2),
    }
    print("📊 Best tuned model metrics:", best_metrics)

    # Save to models/
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f"✅ Best model saved to {out}")

    # Log final best model to MLflow
    with mlflow.start_run(run_name="best_xgb_model") as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        tcols = pd.read_csv(DEFAULT_PATH, nrows=1)
        features = list(tcols.columns)
        with open ("features.json", "w") as f:
            json.dump(features, f)
        mlflow.log_artifact("features.json")
        # Registrar no Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = "xgb_lol_model"

        try:
            client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
            mv = mlflow.register_model(model_uri, model_name)
            print(f"✅ Modelo registrado no MLflow Registry: {model_name}, versão {mv.version}")


            client.set_model_version_tag(model_name, mv.version, "recall", str(best_metrics["recall"]))
            print(f"📊 Tag 'recall' adicionada na versão {mv.version} com valor {best_metrics['recall']}")

        except MlflowException as e:
            print(f"❌ Falha ao registrar modelo no MLflow: {e}")

    return best_params, best_metrics

if __name__ == "__main__":
    tune_model()