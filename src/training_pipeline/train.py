"""
Train a baseline XGBoost model.
Returns metrics and saves model to 'model_output'
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

DEFAULT_PATH = Path("D:/PyCharm 2026.1.4/Projeto_LOL/src/data/cleaned/LOL_limpo.csv")
DEFAULT_OUT = Path("D:/PyCharm 2026.1.4/Projeto_LOL/models/xgb_model.pkl")



def train_model(
        train_path: Path | str = DEFAULT_PATH,
        model_output: Path | str = DEFAULT_OUT,
        model_params: Optional[dict] = None,
        random_state: int = 42,
):
    """
    Train a baseline XGBoost model.
    Returns:
    model: XGBClassifier
    metrics: dict[str, float
    """
    df = pd.read_csv(train_path)

    target = "blueWins"
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 5,
    'reg_alpha': 5,
    'reg_lambda': 5
    }

    if model_params:
        params.update(model_params)

    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    recall = round(recall_score(y_test, y_pred, average='macro'), 2)
    metric ={"recall": recall}

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f"✅ Model trained. Saved to {out}")
    print(f"Classification report for model:\n {report}")

    return model, metric

if __name__ == "__main__":
    train_model()