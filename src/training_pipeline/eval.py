"""
Evaluate a saved XGBoost model
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report

DEFAULT_PATH = Path ("D:/PyCharm 2026.1.4/Projeto_LOL/src/data/cleaned/LOL_limpo.csv")
DEFAULT_MODEL = Path("D:/PyCharm 2026.1.4/Projeto_LOL/models/xgb_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def evaluate_model(
        model_path: Path | str = DEFAULT_MODEL,
        eval_path: Path | str = DEFAULT_PATH,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
) -> Dict[str, float]:
    df = pd.read_csv(eval_path)
    df = _maybe_sample(df, sample_frac, random_state)

    target = "blueWins"
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    model = load(model_path)
    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred)
    recall = round(recall_score(y_test, y_pred, average='macro'),2)
    metric = {"recall": recall}

    print(f"📊 Evaluation:\n {report}")
    print(f"Recall: {recall}")
    return metric

if __name__ == "__main__":
    evaluate_model()
