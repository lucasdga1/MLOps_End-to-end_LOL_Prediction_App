"""
Load and preprocess the data
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/opt/airflow/dags/data/cleaned")

def load_and_preprocess(
        raw_data: pd.DataFrame,
        output_dir: Path | str = DATA_DIR,
):
    # Load dataset
    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError("Esperado um DataFrame vindo da DAG do Airflow")
    df = raw_data.copy()


    # Drop colunas não utilizáveis
    if "gameId" in df.columns:
        df = df.drop(columns=["gameId"])


    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "LOL_limpo.csv", index=False)

    print(f"✅ Data processing completed (saved to {outdir}).")
    print(f"Processed Data: {df.shape}")

    return df

if __name__ == "__main__":
    load_and_preprocess()