from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime, timedelta
from src.feature_pipeline.load_preprocess import load_and_preprocess
from src.training_pipeline.tune import tune_model

def extract_csv_from_bigquery(**context):
    """
    Consulta a tabela Base_LOL no BigQuery e salva sempre como:
    /opt/airflow/dags/data/raw/base_lol.csv (sobrescreve o antigo).
    Empurra XCom key 'raw_path' com o caminho absoluto do CSV salvo.
    """
    import os
    from pathlib import Path
    import traceback

    try:
        # imports pesados dentro da task
        from google.cloud import bigquery


        # caminhos
        DATA_DIR = Path("/opt/airflow/dags/data/raw")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_file = DATA_DIR / "base_lol.csv"

        # opcional: log do ambiente de credenciais
        gcp_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        print(f"[extract] GOOGLE_APPLICATION_CREDENTIALS={gcp_cred}")

        # 1) consulta BigQuery
        client = bigquery.Client(project="projeto-lol-504819")
        query = """
            SELECT *
            FROM `projeto-lol-504819.DatasetLOL.Base_LOL`
        """
        df = client.query(query).to_dataframe()

        # 2) salva sempre como base_lol.csv (sobrescreve)
        df.to_csv(target_file, index=False)
        print(f"[extract] CSV salvo em {target_file} (sobrescrevendo se existia). rows={len(df)}")

        # 3) empurra caminho via XCom
        context['ti'].xcom_push(key='raw_path', value=str(target_file.resolve()))
        print(f"[extract] raw_path XCom set to: {str(target_file.resolve())}")

    except Exception as e:
        # log completo para facilitar debug no UI
        print("[extract] Erro durante extração do BigQuery:")
        traceback.print_exc()
        # re-levanta para que o Airflow registre a falha e aplique retry conforme configuração
        raise


def transform_dataframe(**context):
    raw_path = context['ti'].xcom_pull(key='raw_path')
    df = pd.read_csv(raw_path)
    df_clean = load_and_preprocess(df)
    clean_path = "/opt/airflow/dags/data/cleaned/LOL_limpo.csv"
    context['ti'].xcom_push(key='clean_path', value=clean_path)

def train_and_register_model(**context):
    best_params, best_metrics = tune_model(
        train_path="/opt/airflow/dags/data/cleaned/LOL_limpo.csv",
        model_output="/opt/airflow/dags/models/xgb_best_model.pkl",
        n_trials=15,
        sample_frac=0.2,
        experiment_name="xgboost_optuna_lol",
        random_state=42,
    )
    print("Melhores parâmetros:", best_params)
    print("Métricas finais:", best_metrics)

# DAG
default_args = {
    "owner": "lucas",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

with DAG(
    "projeto_lol_pipeline",
    default_args=default_args,
    description="Pipeline semanal: extração, transformação e treino",
    schedule_interval="@weekly",
    start_date=datetime(2026, 8, 11),
    catchup=False,
) as dag:

    extract_task = PythonOperator(
        task_id="extract_csv",
        python_callable=extract_csv_from_bigquery,
    )

    transform_task = PythonOperator(
        task_id="transform_dataframe",
        python_callable=transform_dataframe,
    )

    train_task = PythonOperator(
        task_id="train_and_register_model",
        python_callable=train_and_register_model,
    )

    extract_task >> transform_task >> train_task