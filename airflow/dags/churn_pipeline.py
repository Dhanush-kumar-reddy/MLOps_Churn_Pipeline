from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

from src.utils import load_config

config = load_config()
base_path = config["project"]["base_path"]

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1
}

with DAG(
    dag_id="customer_churn_pipeline",
    default_args=default_args,
    schedule=config["airflow"]["schedule_interval"],
    catchup=False
) as dag:

    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command=f"python {base_path}/src/data_ingestion/data_ingestion.py"
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"python {base_path}/src/feature_engineering/feature_engineering.py"
    )

    data_preprocessing = BashOperator(
        task_id="data_preprocessing",
        bash_command=f"python {base_path}/src/data_preprocessing/data_preprocessing.py"
    )

    model_training = BashOperator(
        task_id="model_training",
        bash_command=f"python {base_path}/src/model_training/train.py"
    )

    model_evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command=f"python {base_path}/src/model_training/evaluate.py"
    )

    data_ingestion >> feature_engineering >> data_preprocessing >> model_training >> model_evaluation