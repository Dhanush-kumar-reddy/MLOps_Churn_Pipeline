import joblib
import os
from src.logger import logging


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    logging.info(f"Object saved at {file_path}")


def load_object(file_path):
    obj = joblib.load(file_path)
    logging.info(f"Object loaded from {file_path}")
    return obj

import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config