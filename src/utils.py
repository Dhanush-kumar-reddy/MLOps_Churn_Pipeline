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