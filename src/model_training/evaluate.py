import pandas as pd
import sys
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


def evaluate_model(test_path):
    try:
        logging.info("Model Evaluation Started")

        # Load test data
        df = pd.read_csv(test_path)

        X_test = df.drop("Churn", axis=1)
        y_test = df["Churn"]

        # Load model
        model = load_object("models/model.pkl")

        # Predictions (default threshold = 0.5)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Default Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== Default Threshold (0.5) ===")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC-AUC:", roc)
        print("Confusion Matrix:\n", cm)

        logging.info(f"Default Accuracy: {accuracy}")
        logging.info(f"Default Precision: {precision}")
        logging.info(f"Default Recall: {recall}")
        logging.info(f"Default F1 Score: {f1}")
        logging.info(f"Default ROC-AUC: {roc}")

        # -----------------------------
        # Threshold Tuning
        # -----------------------------
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        J = tpr - fpr
        ix = np.argmax(J)
        best_threshold = thresholds[ix]

        logging.info(f"Best Threshold: {best_threshold}")
        print("\nBest Threshold:", best_threshold)

        # Apply new threshold
        y_pred_new = (y_prob > best_threshold).astype(int)

        # Metrics after threshold tuning
        accuracy_new = accuracy_score(y_test, y_pred_new)
        precision_new = precision_score(y_test, y_pred_new)
        recall_new = recall_score(y_test, y_pred_new)
        f1_new = f1_score(y_test, y_pred_new)
        cm_new = confusion_matrix(y_test, y_pred_new)

        print("\n=== After Threshold Tuning ===")
        print("Accuracy:", accuracy_new)
        print("Precision:", precision_new)
        print("Recall:", recall_new)
        print("F1 Score:", f1_new)
        print("Confusion Matrix:\n", cm_new)

        logging.info(f"Tuned Accuracy: {accuracy_new}")
        logging.info(f"Tuned Precision: {precision_new}")
        logging.info(f"Tuned Recall: {recall_new}")
        logging.info(f"Tuned F1 Score: {f1_new}")
        logging.info(f"Tuned Confusion Matrix:\n{cm_new}")

        logging.info("Model Evaluation Completed")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    evaluate_model("data/processed/test_final.csv")