import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


def preprocess_data(train_path, test_path, output_path="data/processed"):
    try:
        logging.info("Data Preprocessing Started")

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Separate features and target
        X_train = train_df.drop("Churn", axis=1)
        y_train = train_df["Churn"]

        X_test = test_df.drop("Churn", axis=1)
        y_test = test_df["Churn"]

        # Column Alignment
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        logging.info("Column alignment completed")

        # Scaling (Fit on Train Only)
        scaler = StandardScaler()
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"]

        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        logging.info("Scaling completed")

        # Save scaler and columns
        os.makedirs("models", exist_ok=True)
        save_object("models/scaler.pkl", scaler)
        save_object("models/columns.pkl", X_train.columns)

        logging.info("Scaler and columns saved")

        # Save final processed data
        X_train["Churn"] = y_train
        X_test["Churn"] = y_test

        os.makedirs(output_path, exist_ok=True)

        train_final_path = f"{output_path}/train_final.csv"
        test_final_path = f"{output_path}/test_final.csv"

        X_train.to_csv(train_final_path, index=False)
        X_test.to_csv(test_final_path, index=False)

        logging.info(f"Train final data saved at {train_final_path}")
        logging.info(f"Test final data saved at {test_final_path}")
        logging.info(f"Train Shape: {X_train.shape}")
        logging.info(f"Test Shape: {X_test.shape}")
        logging.info("Data Preprocessing Completed")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    preprocess_data(
        train_path="data/processed/train_fe.csv",
        test_path="data/processed/test_fe.csv"
    )