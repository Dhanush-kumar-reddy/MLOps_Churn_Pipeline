import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


def data_ingestion(input_path, output_path, test_size=0.2, random_state=42):
    try:
        logging.info("Data Ingestion Started")

        # Load data
        df = pd.read_csv(input_path)
        logging.info("Raw data loaded successfully")

        # Create output directory if not exists
        os.makedirs(output_path, exist_ok=True)

        # Train-test split (Stratified)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["Churn"]
        )

        # Save files
        train_path = os.path.join(output_path, "train.csv")
        test_path = os.path.join(output_path, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Train data saved at {train_path}")
        logging.info(f"Test data saved at {test_path}")
        logging.info(f"Train Shape: {train_df.shape}")
        logging.info(f"Test Shape: {test_df.shape}")
        logging.info("Data Ingestion Completed")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion(
        input_path="/Users/apple/6_Major_Projects/MLOpsChurnPipeline/data/raw/Telco-Customer-Churn.csv",
        output_path="data/interim"
    )