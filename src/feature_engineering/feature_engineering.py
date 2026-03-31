import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
import sys


def feature_engineering(input_path, output_path):
    try:
        df = pd.read_csv(input_path)

        # Basic Cleaning
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        # Convert TotalCharges to numeric
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)

        # Replace categories
        replace_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        for col in replace_cols:
            df[col] = df[col].replace({"No internet service": "No"})

        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

        # Deep Feature Engineering

        # Avg Monthly Spend
        df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]
        df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(0)

        # Tenure Groups
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, 100],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
        )

        # High Monthly Charges
        df["HighMonthlyCharges"] = (df["MonthlyCharges"] > 70).astype(int)

        # Low Tenure
        df["LowTenure"] = (df["tenure"] < 12).astype(int)

        # High Total Charges
        df["HighTotalCharges"] = (df["TotalCharges"] > 2000).astype(int)

        # Fiber indicator
        df["IsFiber"] = (df["InternetService"] == "Fiber optic").astype(int)

        # Monthly contract indicator
        df["IsMonthlyContract"] = (df["Contract"] == "Month-to-month").astype(int)

        # Payment risk
        df["PaymentRisk"] = (df["PaymentMethod"] == "Electronic check").astype(int)

        # Service risk score
        df["ServiceRisk"] = (
            (df["OnlineSecurity"] == "No").astype(int) +
            (df["TechSupport"] == "No").astype(int) +
            (df["OnlineBackup"] == "No").astype(int)
        )

        # Number of services used
        services = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        df["NumServices"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

        # Target Encoding
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # One-Hot Encoding
        df = pd.get_dummies(df, drop_first=True)

        # -----------------------
        # Save Processed File
        # -----------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"Feature Engineering completed for: {input_path}")
        print(f"Saved to: {output_path}")
        print("Shape:", df.shape)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    feature_engineering(
        input_path="data/interim/train.csv",
        output_path="data/processed/train_fe.csv"
    )

    feature_engineering(
        input_path="data/interim/test.csv",
        output_path="data/processed/test_fe.csv"
    )