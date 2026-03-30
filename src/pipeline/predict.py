import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, load_config


class PredictPipeline:
    def __init__(self):
        try:
            config = load_config()

            self.model = load_object(config["data"]["model_path"])
            self.scaler = load_object(config["data"]["scaler_path"])
            self.columns = load_object(config["data"]["columns_path"])
            self.threshold = config["threshold"]["tuned"]

            logging.info("Prediction Pipeline Initialized")

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess(self, df):
        try:
            if "customerID" in df.columns:
                df.drop("customerID", axis=1, inplace=True)

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(inplace=True)

            replace_cols = [
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]

            for col in replace_cols:
                df[col] = df[col].replace({"No internet service": "No"})

            df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

            df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]
            df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(0)

            df["HasSecurity"] = df["OnlineSecurity"].apply(lambda x: 1 if x == "Yes" else 0)
            df["HasBackup"] = df["OnlineBackup"].apply(lambda x: 1 if x == "Yes" else 0)
            df["HasTechSupport"] = df["TechSupport"].apply(lambda x: 1 if x == "Yes" else 0)

            df["IsFiber"] = df["InternetService"].apply(lambda x: 1 if x == "Fiber optic" else 0)
            df["IsMonthlyContract"] = df["Contract"].apply(lambda x: 1 if x == "Month-to-month" else 0)

            services = [
                "PhoneService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
            ]

            df["NumServices"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

            df = pd.get_dummies(df, drop_first=True)
            df.columns = df.columns.str.replace(" ", "_")
            df = df.reindex(columns=self.columns, fill_value=0)

            num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"]
            df[num_cols] = self.scaler.transform(df[num_cols])

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df):
        try:
            data = self.preprocess(df)

            prob = self.model.predict_proba(data)[:, 1]
            pred = (prob > self.threshold).astype(int)

            result = pd.DataFrame({
                "Churn_Prediction": pred,
                "Churn_Probability": prob
            })

            return result

        except Exception as e:
            raise CustomException(e, sys)