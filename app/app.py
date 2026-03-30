from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.pipeline.predict import PredictPipeline

app = FastAPI()

pipeline = PredictPipeline()


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    result = pipeline.predict(df)

    return {
        "Churn Prediction": int(result["Churn_Prediction"][0]),
        "Churn Probability": float(result["Churn_Probability"][0])
    }