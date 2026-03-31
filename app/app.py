from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.pipeline.predict import PredictPipeline

app = FastAPI()

pipeline = PredictPipeline()

from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    gender: str = Field(example="Female")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="No")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=1)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="No")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="Yes")
    StreamingMovies: str = Field(example="Yes")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=95)
    TotalCharges: float = Field(example=95)

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