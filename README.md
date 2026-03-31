# MLOps Customer Churn Prediction Pipeline

## 📌 Project Overview
This project is an **end-to-end MLOps pipeline** for predicting customer churn using Machine Learning.  
The system automates data ingestion, feature engineering, model training, evaluation, experiment tracking, and deployment using modern MLOps tools.

The trained model is deployed as an API using FastAPI and Docker, and the entire ML pipeline is orchestrated using Airflow and automated using GitHub Actions (CI/CD).

---

## 📊 Problem Statement
Customer churn is a major problem for telecom and subscription-based companies. Acquiring new customers is more expensive than retaining existing customers.  
This project predicts whether a customer will churn based on their service usage and account information.

**Business Goal:**  
Identify customers likely to churn so the company can take preventive actions such as offers, discounts, or customer support.

## Business Problem

![Business](assets/business.png)

---

## 🏗️ Project Architecture

![Architecture](assets/architecture.png)

---

## 🔄 ML Pipeline Flow

![Flow](assets/flow.png)

---

## 🧰 Tech Stack

| Category | Tools Used |
|---------|------------|
| Programming | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| Workflow Orchestration | Airflow |
| CI/CD | GitHub Actions |
| Configuration | YAML |
| Version Control | GitHub |

---

## 📁 Project Structure

MLOps-Churn-Pipeline/
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── data_ingestion/
│   ├── feature_engineering/
│   ├── data_preprocessing/
│   ├── model_training/
│   ├── pipeline/
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── app/
│   └── app.py
│
├── airflow/
│   └── dags/
│       └── churn_pipeline.py
│
├── models/
│
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
├── README.md
└── .github/workflows/mlops.yml

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Create Virtual Environment

python3.10 -m venv venv
source venv/bin/activate

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run Training Pipeline 

python -m src.data_ingestion.data_ingestion
python -m src.feature_engineering.feature_engineering
python -m src.data_preprocessing.data_preprocessing
python -m src.model_training.train
python -m src.model_training.evaluate

### 4️⃣ Run FastAPI

uvicorn app.app:app –reload

Open:

http://127.0.0.1:8000/docs

---

## 🐳 Run Using Docker

### Build Docker Image

docker build -t churn-api .

### Run Docker Container

docker run -p 8000:8000 churn-api

Open:

http://127.0.0.1:8000/docs

---

## ⏰ Run Airflow Pipeline

export AIRFLOW_HOME=~/MLOps-Churn-Pipeline/airflow
airflow standalone

Open Airflow UI:

http://localhost:8080

Trigger DAG:

customer_churn_pipeline

---

## 📈 MLflow Experiment Tracking

Run MLflow:

mlflow ui –port 5001

Open:

http://127.0.0.1:5001

---

## 🤖 Model Output

The model returns:

| Output | Meaning |
|-------|---------|
| Churn_Prediction = 1 | Customer will churn |
| Churn_Prediction = 0 | Customer will not churn |
| Churn_Probability | Probability of churn |

Threshold tuned to **0.40** to improve recall and catch more churn customers.

---

## 🔁 CI/CD Pipeline (GitHub Actions)

This project uses GitHub Actions to:
- Install dependencies
- Run training pipeline
- Train model
- Evaluate model
- Build Docker image

This automates the ML pipeline when new code is pushed.

---

## 🚀 Future Improvements
- Deploy on AWS EC2
- Use AWS S3 for data storage
- Use AWS ECR for Docker images
- Add model monitoring
- Add data drift detection

---

## 👨‍💻 Author
**Dhanush Kumar**  
AIML Engineer | Data Science | MLOps

---

## ⭐ Conclusion
This project demonstrates a **complete end-to-end MLOps pipeline** including:
- Data pipeline
- Model training
- Experiment tracking
- API deployment
- Docker containerization
- Workflow automation
- CI/CD automation

This is a production-style ML system.