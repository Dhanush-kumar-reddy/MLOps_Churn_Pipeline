# Customer Churn Prediction — End-to-End MLOps Project

## 📌 Business Problem

Customer churn is a major problem in telecom, banking, and subscription-based companies.  
Acquiring a new customer costs 5–10× more than retaining an existing customer.

Companies want to:
- Predict which customers will churn
- Take action (offers, discounts, support)
- Reduce revenue loss

This project builds a **Customer Churn Prediction System** and deploys it using a complete **MLOps pipeline**.

---

## 🎯 Problem Statement

The goal of this project is to build a machine learning system that:

1. Predicts whether a customer will churn or not
2. Tracks experiments using MLflow
3. Automates training using Airflow
4. Deploys the model using FastAPI
5. Containerizes the application using Docker
6. Automates deployment using GitHub Actions
7. Monitors the system using Prometheus & Grafana

This is a **production-level ML system**, not just a model.

---

## 🏗️ Project Architecture

![Architecture](images/architecture.png)

### Architecture Explanation

The system consists of the following components:

| Component | Description |
|----------|-------------|
| Data Ingestion | Loads raw data |
| Feature Engineering | Creates new features |
| Data Preprocessing | Scaling & encoding |
| Model Training | Train ML models |
| MLflow | Experiment tracking |
| Airflow | Pipeline automation |
| FastAPI | Model API |
| Docker | Containerization |
| GitHub Actions | CI/CD |

---

## 🔄 Project Flow Diagram

![Flow Diagram](images/flow.png)

### Pipeline Flow

Data → Feature Engineering → Preprocessing → Model Training → Evaluation
↓
MLflow
↓
Best Model
↓
FastAPI
↓
Docker
↓
Deployment



---

## 💼 Business Use Case

![Business Use Case](images/business.png)

The model predicts customers who are likely to churn so that the company can:
- Offer discounts
- Provide better support
- Improve customer retention
- Increase revenue

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (Best Model)

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Final Model Performance

| Metric | Score |
|-------|------|
| Accuracy | ~0.77 |
| Precision | ~0.55 |
| Recall | ~0.74 |
| F1 Score | ~0.63 |
| ROC-AUC | ~0.84 |

Threshold tuning improved **Recall to ~0.81**, which is important for churn prediction.

---

## 🛠️ Tech Stack

| Category | Tools Used |
|---------|------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Pipeline Automation | Apache Airflow |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus |
| Dashboard | Grafana |

---

## 📁 Project Structure


Data → Feature Engineering → Preprocessing → Model Training → Evaluation
↓
MLflow
↓
Best Model
↓
FastAPI
↓
Docker
↓
Deployment


---


---

## ▶️ How to Run the Project

### Step 1 — Train Model

```bash
python -m src.data_ingestion.data_ingestion
python -m src.feature_engineering.feature_engineering
python -m src.data_preprocessing.data_preprocessing
python -m src.model_training.train
python -m src.model_training.evaluate


Step 2 — Run MLflow
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlflow \
--host 127.0.0.1 \
--port 5001

Open:
http://127.0.0.1:5001

Step 3 — Run FastAPI
uvicorn app.app:app --reload

Open:
http://127.0.0.1:8000/docs

Step 4 — Run Docker
docker build -t churn-api .
docker run -p 8000:8000 churn-api
Step 5 — Run Airflow
airflow standalone

Open:
http://localhost:8080
🔁 CI/CD Pipeline (GitHub Actions)

Whenever code is pushed to GitHub:

GitHub Actions runs
Docker image builds
API deploys automatically

This automates deployment.


API requests
Errors
Latency
Model performance
🚀 Key MLOps Concepts Used
Data Pipeline
Feature Engineering Pipeline
Model Training Pipeline
Experiment Tracking
Model Versioning
CI/CD Pipeline
Containerization
API Deployment
Pipeline Scheduling

🏁 Conclusion

This project demonstrates how to build a production-ready end-to-end machine learning system using MLOps tools.
It covers the complete lifecycle from data ingestion to model deployment and monitoring.

This is not just a machine learning model — it is a complete ML production system.

👤 Author

Gajjala Dhanush kumar reddy
Data Science | Machine Learning | MLOps


---


