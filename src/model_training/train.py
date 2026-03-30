import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from xgboost import XGBClassifier


def train_model(train_path):
    try:
        
        logging.info("Model Training Started")

        # Load training data
        df = pd.read_csv(train_path)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        
        #tracking with mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_experiment("Customer_Churn_Prediction")  
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Models and hyperparameters
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=2000),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "class_weight": [None, "balanced"]
                }
            },

            "Decision Tree": {
                "model": DecisionTreeClassifier(),
                "params": {
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [5, 10, 20],
                    "min_samples_leaf": [2, 5, 10],
                    "class_weight": [None, "balanced"]
                }
            },

            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {
                    "n_estimators": [200, 300],
                    "max_depth": [8, 10, 12],
                    "min_samples_split": [5, 10],
                    "min_samples_leaf": [2, 4],
                    "max_features": ["sqrt"],
                    "class_weight": [
                        {0: 1, 1: 2},
                        {0: 1, 1: 3},
                        {0: 1, 1: 4}
                    ]
                }
            },

            "XGBoost": {
                "model": XGBClassifier(eval_metric="logloss"),
                "params": {
                    "n_estimators": [200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "scale_pos_weight": [2, 3, 4]
                }
            }
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        # Training loop
        for name, mp in models.items():
            logging.info(f"Running GridSearch for {name}")

            with mlflow.start_run(run_name=name):

                grid = GridSearchCV(
                    estimator=mp["model"],
                    param_grid=mp["params"],
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1
                )

                grid.fit(X, y)

                best = grid.best_estimator_
                best_cv_score = grid.best_score_

                # Log parameters
                mlflow.log_params(grid.best_params_)

                # Log metric
                mlflow.log_metric("ROC_AUC", best_cv_score)

                # Log model
                mlflow.sklearn.log_model(best, name)

                logging.info(f"{name} Best Parameters: {grid.best_params_}")
                logging.info(f"{name} Best CV ROC-AUC: {best_cv_score}")

                print(f"{name} Best ROC-AUC:", best_cv_score)

                if best_cv_score > best_score:
                    best_score = best_cv_score
                    best_model = best
                    best_model_name = name
                    
        # Save best model
        save_object("models/model.pkl", best_model)

        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best CV ROC-AUC: {best_score}")
        logging.info("Model Training Completed")

        print("\nBest Model:", best_model_name)
        print("Best ROC-AUC:", best_score)

        # Feature Importance (for tree models)
        if hasattr(best_model, "feature_importances_"):
            feature_importance = pd.DataFrame({
                "feature": X.columns,
                "importance": best_model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            feature_importance.to_csv("models/feature_importance.csv", index=False)
            logging.info("Feature importance saved at models/feature_importance.csv")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    train_model("data/processed/train_final.csv")