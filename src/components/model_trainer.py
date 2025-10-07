import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.components.model_evaluator import ModelEvaluator
from src.components.model_explainer import ModelExplainer


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    metadata_file_path = os.path.join("artifacts", "model_metadata.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define candidate models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(eval_metric="rmse", early_stopping_rounds=10),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, early_stopping_rounds=10),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.8, 0.9],
                    "n_estimators": [16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [16, 32, 64, 128, 256],
                },
            }

            # Evaluate models with hyperparameter tuning
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Select best model based on R2 score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with score > 0.6")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Initialize comprehensive model evaluator and explainer
            evaluator = ModelEvaluator()
            explainer = ModelExplainer()
            
            # Define feature names for feature importance plots
            feature_names = ["writing_score", "reading_score", "gender", "race_ethnicity", 
                           "parental_level_of_education", "lunch", "test_preparation_course"]
            
            # Perform comprehensive evaluation with visualizations
            logging.info("Starting comprehensive model evaluation with visualizations")
            all_metrics, eval_best_model_name = evaluator.evaluate_models_comprehensive(
                models=models, 
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                feature_names=feature_names
            )
            
            # Model explainability for best model
            logging.info("Generating model explainability analysis")
            explainer.explain_model(
                model=best_model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                model_name=best_model_name
            )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Get best model metrics for return
            best_model_metrics = next(metric for metric in all_metrics if metric['model_name'] == best_model_name)
            r2_square = best_model_metrics['r2_score']

            # Cross-validation for robustness
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
            logging.info(f"Cross-validation R2 scores: {cv_scores}")
            logging.info(f"Mean CV R2: {cv_scores.mean()}")

            # Save comprehensive metadata
            metadata = {
                "model_name": best_model_name,
                "train_score": best_model.score(X_train, y_train),
                "test_score": r2_square,
                "cv_mean_score": cv_scores.mean(),
                "mae": best_model_metrics['mae'],
                "mse": best_model_metrics['mse'],
                "rmse": best_model_metrics['rmse'],
                "mape": best_model_metrics['mape'],
                "adjusted_r2": best_model_metrics['adjusted_r2'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "all_model_scores": model_report
            }

            os.makedirs(os.path.dirname(self.model_trainer_config.metadata_file_path), exist_ok=True)
            with open(self.model_trainer_config.metadata_file_path, "w") as f:
                json.dump(metadata, f, indent=4)

            logging.info("Comprehensive model evaluation completed with visualizations and detailed metrics")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
