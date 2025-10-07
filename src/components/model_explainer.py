import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelExplainabilityConfig:
    explainability_plots_dir: str = os.path.join("artifacts", "explainability")
    shap_plots_dir: str = os.path.join("artifacts", "explainability", "shap")
    feature_importance_dir: str = os.path.join("artifacts", "explainability", "feature_importance")

class ModelExplainer:
    def __init__(self):
        self.config = ModelExplainabilityConfig()
        os.makedirs(self.config.explainability_plots_dir, exist_ok=True)
        os.makedirs(self.config.shap_plots_dir, exist_ok=True)
        os.makedirs(self.config.feature_importance_dir, exist_ok=True)
    
    def create_feature_importance_plot(self, model, feature_names, model_name):
        """
        Create feature importance plots using different methods
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Method 1: Built-in feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.subplot(2, 2, 1)
                plt.title(f'{model_name} - Built-in Feature Importance')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                
            # Method 2: Coefficients (for linear models)
            elif hasattr(model, 'coef_'):
                coef = np.abs(model.coef_)
                indices = np.argsort(coef)[::-1]
                
                plt.subplot(2, 2, 1)
                plt.title(f'{model_name} - Coefficient Importance')
                plt.bar(range(len(coef)), coef[indices])
                plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.feature_importance_dir, f'{model_name}_feature_importance.png'))
            plt.close()
            
            logging.info(f"Feature importance plot saved for {model_name}")
            
        except Exception as e:
            logging.warning(f"Could not create feature importance plot for {model_name}: {str(e)}")
    
    def create_permutation_importance(self, model, X_test, y_test, feature_names, model_name):
        """
        Create permutation importance plot
        """
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            
            # Sort features by importance
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.boxplot([perm_importance.importances[i] for i in sorted_idx], 
                       labels=[feature_names[i] for i in sorted_idx])
            plt.title(f'{model_name} - Permutation Feature Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.feature_importance_dir, f'{model_name}_permutation_importance.png'))
            plt.close()
            
            logging.info(f"Permutation importance plot saved for {model_name}")
            
        except Exception as e:
            logging.warning(f"Could not create permutation importance for {model_name}: {str(e)}")
    
    def create_shap_plots(self, model, X_train, X_test, feature_names, model_name):
        """
        Create SHAP explanation plots
        """
        try:
            # Initialize SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For classification models
                explainer = shap.Explainer(model, X_train)
            else:
                # For regression models
                explainer = shap.Explainer(model, X_train)
            
            # Calculate SHAP values for test set (sample for performance)
            sample_size = min(100, len(X_test))
            X_test_sample = X_test[:sample_size]
            shap_values = explainer(X_test_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.shap_plots_dir, f'{model_name}_shap_summary.png'))
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f'{model_name} - SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.shap_plots_dir, f'{model_name}_shap_bar.png'))
            plt.close()
            
            # Waterfall plot for first prediction
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap_values[0], show=False)
            plt.title(f'{model_name} - SHAP Waterfall Plot (First Prediction)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.shap_plots_dir, f'{model_name}_shap_waterfall.png'))
            plt.close()
            
            logging.info(f"SHAP plots saved for {model_name}")
            
        except Exception as e:
            logging.warning(f"Could not create SHAP plots for {model_name}: {str(e)}")
    
    def create_partial_dependence_plots(self, model, X_train, feature_names, model_name):
        """
        Create partial dependence plots for top features
        """
        try:
            # Select top 4 features for PDP (assuming first 4 are most important)
            top_features = list(range(min(4, len(feature_names))))
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for i, feature_idx in enumerate(top_features):
                if i < len(axes):
                    # Calculate partial dependence
                    pdp_results = partial_dependence(model, X_train, [feature_idx], kind="average")
                    
                    axes[i].plot(pdp_results[1][0], pdp_results[0][0])
                    axes[i].set_title(f'Partial Dependence: {feature_names[feature_idx]}')
                    axes[i].set_xlabel(feature_names[feature_idx])
                    axes[i].set_ylabel('Partial Dependence')
            
            plt.suptitle(f'{model_name} - Partial Dependence Plots')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.explainability_plots_dir, f'{model_name}_partial_dependence.png'))
            plt.close()
            
            logging.info(f"Partial dependence plots saved for {model_name}")
            
        except Exception as e:
            logging.warning(f"Could not create partial dependence plots for {model_name}: {str(e)}")
    
    def generate_explainability_report(self, model_name, feature_names):
        """
        Generate HTML explainability report
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Explainability Report - {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #333; }}
                    .plot-section {{ margin: 20px 0; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Model Explainability Report: {model_name}</h1>
                
                <h2>Feature Importance</h2>
                <div class="plot-section">
                    <img src="feature_importance/{model_name}_feature_importance.png" alt="Feature Importance">
                    <img src="feature_importance/{model_name}_permutation_importance.png" alt="Permutation Importance">
                </div>
                
                <h2>SHAP Analysis</h2>
                <div class="plot-section">
                    <img src="shap/{model_name}_shap_summary.png" alt="SHAP Summary">
                    <img src="shap/{model_name}_shap_bar.png" alt="SHAP Bar Plot">
                    <img src="shap/{model_name}_shap_waterfall.png" alt="SHAP Waterfall">
                </div>
                
                <h2>Partial Dependence</h2>
                <div class="plot-section">
                    <img src="{model_name}_partial_dependence.png" alt="Partial Dependence">
                </div>
                
                <h2>Feature Names</h2>
                <ul>
            """
            
            for i, feature in enumerate(feature_names):
                html_content += f"<li>{i}: {feature}</li>"
            
            html_content += """
                </ul>
            </body>
            </html>
            """
            
            report_path = os.path.join(self.config.explainability_plots_dir, f'{model_name}_explainability_report.html')
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logging.info(f"Explainability report saved: {report_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def explain_model(self, model, X_train, X_test, y_test, feature_names, model_name):
        """
        Main method to generate all explainability plots and reports
        """
        try:
            logging.info(f"Starting explainability analysis for {model_name}")
            
            # Feature importance
            self.create_feature_importance_plot(model, feature_names, model_name)
            
            # Permutation importance
            self.create_permutation_importance(model, X_test, y_test, feature_names, model_name)
            
            # SHAP plots
            self.create_shap_plots(model, X_train, X_test, feature_names, model_name)
            
            # Partial dependence plots
            self.create_partial_dependence_plots(model, X_train, feature_names, model_name)
            
            # Generate report
            self.generate_explainability_report(model_name, feature_names)
            
            logging.info(f"Explainability analysis completed for {model_name}")
            
        except Exception as e:
            raise CustomException(e, sys)