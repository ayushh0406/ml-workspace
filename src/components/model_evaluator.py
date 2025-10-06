import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelEvaluationConfig:
    evaluation_report_file_path: str = os.path.join("artifacts", "model_evaluation_report.html")
    evaluation_plots_dir: str = os.path.join("artifacts", "evaluation_plots")
    metrics_file_path: str = os.path.join("artifacts", "model_metrics.csv")

class ModelEvaluator:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()
        os.makedirs(self.model_evaluation_config.evaluation_plots_dir, exist_ok=True)
    
    def calculate_regression_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate comprehensive regression metrics
        """
        try:
            metrics = {
                'model_name': model_name,
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
                'adjusted_r2': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - 1 - 1)
            }
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_prediction_vs_actual_plot(self, y_true, y_pred, model_name, save_path):
        """
        Create prediction vs actual values plot
        """
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Prediction vs Actual', 'Residuals Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Prediction vs Actual
            fig.add_trace(
                go.Scatter(
                    x=y_true, 
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=6, opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Residuals plot
            residuals = y_true - y_pred
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='green', size=6, opacity=0.6)
                ),
                row=1, col=2
            )
            
            # Zero line for residuals
            fig.add_trace(
                go.Scatter(
                    x=[min(y_pred), max(y_pred)],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Actual Values", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
            fig.update_yaxes(title_text="Residuals", row=1, col=2)
            
            fig.update_layout(
                title=f'{model_name} - Model Performance Analysis',
                showlegend=True,
                height=500,
                width=1000
            )
            
            fig.write_html(save_path)
            logging.info(f"Prediction vs actual plot saved: {save_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_residual_analysis_plots(self, y_true, y_pred, model_name, save_path):
        """
        Create comprehensive residual analysis plots
        """
        try:
            residuals = y_true - y_pred
            standardized_residuals = residuals / np.std(residuals)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Residuals Distribution',
                    'Q-Q Plot of Residuals',
                    'Residuals vs Fitted Values',
                    'Residuals Histogram'
                )
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(residuals))),
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='blue', size=4)
                ),
                row=1, col=1
            )
            
            # Q-Q plot approximation
            from scipy import stats
            sorted_residuals = np.sort(standardized_residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='red', size=4)
                ),
                row=1, col=2
            )
            
            # Perfect Q-Q line
            fig.add_trace(
                go.Scatter(
                    x=[-3, 3],
                    y=[-3, 3],
                    mode='lines',
                    name='Normal Line',
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=2
            )
            
            # Residuals vs fitted
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals vs Fitted',
                    marker=dict(color='green', size=4)
                ),
                row=2, col=1
            )
            
            # Histogram of residuals
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Residuals Histogram',
                    nbinsx=30,
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{model_name} - Residual Analysis',
                showlegend=True,
                height=800,
                width=1000
            )
            
            fig.write_html(save_path)
            logging.info(f"Residual analysis plots saved: {save_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_feature_importance_plot(self, model, feature_names, model_name, save_path):
        """
        Create feature importance plot if model supports it
        """
        try:
            importance = None
            
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            
            if importance is not None:
                # Create feature importance dataframe
                feature_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=feature_imp_df['importance'],
                    y=feature_imp_df['feature'],
                    orientation='h',
                    marker_color='skyblue'
                ))
                
                fig.update_layout(
                    title=f'{model_name} - Feature Importance',
                    xaxis_title='Importance',
                    yaxis_title='Features',
                    height=400,
                    width=800
                )
                
                fig.write_html(save_path)
                logging.info(f"Feature importance plot saved: {save_path}")
                
        except Exception as e:
            logging.warning(f"Could not create feature importance plot for {model_name}: {str(e)}")
    
    def create_model_comparison_plot(self, all_metrics, save_path):
        """
        Create model comparison visualization
        """
        try:
            metrics_df = pd.DataFrame(all_metrics)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score', 'RMSE', 'MAE', 'MAPE'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # R² Score
            fig.add_trace(
                go.Bar(x=metrics_df['model_name'], y=metrics_df['r2_score'], name='R² Score'),
                row=1, col=1
            )
            
            # RMSE
            fig.add_trace(
                go.Bar(x=metrics_df['model_name'], y=metrics_df['rmse'], name='RMSE'),
                row=1, col=2
            )
            
            # MAE
            fig.add_trace(
                go.Bar(x=metrics_df['model_name'], y=metrics_df['mae'], name='MAE'),
                row=2, col=1
            )
            
            # MAPE
            fig.add_trace(
                go.Bar(x=metrics_df['model_name'], y=metrics_df['mape'], name='MAPE'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Model Performance Comparison',
                showlegend=False,
                height=600,
                width=1000
            )
            
            fig.write_html(save_path)
            logging.info(f"Model comparison plot saved: {save_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def generate_evaluation_report(self, all_metrics, best_model_name, save_path):
        """
        Generate comprehensive HTML evaluation report
        """
        try:
            metrics_df = pd.DataFrame(all_metrics)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .best-model {{ background-color: #e8f5e8; }}
                    .metric-value {{ font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Model Evaluation Report</h1>
                <h2>Best Model: {best_model_name}</h2>
                
                <h2>Model Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Model Name</th>
                        <th>R² Score</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>MSE</th>
                        <th>MAPE (%)</th>
                        <th>Adjusted R²</th>
                    </tr>
            """
            
            for _, row in metrics_df.iterrows():
                row_class = "best-model" if row['model_name'] == best_model_name else ""
                html_content += f"""
                    <tr class="{row_class}">
                        <td>{row['model_name']}</td>
                        <td class="metric-value">{row['r2_score']:.4f}</td>
                        <td class="metric-value">{row['rmse']:.4f}</td>
                        <td class="metric-value">{row['mae']:.4f}</td>
                        <td class="metric-value">{row['mse']:.4f}</td>
                        <td class="metric-value">{row['mape']:.2f}%</td>
                        <td class="metric-value">{row['adjusted_r2']:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Evaluation Metrics Explanation</h2>
                <ul>
                    <li><strong>R² Score:</strong> Coefficient of determination (higher is better, max = 1)</li>
                    <li><strong>RMSE:</strong> Root Mean Square Error (lower is better)</li>
                    <li><strong>MAE:</strong> Mean Absolute Error (lower is better)</li>
                    <li><strong>MSE:</strong> Mean Square Error (lower is better)</li>
                    <li><strong>MAPE:</strong> Mean Absolute Percentage Error (lower is better)</li>
                    <li><strong>Adjusted R²:</strong> R² adjusted for number of predictors</li>
                </ul>
                
                <h2>Visualization Files</h2>
                <p>Check the evaluation_plots directory for detailed visualizations including:</p>
                <ul>
                    <li>Prediction vs Actual plots for each model</li>
                    <li>Residual analysis plots</li>
                    <li>Feature importance plots (where applicable)</li>
                    <li>Model comparison charts</li>
                </ul>
            </body>
            </html>
            """
            
            with open(save_path, 'w') as f:
                f.write(html_content)
            
            logging.info(f"Evaluation report saved: {save_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_models_comprehensive(self, models, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Comprehensive model evaluation with visualizations
        """
        try:
            logging.info("Starting comprehensive model evaluation")
            
            all_metrics = []
            
            for model_name, model in models.items():
                logging.info(f"Evaluating {model_name}")
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics for test set
                test_metrics = self.calculate_regression_metrics(y_test, y_test_pred, model_name)
                all_metrics.append(test_metrics)
                
                # Create individual model plots
                pred_vs_actual_path = os.path.join(
                    self.model_evaluation_config.evaluation_plots_dir,
                    f"{model_name.replace(' ', '_').lower()}_prediction_analysis.html"
                )
                self.create_prediction_vs_actual_plot(y_test, y_test_pred, model_name, pred_vs_actual_path)
                
                residual_analysis_path = os.path.join(
                    self.model_evaluation_config.evaluation_plots_dir,
                    f"{model_name.replace(' ', '_').lower()}_residual_analysis.html"
                )
                self.create_residual_analysis_plots(y_test, y_test_pred, model_name, residual_analysis_path)
                
                # Feature importance plot (if applicable)
                if feature_names:
                    feature_imp_path = os.path.join(
                        self.model_evaluation_config.evaluation_plots_dir,
                        f"{model_name.replace(' ', '_').lower()}_feature_importance.html"
                    )
                    self.create_feature_importance_plot(model, feature_names, model_name, feature_imp_path)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(self.model_evaluation_config.metrics_file_path, index=False)
            
            # Create model comparison plot
            comparison_plot_path = os.path.join(
                self.model_evaluation_config.evaluation_plots_dir,
                "model_comparison.html"
            )
            self.create_model_comparison_plot(all_metrics, comparison_plot_path)
            
            # Find best model
            best_model_name = max(all_metrics, key=lambda x: x['r2_score'])['model_name']
            
            # Generate comprehensive report
            self.generate_evaluation_report(all_metrics, best_model_name, 
                                          self.model_evaluation_config.evaluation_report_file_path)
            
            logging.info("Comprehensive model evaluation completed")
            
            return all_metrics, best_model_name
            
        except Exception as e:
            raise CustomException(e, sys)