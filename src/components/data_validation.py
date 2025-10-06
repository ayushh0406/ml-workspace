import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataValidationConfig:
    validation_report_file_path: str = os.path.join("artifacts", "validation_report.txt")
    validation_plots_dir: str = os.path.join("artifacts", "validation_plots")

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
        
    def validate_data_types(self, df: pd.DataFrame, expected_dtypes: dict) -> dict:
        """
        Validate data types of columns
        """
        try:
            validation_results = {}
            
            for column, expected_dtype in expected_dtypes.items():
                if column in df.columns:
                    actual_dtype = df[column].dtype
                    validation_results[column] = {
                        'expected': expected_dtype,
                        'actual': str(actual_dtype),
                        'valid': str(actual_dtype) == expected_dtype or 
                                (expected_dtype == 'numeric' and pd.api.types.is_numeric_dtype(df[column]))
                    }
                else:
                    validation_results[column] = {
                        'expected': expected_dtype,
                        'actual': 'missing',
                        'valid': False
                    }
            
            return validation_results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_missing_values(self, df: pd.DataFrame) -> dict:
        """
        Check for missing values in the dataset
        """
        try:
            missing_info = {}
            
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100
                
                missing_info[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': round(missing_percentage, 2),
                    'has_missing': missing_count > 0
                }
            
            return missing_info
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def detect_outliers(self, df: pd.DataFrame, numerical_columns: list, method='iqr') -> dict:
        """
        Detect outliers using IQR or Z-score method
        """
        try:
            outlier_info = {}
            
            for column in numerical_columns:
                if column in df.columns:
                    if method == 'iqr':
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                        
                    elif method == 'zscore':
                        z_scores = np.abs(stats.zscore(df[column].dropna()))
                        outliers = df[z_scores > 3]
                    
                    outlier_info[column] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': round((len(outliers) / len(df)) * 100, 2),
                        'method': method,
                        'outlier_indices': outliers.index.tolist()
                    }
            
            return outlier_info
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def validate_data_ranges(self, df: pd.DataFrame, range_constraints: dict) -> dict:
        """
        Validate if data falls within expected ranges
        """
        try:
            range_validation = {}
            
            for column, constraints in range_constraints.items():
                if column in df.columns:
                    min_val = constraints.get('min')
                    max_val = constraints.get('max')
                    
                    violations = 0
                    if min_val is not None:
                        violations += (df[column] < min_val).sum()
                    if max_val is not None:
                        violations += (df[column] > max_val).sum()
                    
                    range_validation[column] = {
                        'expected_range': f"[{min_val}, {max_val}]",
                        'actual_range': f"[{df[column].min()}, {df[column].max()}]",
                        'violations': violations,
                        'violation_percentage': round((violations / len(df)) * 100, 2)
                    }
            
            return range_validation
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_validation_plots(self, df: pd.DataFrame, numerical_columns: list):
        """
        Create visualization plots for data validation
        """
        try:
            os.makedirs(self.data_validation_config.validation_plots_dir, exist_ok=True)
            
            # Distribution plots for numerical columns
            for column in numerical_columns:
                if column in df.columns:
                    plt.figure(figsize=(12, 4))
                    
                    # Histogram
                    plt.subplot(1, 3, 1)
                    plt.hist(df[column].dropna(), bins=30, alpha=0.7)
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                    
                    # Box plot
                    plt.subplot(1, 3, 2)
                    plt.boxplot(df[column].dropna())
                    plt.title(f'Box Plot of {column}')
                    plt.ylabel(column)
                    
                    # Q-Q plot
                    plt.subplot(1, 3, 3)
                    stats.probplot(df[column].dropna(), dist="norm", plot=plt)
                    plt.title(f'Q-Q Plot of {column}')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.data_validation_config.validation_plots_dir, 
                                           f'{column}_validation_plots.png'))
                    plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numerical_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_validation_config.validation_plots_dir, 
                                   'correlation_heatmap.png'))
            plt.close()
            
            logging.info(f"Validation plots saved to {self.data_validation_config.validation_plots_dir}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def generate_validation_report(self, validation_results: dict, file_path: str):
        """
        Generate a comprehensive validation report
        """
        try:
            with open(file_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("DATA VALIDATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                for section, results in validation_results.items():
                    f.write(f"{section.upper().replace('_', ' ')}\n")
                    f.write("-" * 40 + "\n")
                    
                    if isinstance(results, dict):
                        for key, value in results.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{results}\n")
                    
                    f.write("\n")
            
            logging.info(f"Validation report saved to {file_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_validation(self, train_path: str, test_path: str):
        """
        Main method to initiate data validation process
        """
        try:
            logging.info("Starting data validation process")
            
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data loaded successfully for validation")
            
            # Define expected data types
            expected_dtypes = {
                'gender': 'object',
                'race_ethnicity': 'object',
                'parental_level_of_education': 'object',
                'lunch': 'object',
                'test_preparation_course': 'object',
                'math_score': 'numeric',
                'reading_score': 'numeric',
                'writing_score': 'numeric'
            }
            
            # Define numerical columns
            numerical_columns = ['math_score', 'reading_score', 'writing_score']
            
            # Define expected ranges
            range_constraints = {
                'math_score': {'min': 0, 'max': 100},
                'reading_score': {'min': 0, 'max': 100},
                'writing_score': {'min': 0, 'max': 100}
            }
            
            validation_results = {}
            
            # Validate training data
            logging.info("Validating training data")
            validation_results['train_data_types'] = self.validate_data_types(train_df, expected_dtypes)
            validation_results['train_missing_values'] = self.check_missing_values(train_df)
            validation_results['train_outliers'] = self.detect_outliers(train_df, numerical_columns)
            validation_results['train_range_validation'] = self.validate_data_ranges(train_df, range_constraints)
            
            # Validate test data
            logging.info("Validating test data")
            validation_results['test_data_types'] = self.validate_data_types(test_df, expected_dtypes)
            validation_results['test_missing_values'] = self.check_missing_values(test_df)
            validation_results['test_outliers'] = self.detect_outliers(test_df, numerical_columns)
            validation_results['test_range_validation'] = self.validate_data_ranges(test_df, range_constraints)
            
            # Create validation plots
            logging.info("Creating validation plots")
            self.create_validation_plots(train_df, numerical_columns)
            
            # Generate validation report
            logging.info("Generating validation report")
            self.generate_validation_report(validation_results, 
                                          self.data_validation_config.validation_report_file_path)
            
            logging.info("Data validation completed successfully")
            
            return validation_results
            
        except Exception as e:
            raise CustomException(e, sys)