import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from dataclasses import dataclass
from typing import Tuple, Optional

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")
    validation_data_path: str = os.path.join('artifacts', "validation.csv")
    
    # Data source configurations
    data_source: str = 'notebook/data/stud.csv'
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify_column: Optional[str] = None

class DataIngestion:
    """Enhanced data ingestion component with advanced features"""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def validate_data_source(self, data_path: str) -> bool:
        """Validate if data source exists and is accessible"""
        try:
            if urlparse(data_path).scheme:  # URL
                import requests
                response = requests.head(data_path)
                return response.status_code == 200
            else:  # Local file
                return Path(data_path).exists()
        except Exception:
            return False
    
    def read_data_from_source(self, data_path: str) -> pd.DataFrame:
        """Read data with error handling and multiple format support"""
        try:
            file_extension = Path(data_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(data_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(data_path)
            elif file_extension == '.json':
                df = pd.read_json(data_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                # Default to CSV
                df = pd.read_csv(data_path)
            
            logging.info(f"Successfully read data from {data_path}")
            logging.info(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            raise CustomException(f"Error reading data from {data_path}: {str(e)}", sys)
    
    def perform_basic_data_checks(self, df: pd.DataFrame) -> dict:
        """Perform basic data quality checks"""
        try:
            checks = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            }
            
            logging.info(f"Data quality checks: {checks}")
            return checks
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_train_test_validation_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, test, and validation splits"""
        try:
            # First split: train+val vs test
            if self.ingestion_config.stratify_column and self.ingestion_config.stratify_column in df.columns:
                # Stratified split for imbalanced datasets
                stratify_data = df[self.ingestion_config.stratify_column]
                train_val, test = train_test_split(
                    df, 
                    test_size=self.ingestion_config.test_size,
                    random_state=self.ingestion_config.random_state,
                    stratify=stratify_data
                )
            else:
                train_val, test = train_test_split(
                    df,
                    test_size=self.ingestion_config.test_size,
                    random_state=self.ingestion_config.random_state
                )
            
            # Second split: train vs validation
            val_size_adjusted = self.ingestion_config.validation_size / (1 - self.ingestion_config.test_size)
            train, validation = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=self.ingestion_config.random_state
            )
            
            logging.info(f"Data split completed:")
            logging.info(f"Train: {len(train)} rows ({len(train)/len(df)*100:.1f}%)")
            logging.info(f"Validation: {len(validation)} rows ({len(validation)/len(df)*100:.1f}%)")
            logging.info(f"Test: {len(test)} rows ({len(test)/len(df)*100:.1f}%)")
            
            return train, validation, test
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, raw_df: pd.DataFrame):
        """Save all datasets to specified paths"""
        try:
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save datasets
            raw_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_df.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("All datasets saved successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> Tuple[str, str, str]:
        """Enhanced data ingestion process"""
        logging.info("Starting enhanced data ingestion process")
        
        try:
            # Validate data source
            if not self.validate_data_source(self.ingestion_config.data_source):
                raise FileNotFoundError(f"Data source not found: {self.ingestion_config.data_source}")
            
            # Read data
            df = self.read_data_from_source(self.ingestion_config.data_source)
            
            # Perform basic data checks
            data_quality = self.perform_basic_data_checks(df)
            
            # Check for minimum data requirements
            if data_quality['total_rows'] < 10:
                raise ValueError("Dataset too small for meaningful analysis")
            
            # Create splits
            train_df, val_df, test_df = self.create_train_test_validation_split(df)
            
            # Save datasets
            self.save_datasets(train_df, val_df, test_df, df)
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.validation_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


