import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Enhanced configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    feature_selector_path: str = os.path.join('artifacts', "feature_selector.pkl")
    scaler_info_path: str = os.path.join('artifacts', "scaler_info.json")
    
    # Transformation parameters
    numerical_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    scaling_method: str = "standard"  # standard, minmax, robust
    handle_outliers: bool = True
    apply_feature_selection: bool = False
    n_features_to_select: int = 10
    apply_pca: bool = False
    pca_variance_threshold: float = 0.95

class DataTransformation:
    """Enhanced data transformation component"""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.feature_names_ = None
        self.target_column = "math_score"
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically identify numerical and categorical columns"""
        try:
            # Remove target column from features
            feature_columns = [col for col in df.columns if col != self.target_column]
            
            numerical_columns = []
            categorical_columns = []
            
            for col in feature_columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Check if it's actually categorical (like scores with limited range)
                    unique_values = df[col].nunique()
                    if unique_values > 10 or df[col].dtype == 'float64':
                        numerical_columns.append(col)
                    else:
                        categorical_columns.append(col)
                else:
                    categorical_columns.append(col)
            
            logging.info(f"Identified numerical columns: {numerical_columns}")
            logging.info(f"Identified categorical columns: {categorical_columns}")
            
            return numerical_columns, categorical_columns
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_numerical_pipeline(self) -> Pipeline:
        """Create enhanced numerical preprocessing pipeline"""
        try:
            steps = []
            
            # Imputation
            if self.data_transformation_config.numerical_imputation_strategy == "knn":
                steps.append(("imputer", KNNImputer(n_neighbors=5)))
            else:
                steps.append(("imputer", SimpleImputer(
                    strategy=self.data_transformation_config.numerical_imputation_strategy
                )))
            
            # Outlier handling (optional)
            if self.data_transformation_config.handle_outliers:
                from sklearn.preprocessing import RobustScaler
                steps.append(("outlier_scaler", RobustScaler()))
            
            # Scaling
            scaling_method = self.data_transformation_config.scaling_method
            if scaling_method == "standard":
                steps.append(("scaler", StandardScaler()))
            elif scaling_method == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            elif scaling_method == "robust":
                steps.append(("scaler", RobustScaler()))
            
            return Pipeline(steps=steps)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_categorical_pipeline(self) -> Pipeline:
        """Create enhanced categorical preprocessing pipeline"""
        try:
            steps = [
                ("imputer", SimpleImputer(
                    strategy=self.data_transformation_config.categorical_imputation_strategy
                )),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler", StandardScaler())
            ]
            
            return Pipeline(steps=steps)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_feature_selector(self, X: np.ndarray, y: np.ndarray) -> SelectKBest:
        """Create feature selector based on statistical tests"""
        try:
            selector = SelectKBest(
                score_func=f_regression,
                k=min(self.data_transformation_config.n_features_to_select, X.shape[1])
            )
            selector.fit(X, y)
            
            logging.info(f"Feature selection completed. Selected {selector.k_} features")
            return selector
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_transformer_object(self, df: pd.DataFrame = None) -> ColumnTransformer:
        """
        Create comprehensive data transformation pipeline
        """
        try:
            if df is not None:
                numerical_columns, categorical_columns = self.identify_column_types(df)
            else:
                # Default columns for backward compatibility
                numerical_columns = ["writing_score", "reading_score"]
                categorical_columns = [
                    "gender",
                    "race_ethnicity", 
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",
                ]
            
            # Create pipelines
            num_pipeline = self.create_numerical_pipeline()
            cat_pipeline = self.create_categorical_pipeline()
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ],
                remainder='passthrough',  # Keep other columns
                n_jobs=-1  # Use all available cores
            )
            
            # Store feature names for later use
            self.feature_names_ = numerical_columns + categorical_columns
            
            return preprocessor

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)