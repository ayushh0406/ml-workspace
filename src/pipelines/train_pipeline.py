import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_training_pipeline(self):
        """
        Execute the complete training pipeline with data validation
        """
        try:
            logging.info("Training pipeline started")
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Validation
            logging.info("Step 2: Data Validation")
            validation_results = self.data_validation.initiate_data_validation(train_data_path, test_data_path)
            
            # Check validation results
            critical_issues = []
            for section, results in validation_results.items():
                if 'missing_values' in section:
                    for column, info in results.items():
                        if info['missing_percentage'] > 20:  # More than 20% missing
                            critical_issues.append(f"High missing values in {column}: {info['missing_percentage']}%")
                
                if 'outliers' in section:
                    for column, info in results.items():
                        if info['outlier_percentage'] > 10:  # More than 10% outliers
                            critical_issues.append(f"High outliers in {column}: {info['outlier_percentage']}%")
            
            if critical_issues:
                logging.warning(f"Data validation found issues: {critical_issues}")
                # In production, you might want to stop here or apply data cleaning
            
            # Step 3: Data Transformation
            logging.info("Step 3: Data Transformation")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 4: Model Training with Comprehensive Evaluation
            logging.info("Step 4: Model Training with Comprehensive Evaluation")
            r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Training pipeline completed successfully with R2 score: {r2_score}")
            
            return {
                'r2_score': r2_score,
                'preprocessor_path': preprocessor_path,
                'validation_results': validation_results
            }
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run_training_pipeline()
        print(f"Training completed with R2 score: {results['r2_score']}")
        print("Check artifacts/ directory for:")
        print("- Model evaluation report (HTML)")
        print("- Validation plots and reports")
        print("- Model comparison visualizations")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        print(f"Error: {str(e)}")