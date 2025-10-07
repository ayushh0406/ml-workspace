import os
import sys
import yaml
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger("ConfigManager")

@dataclass
class ModelTrainingConfig:
    """Model training configuration"""
    random_state: int
    test_size: float
    cross_validation_folds: int
    models: Dict[str, Dict[str, Any]]

@dataclass
class DataProcessingConfig:
    """Data processing configuration"""
    missing_value_strategy: Dict[str, str]
    outlier_detection: Dict[str, Any]
    scaling: Dict[str, str]

@dataclass
class DataValidationConfig:
    """Data validation configuration"""
    score_ranges: Dict[str, list]
    missing_value_threshold: float
    outlier_threshold: float
    expected_dtypes: Dict[str, str]

@dataclass
class ModelEvaluationConfig:
    """Model evaluation configuration"""
    regression_metrics: list
    plot_settings: Dict[str, Any]
    selection_criteria: Dict[str, Any]

@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    shap: Dict[str, Any]
    feature_importance: Dict[str, list]
    partial_dependence: Dict[str, int]

@dataclass
class PathsConfig:
    """File paths configuration"""
    data: Dict[str, str]
    models: Dict[str, str]
    reports: Dict[str, str]
    logs: Dict[str, str]

class ConfigurationManager:
    """
    Configuration manager to handle YAML configuration files
    """
    
    def __init__(self, config_path: str = "config/params.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize configuration objects
        self.model_training = self._create_model_training_config()
        self.data_processing = self._create_data_processing_config()
        self.data_validation = self._create_data_validation_config()
        self.model_evaluation = self._create_model_evaluation_config()
        self.explainability = self._create_explainability_config()
        self.paths = self._create_paths_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_model_training_config(self) -> ModelTrainingConfig:
        """Create model training configuration object"""
        try:
            config = self.config['model_training']
            return ModelTrainingConfig(
                random_state=config['random_state'],
                test_size=config['test_size'],
                cross_validation_folds=config['cross_validation_folds'],
                models=config['models']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_data_processing_config(self) -> DataProcessingConfig:
        """Create data processing configuration object"""
        try:
            config = self.config['data_processing']
            return DataProcessingConfig(
                missing_value_strategy=config['missing_value_strategy'],
                outlier_detection=config['outlier_detection'],
                scaling=config['scaling']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_data_validation_config(self) -> DataValidationConfig:
        """Create data validation configuration object"""
        try:
            config = self.config['data_validation']
            return DataValidationConfig(
                score_ranges=config['score_ranges'],
                missing_value_threshold=config['missing_value_threshold'],
                outlier_threshold=config['outlier_threshold'],
                expected_dtypes=config['expected_dtypes']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Create model evaluation configuration object"""
        try:
            config = self.config['model_evaluation']
            return ModelEvaluationConfig(
                regression_metrics=config['regression_metrics'],
                plot_settings=config['plot_settings'],
                selection_criteria=config['selection_criteria']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_explainability_config(self) -> ExplainabilityConfig:
        """Create explainability configuration object"""
        try:
            config = self.config['explainability']
            return ExplainabilityConfig(
                shap=config['shap'],
                feature_importance=config['feature_importance'],
                partial_dependence=config['partial_dependence']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_paths_config(self) -> PathsConfig:
        """Create paths configuration object"""
        try:
            config = self.config['paths']
            return PathsConfig(
                data=config['data'],
                models=config['models'],
                reports=config['reports'],
                logs=config['logs']
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model"""
        try:
            if model_name not in self.model_training.models:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            return self.model_training.models[model_name]
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value"""
        try:
            if section not in self.config:
                self.config[section] = {}
                
            self.config[section][key] = value
            
            # Save updated configuration
            self.save_config()
            
            logger.info(f"Configuration updated: {section}.{key} = {value}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_config(self, output_path: str = None):
        """Save configuration to YAML file"""
        try:
            output_path = output_path or self.config_path
            
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model_selection_criteria(self) -> Dict[str, Any]:
        """Get model selection criteria"""
        return self.model_evaluation.selection_criteria
    
    def get_data_validation_thresholds(self) -> Dict[str, float]:
        """Get data validation thresholds"""
        return {
            'missing_value_threshold': self.data_validation.missing_value_threshold,
            'outlier_threshold': self.data_validation.outlier_threshold
        }
    
    def get_file_paths(self) -> PathsConfig:
        """Get file paths configuration"""
        return self.paths
    
    def validate_configuration(self) -> bool:
        """Validate configuration completeness and correctness"""
        try:
            required_sections = [
                'model_training', 'data_processing', 'data_validation',
                'model_evaluation', 'explainability', 'paths'
            ]
            
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Validate specific requirements
            if self.model_training.test_size <= 0 or self.model_training.test_size >= 1:
                raise ValueError("test_size must be between 0 and 1")
            
            if self.model_training.cross_validation_folds < 2:
                raise ValueError("cross_validation_folds must be at least 2")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise CustomException(e, sys)

# Global configuration manager instance
config_manager = ConfigurationManager()

def get_config() -> ConfigurationManager:
    """Get global configuration manager instance"""
    return config_manager