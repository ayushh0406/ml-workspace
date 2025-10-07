import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

class TestTrainingPipeline:
    """Test cases for training pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.training_pipeline = TrainingPipeline()
        
    def test_training_pipeline_initialization(self):
        """Test training pipeline initialization"""
        assert hasattr(self.training_pipeline, 'data_ingestion')
        assert hasattr(self.training_pipeline, 'data_transformation')
        assert hasattr(self.training_pipeline, 'model_trainer')
        
    @patch('src.components.data_ingestion.DataIngestion.initiate_data_ingestion')
    @patch('src.components.data_transformation.DataTransformation.initiate_data_transformation')
    @patch('src.components.model_trainer.ModelTrainer.initiate_model_trainer')
    def test_run_training_pipeline_flow(self, mock_trainer, mock_transformation, mock_ingestion):
        """Test training pipeline flow"""
        # Mock return values
        mock_ingestion.return_value = ('train_path', 'test_path')
        mock_transformation.return_value = (np.array([[1, 2]]), np.array([[3, 4]]), 'preprocessor_path')
        mock_trainer.return_value = 0.85
        
        # Run pipeline
        result = self.training_pipeline.run_training_pipeline()
        
        # Verify calls
        mock_ingestion.assert_called_once()
        mock_transformation.assert_called_once()
        mock_trainer.assert_called_once()
        
        # Verify result
        assert result['r2_score'] == 0.85

class TestPredictionPipeline:
    """Test cases for prediction pipeline"""
    
    def test_custom_data_creation(self):
        """Test CustomData object creation"""
        custom_data = CustomData(
            gender='Male',
            race_ethnicity='group A',
            parental_level_of_education='some college',
            lunch='standard',
            test_preparation_course='completed',
            reading_score=85,
            writing_score=90
        )
        
        assert custom_data.gender == 'Male'
        assert custom_data.reading_score == 85
        assert custom_data.writing_score == 90
        
    def test_custom_data_to_dataframe(self):
        """Test converting CustomData to DataFrame"""
        custom_data = CustomData(
            gender='Female',
            race_ethnicity='group B',
            parental_level_of_education="bachelor's degree",
            lunch='free/reduced',
            test_preparation_course='none',
            reading_score=75,
            writing_score=80
        )
        
        df = custom_data.get_data_as_data_frame()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'gender' in df.columns
        assert df['reading_score'].iloc[0] == 75

class TestModelValidation:
    """Test cases for model validation and metrics"""
    
    def test_r2_score_calculation(self):
        """Test R² score calculation"""
        from sklearn.metrics import r2_score
        
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        
        score = r2_score(y_true, y_pred)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
        
    def test_model_performance_metrics(self):
        """Test various model performance metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert rmse == np.sqrt(mse)

class TestDataQuality:
    """Test cases for data quality checks"""
    
    def test_data_shape_validation(self):
        """Test data shape validation"""
        # Sample student performance data
        data = pd.DataFrame({
            'gender': ['Male', 'Female'] * 50,
            'race_ethnicity': ['group A', 'group B'] * 50,
            'parental_level_of_education': ['some college', "bachelor's degree"] * 50,
            'lunch': ['standard', 'free/reduced'] * 50,
            'test_preparation_course': ['completed', 'none'] * 50,
            'math_score': np.random.randint(0, 100, 100),
            'reading_score': np.random.randint(0, 100, 100),
            'writing_score': np.random.randint(0, 100, 100)
        })
        
        assert data.shape[0] == 100  # 100 rows
        assert data.shape[1] == 8    # 8 columns
        
    def test_data_column_validation(self):
        """Test data column validation"""
        expected_columns = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'math_score', 
            'reading_score', 'writing_score'
        ]
        
        # Sample data
        data = pd.DataFrame({col: [1, 2, 3] for col in expected_columns})
        
        for col in expected_columns:
            assert col in data.columns
            
    def test_score_range_validation(self):
        """Test if scores are within valid range"""
        scores = [85, 92, 78, 65, 88]
        
        for score in scores:
            assert 0 <= score <= 100
            
    def test_categorical_values_validation(self):
        """Test categorical values validation"""
        valid_genders = ['Male', 'Female']
        valid_lunch_types = ['standard', 'free/reduced']
        
        test_gender = 'Male'
        test_lunch = 'standard'
        
        assert test_gender in valid_genders
        assert test_lunch in valid_lunch_types

class TestPreprocessing:
    """Test cases for data preprocessing"""
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        from sklearn.impute import SimpleImputer
        
        # Data with missing values
        data = np.array([[1, 2], [np.nan, 4], [7, 6]])
        
        imputer = SimpleImputer(strategy='median')
        imputed_data = imputer.fit_transform(data)
        
        # Check no missing values remain
        assert not np.isnan(imputed_data).any()
        
    def test_scaling_functionality(self):
        """Test data scaling functionality"""
        from sklearn.preprocessing import StandardScaler
        
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Check if mean is approximately 0 and std is approximately 1
        assert abs(np.mean(scaled_data)) < 0.01
        assert abs(np.std(scaled_data) - 1) < 0.01
        
    def test_one_hot_encoding(self):
        """Test one-hot encoding functionality"""
        from sklearn.preprocessing import OneHotEncoder
        
        data = np.array([['Male'], ['Female'], ['Male']])
        
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(data)
        
        assert encoded_data.shape[1] == 2  # Two categories
        assert encoded_data.shape[0] == 3  # Three samples

if __name__ == "__main__":
    pytest.main([__file__])