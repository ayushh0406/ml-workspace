import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_validation import DataValidation, DataValidationConfig
from src.exception import CustomException

class TestDataIngestion:
    """Test cases for data ingestion component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_ingestion = DataIngestion()
        
    def test_data_ingestion_config(self):
        """Test data ingestion configuration"""
        config = DataIngestionConfig()
        assert hasattr(config, 'train_data_path')
        assert hasattr(config, 'test_data_path')
        assert hasattr(config, 'raw_data_path')
        
    def test_read_data_from_source(self):
        """Test reading data from source"""
        # Create sample data
        sample_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'race_ethnicity': ['group A', 'group B', 'group A'],
            'parental_level_of_education': ['some college', "bachelor's degree", 'high school'],
            'lunch': ['standard', 'free/reduced', 'standard'],
            'test_preparation_course': ['completed', 'none', 'completed'],
            'math_score': [72, 69, 90],
            'reading_score': [72, 90, 95],
            'writing_score': [74, 88, 93]
        })
        
        # Mock the data reading
        with patch('pandas.read_csv', return_value=sample_data):
            result = pd.read_csv('dummy_path.csv')
            assert len(result) == 3
            assert 'math_score' in result.columns
            
    def test_train_test_split_functionality(self):
        """Test if train-test split works correctly"""
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(sample_data, test_size=0.2, random_state=42)
        
        assert len(train) == 80
        assert len(test) == 20

class TestDataValidation:
    """Test cases for data validation component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_validation = DataValidation()
        
    def test_data_validation_config(self):
        """Test data validation configuration"""
        config = DataValidationConfig()
        assert hasattr(config, 'validation_report_file_path')
        assert hasattr(config, 'validation_plots_dir')
        
    def test_validate_data_types(self):
        """Test data type validation"""
        # Sample data
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'string_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3]
        })
        
        expected_dtypes = {
            'numeric_col': 'numeric',
            'string_col': 'object',
            'float_col': 'numeric'
        }
        
        result = self.data_validation.validate_data_types(df, expected_dtypes)
        
        assert result['numeric_col']['valid'] == True
        assert result['string_col']['valid'] == True
        assert result['float_col']['valid'] == True
        
    def test_check_missing_values(self):
        """Test missing values detection"""
        # Sample data with missing values
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': [1, 2, 3, 4],
            'col3': [None, None, 3, 4]
        })
        
        result = self.data_validation.check_missing_values(df)
        
        assert result['col1']['missing_count'] == 1
        assert result['col1']['missing_percentage'] == 25.0
        assert result['col2']['missing_count'] == 0
        assert result['col3']['missing_count'] == 2
        
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method"""
        # Sample data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 100)
        outlier_data = np.concatenate([normal_data, [150, -50]])  # Add outliers
        
        df = pd.DataFrame({'score': outlier_data})
        
        result = self.data_validation.detect_outliers(df, ['score'], method='iqr')
        
        assert result['score']['outlier_count'] > 0
        assert result['score']['method'] == 'iqr'
        
    def test_validate_data_ranges(self):
        """Test data range validation"""
        df = pd.DataFrame({
            'score': [85, 92, 78, 105, -5]  # 105 and -5 are out of range
        })
        
        range_constraints = {
            'score': {'min': 0, 'max': 100}
        }
        
        result = self.data_validation.validate_data_ranges(df, range_constraints)
        
        assert result['score']['violations'] == 2  # Two values out of range

class TestDataTransformation:
    """Test cases for data transformation component"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_transformation = DataTransformation()
        
    def test_data_transformation_config(self):
        """Test data transformation configuration"""
        config = DataTransformationConfig()
        assert hasattr(config, 'preprocessor_obj_file_path')
        
    def test_get_data_transformer_object(self):
        """Test data transformer object creation"""
        transformer = self.data_transformation.get_data_transformer_object()
        
        # Check if transformer is created
        assert transformer is not None
        assert hasattr(transformer, 'fit_transform')
        
    def test_numerical_pipeline(self):
        """Test numerical preprocessing pipeline"""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Create numerical pipeline
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Test data
        data = np.array([[1, 2], [3, 4], [np.nan, 6]])
        
        transformed = num_pipeline.fit_transform(data)
        
        assert transformed.shape == (3, 2)
        assert not np.isnan(transformed).any()
        
    def test_categorical_pipeline(self):
        """Test categorical preprocessing pipeline"""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        # Create categorical pipeline
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])
        
        # Test data
        data = np.array([['A'], ['B'], ['A'], [None]])
        
        transformed = cat_pipeline.fit_transform(data)
        
        assert transformed.shape[0] == 4  # Same number of rows

class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_save_and_load_object(self):
        """Test save and load object functionality"""
        from src.utils import save_object, load_object
        import tempfile
        import os
        
        # Create a test object
        test_obj = {'test': 'data', 'numbers': [1, 2, 3]}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            file_path = tmp_file.name
        
        try:
            # Save object
            save_object(file_path, test_obj)
            
            # Load object
            loaded_obj = load_object(file_path)
            
            # Verify
            assert loaded_obj == test_obj
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)

class TestExceptionHandling:
    """Test cases for custom exception handling"""
    
    def test_custom_exception_creation(self):
        """Test custom exception creation"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            custom_exc = CustomException(e, sys)
            
            assert isinstance(custom_exc, CustomException)
            assert "Test error" in str(custom_exc)
            
    def test_exception_context(self):
        """Test exception context addition"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            custom_exc = CustomException(e, sys)
            custom_exc.add_context("test_key", "test_value")
            
            assert "test_key" in custom_exc.context
            assert custom_exc.context["test_key"] == "test_value"

if __name__ == "__main__":
    pytest.main([__file__])