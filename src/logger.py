import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class CustomLogger:
    def __init__(self, name=None, log_level=logging.INFO):
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers with proper formatting"""
        
        # Create logs directory
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # File handler with rotation
        log_file = f"{datetime.now().strftime('%Y_%m_%d')}.log"
        log_file_path = os.path.join(logs_dir, log_file)
        
        # Rotating file handler (max 10MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s'
        )
        
        # Set formatters
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error log file handler
        error_log_file = f"error_{datetime.now().strftime('%Y_%m_%d')}.log"
        error_log_path = os.path.join(logs_dir, error_log_file)
        
        error_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        self.logger.addHandler(error_handler)
    
    def get_logger(self):
        return self.logger

# Create default logger instance
default_logger = CustomLogger("MLProject").get_logger()

def get_logger(name=None):
    """Get a logger instance with custom configuration"""
    return CustomLogger(name).get_logger()

# For backward compatibility
logging = default_logger