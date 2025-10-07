import sys
import traceback
import logging
from datetime import datetime
from src.logger import get_logger

# Get logger for exceptions
exception_logger = get_logger("ExceptionHandler")

def error_message_detail(error, error_detail: sys):
    """
    Extract detailed error information including stack trace
    """
    _, _, exc_tb = error_detail.exc_info()
    
    # Get full stack trace
    stack_trace = traceback.format_exception(type(error), error, exc_tb)
    stack_trace_str = ''.join(stack_trace)
    
    # Extract file and line info
    file_name = exc_tb.tb_frame.f_code.co_filename
    function_name = exc_tb.tb_frame.f_code.co_name
    line_number = exc_tb.tb_lineno
    
    error_message = f"""
    ==================== ERROR DETAILS ====================
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    File: {file_name}
    Function: {function_name}
    Line: {line_number}
    Error Type: {type(error).__name__}
    Error Message: {str(error)}
    
    Stack Trace:
    {stack_trace_str}
    ======================================================
    """
    
    return error_message

class CustomException(Exception):
    """
    Enhanced custom exception with detailed logging and context
    """
    def __init__(self, error_message, error_detail: sys, context: dict = None):
        super().__init__(error_message)
        self.original_error = error_message
        self.context = context or {}
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log the error immediately
        self._log_error()
    
    def _log_error(self):
        """Log the error with context"""
        try:
            context_str = f"Context: {self.context}" if self.context else ""
            exception_logger.error(f"{self.error_message}\n{context_str}")
        except Exception:
            # Fallback logging if custom logger fails
            logging.error(f"Failed to log exception: {self.original_error}")
    
    def add_context(self, key: str, value):
        """Add context information to the exception"""
        self.context[key] = value
        return self
    
    def get_short_message(self):
        """Get a shorter version of the error message"""
        return f"[{self.timestamp}] {type(self.original_error).__name__}: {str(self.original_error)}"
    
    def __str__(self):
        return self.error_message

def handle_exception(func):
    """
    Decorator to handle exceptions with custom logging
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CustomException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            # Convert regular exceptions to custom exceptions
            context = {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            raise CustomException(e, sys, context)
    
    return wrapper
    


        