"""
Custom exception handlers for the application.
"""
import logging
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    """
    Custom exception handler that provides consistent error responses.
    """
    # Call REST framework's default exception handler first
    response = exception_handler(exc, context)
    
    # Log the exception
    logger.error(f"Exception occurred: {exc}", exc_info=True, extra={
        'request': context.get('request'),
        'view': context.get('view'),
    })
    
    if response is not None:
        # Customize the response format
        custom_response_data = {
            'error': True,
            'message': 'An error occurred',
            'details': response.data,
            'status_code': response.status_code
        }
        
        # Handle specific error types
        if response.status_code == status.HTTP_404_NOT_FOUND:
            custom_response_data['message'] = 'Resource not found'
        elif response.status_code == status.HTTP_400_BAD_REQUEST:
            custom_response_data['message'] = 'Invalid request data'
        elif response.status_code == status.HTTP_401_UNAUTHORIZED:
            custom_response_data['message'] = 'Authentication required'
        elif response.status_code == status.HTTP_403_FORBIDDEN:
            custom_response_data['message'] = 'Permission denied'
        elif response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            custom_response_data['message'] = 'Internal server error'
        
        response.data = custom_response_data
    
    return response


class APIException(Exception):
    """Base API exception class."""
    
    def __init__(self, message, status_code=status.HTTP_400_BAD_REQUEST, details=None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(APIException):
    """Exception for validation errors."""
    
    def __init__(self, message, details=None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, details)


class NotFoundError(APIException):
    """Exception for resource not found errors."""
    
    def __init__(self, message="Resource not found", details=None):
        super().__init__(message, status.HTTP_404_NOT_FOUND, details)


class PermissionDeniedError(APIException):
    """Exception for permission denied errors."""
    
    def __init__(self, message="Permission denied", details=None):
        super().__init__(message, status.HTTP_403_FORBIDDEN, details)