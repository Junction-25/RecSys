"""
Core middleware for the real estate system
"""
import time
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(MiddlewareMixin):
    """Middleware to log requests"""
    
    def process_request(self, request):
        request.start_time = time.time()
        return None
    
    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
        return response


class MetricsMiddleware(MiddlewareMixin):
    """Middleware to collect metrics"""
    
    def process_request(self, request):
        request.start_time = time.time()
        return None
    
    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            # Here you could send metrics to monitoring system
            # For now, just log
            if duration > 1.0:  # Log slow requests
                logger.warning(f"Slow request: {request.method} {request.path} - {duration:.3f}s")
        return response