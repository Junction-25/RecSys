"""
Core views for the real estate recommendation system
"""
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import time


@require_http_methods(["GET"])
@csrf_exempt
def health_check(request):
    """Simple health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'Real Estate Recommendation System',
        'version': '1.0.0'
    })


@require_http_methods(["GET"])
@csrf_exempt
def api_health(request):
    """API health check with more details"""
    try:
        # Test database connection
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'
    
    try:
        # Test Redis connection
        from django.core.cache import cache
        cache.set('health_check', 'ok', 10)
        redis_status = 'connected' if cache.get('health_check') == 'ok' else 'error'
    except Exception as e:
        redis_status = f'error: {str(e)}'
    
    return JsonResponse({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'Real Estate Recommendation System',
        'version': '1.0.0',
        'components': {
            'database': db_status,
            'redis': redis_status,
            'api': 'operational'
        }
    })