"""
API Gateway service for routing requests to microservices.
"""
import httpx
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
import json

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Service registry for managing microservice endpoints and health.
    """
    
    def __init__(self):
        """Initialize the service registry."""
        self.services = {
            'properties': {
                'name': 'Properties Service',
                'base_url': settings.MICROSERVICES.get('PROPERTIES_SERVICE_URL', 'http://localhost:8001'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/properties/',
                    '/api/v1/properties/{id}/',
                    '/api/v1/properties/search/',
                    '/api/v1/properties/statistics/',
                    '/api/v1/properties/compare/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            },
            'contacts': {
                'name': 'Contacts Service',
                'base_url': settings.MICROSERVICES.get('CONTACTS_SERVICE_URL', 'http://localhost:8002'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/contacts/',
                    '/api/v1/contacts/{id}/',
                    '/api/v1/contacts/search/',
                    '/api/v1/contacts/statistics/',
                    '/api/v1/contacts/{id}/activate/',
                    '/api/v1/contacts/{id}/deactivate/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            },
            'recommendations': {
                'name': 'Recommendations Service',
                'base_url': settings.MICROSERVICES.get('RECOMMENDATIONS_SERVICE_URL', 'http://localhost:8003'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/recommendations/property/{id}/',
                    '/api/v1/recommendations/contact/{id}/',
                    '/api/v1/recommendations/comprehensive-analysis/',
                    '/api/v1/recommendations/vector/property/{id}/',
                    '/api/v1/recommendations/vector/contact/{id}/',
                    '/api/v1/recommendations/compare-methods/',
                    '/api/v1/recommendations/metrics/',
                    '/api/v1/recommendations/generate-embeddings/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            },
            'ai-agents': {
                'name': 'AI Agents Service',
                'base_url': settings.MICROSERVICES.get('AI_AGENTS_SERVICE_URL', 'http://localhost:8004'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/ai-agents/analyze-property/{id}/',
                    '/api/v1/ai-agents/analyze-contact/{id}/',
                    '/api/v1/ai-agents/generate-description/{id}/',
                    '/api/v1/ai-agents/explain-match/',
                    '/api/v1/ai-agents/capabilities/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            },
            'analytics': {
                'name': 'Analytics Service',
                'base_url': settings.MICROSERVICES.get('ANALYTICS_SERVICE_URL', 'http://localhost:8005'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/analytics/dashboard/',
                    '/api/v1/analytics/market-trends/',
                    '/api/v1/analytics/location-insights/',
                    '/api/v1/analytics/performance-metrics/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            },
            'quotes': {
                'name': 'Quotes Service',
                'base_url': settings.MICROSERVICES.get('QUOTES_SERVICE_URL', 'http://localhost:8006'),
                'health_endpoint': '/health/',
                'routes': [
                    '/api/v1/quotes/',
                    '/api/v1/quotes/{id}/',
                    '/api/v1/quotes/generate-property-quote/',
                    '/api/v1/quotes/generate-comparison-quote/',
                    '/api/v1/quotes/{id}/download-pdf/',
                    '/api/v1/quotes/statistics/',
                ],
                'status': 'unknown',
                'last_check': None,
                'response_time': None
            }
        }
    
    def get_service_for_route(self, path: str) -> Optional[Dict]:
        """
        Find the appropriate service for a given route.
        
        Args:
            path: Request path
            
        Returns:
            Service configuration or None
        """
        for service_name, service_config in self.services.items():
            for route_pattern in service_config['routes']:
                # Simple pattern matching (can be enhanced with regex)
                if self._matches_route(path, route_pattern):
                    return {
                        'name': service_name,
                        'config': service_config
                    }
        return None
    
    def _matches_route(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a route pattern.
        
        Args:
            path: Request path
            pattern: Route pattern with {id} placeholders
            
        Returns:
            Boolean indicating match
        """
        # Convert pattern to regex-like matching
        pattern_parts = pattern.split('/')
        path_parts = path.split('/')
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for pattern_part, path_part in zip(pattern_parts, path_parts):
            if pattern_part.startswith('{') and pattern_part.endswith('}'):
                # This is a parameter, skip validation
                continue
            elif pattern_part != path_part:
                return False
        
        return True
    
    async def check_service_health(self, service_name: str) -> Dict:
        """
        Check the health of a specific service.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            Health status information
        """
        if service_name not in self.services:
            return {'status': 'not_found', 'error': 'Service not registered'}
        
        service = self.services[service_name]
        health_url = f"{service['base_url']}{service['health_endpoint']}"
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                service['status'] = 'healthy'
                service['response_time'] = response_time
                service['last_check'] = time.time()
                
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'last_check': service['last_check']
                }
            else:
                service['status'] = 'unhealthy'
                service['last_check'] = time.time()
                
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'last_check': service['last_check']
                }
                
        except Exception as e:
            service['status'] = 'unreachable'
            service['last_check'] = time.time()
            
            return {
                'status': 'unreachable',
                'error': str(e),
                'last_check': service['last_check']
            }
    
    async def check_all_services_health(self) -> Dict:
        """
        Check health of all registered services.
        
        Returns:
            Health status for all services
        """
        health_checks = {}
        
        # Run health checks concurrently
        tasks = []
        for service_name in self.services.keys():
            task = self.check_service_health(service_name)
            tasks.append((service_name, task))
        
        # Wait for all health checks to complete
        for service_name, task in tasks:
            health_checks[service_name] = await task
        
        return health_checks
    
    def get_service_info(self) -> Dict:
        """
        Get information about all registered services.
        
        Returns:
            Service registry information
        """
        return {
            'total_services': len(self.services),
            'services': {
                name: {
                    'name': config['name'],
                    'base_url': config['base_url'],
                    'status': config['status'],
                    'last_check': config['last_check'],
                    'response_time': config['response_time'],
                    'routes_count': len(config['routes'])
                }
                for name, config in self.services.items()
            }
        }


class APIGateway:
    """
    API Gateway for routing requests to appropriate microservices.
    """
    
    def __init__(self):
        """Initialize the API Gateway."""
        self.service_registry = ServiceRegistry()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
    
    async def route_request(self, request, path: str) -> JsonResponse:
        """
        Route a request to the appropriate microservice.
        
        Args:
            request: Django request object
            path: Request path
            
        Returns:
            JsonResponse with the result
        """
        # Check rate limiting
        if not self.rate_limiter.allow_request(request):
            return JsonResponse(
                {'error': 'Rate limit exceeded', 'retry_after': 60},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Find appropriate service
        service_info = self.service_registry.get_service_for_route(path)
        if not service_info:
            return JsonResponse(
                {'error': 'Service not found for this route'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        service_name = service_info['name']
        service_config = service_info['config']
        
        # Check circuit breaker
        if not self.circuit_breaker.allow_request(service_name):
            return JsonResponse(
                {'error': 'Service temporarily unavailable'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Build target URL
        target_url = f"{service_config['base_url']}{path}"
        
        try:
            # Forward request to microservice
            response_data = await self._forward_request(
                request, target_url, service_name
            )
            
            # Record successful request
            self.circuit_breaker.record_success(service_name)
            
            return JsonResponse(response_data, safe=False)
            
        except Exception as e:
            # Record failed request
            self.circuit_breaker.record_failure(service_name)
            
            logger.error(f"Error routing request to {service_name}: {e}")
            
            return JsonResponse(
                {'error': 'Service request failed', 'details': str(e)},
                status=status.HTTP_502_BAD_GATEWAY
            )
    
    async def _forward_request(self, request, target_url: str, service_name: str) -> Any:
        """
        Forward request to target microservice.
        
        Args:
            request: Django request object
            target_url: Target service URL
            service_name: Name of the target service
            
        Returns:
            Response data from the service
        """
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'X-Gateway-Service': service_name,
            'X-Request-ID': getattr(request, 'request_id', 'unknown')
        }
        
        # Add authentication headers if present
        if hasattr(request, 'META') and 'HTTP_AUTHORIZATION' in request.META:
            headers['Authorization'] = request.META['HTTP_AUTHORIZATION']
        
        # Prepare request data
        request_data = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            if hasattr(request, 'body') and request.body:
                try:
                    request_data = json.loads(request.body)
                except json.JSONDecodeError:
                    request_data = request.body.decode('utf-8')
        
        # Make request to microservice
        async with httpx.AsyncClient(timeout=30.0) as client:
            if request.method == 'GET':
                response = await client.get(
                    target_url,
                    headers=headers,
                    params=request.GET.dict()
                )
            elif request.method == 'POST':
                response = await client.post(
                    target_url,
                    headers=headers,
                    json=request_data,
                    params=request.GET.dict()
                )
            elif request.method == 'PUT':
                response = await client.put(
                    target_url,
                    headers=headers,
                    json=request_data,
                    params=request.GET.dict()
                )
            elif request.method == 'PATCH':
                response = await client.patch(
                    target_url,
                    headers=headers,
                    json=request_data,
                    params=request.GET.dict()
                )
            elif request.method == 'DELETE':
                response = await client.delete(
                    target_url,
                    headers=headers,
                    params=request.GET.dict()
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {request.method}")
        
        # Handle response
        if response.status_code >= 400:
            raise Exception(f"Service returned {response.status_code}: {response.text}")
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return {'data': response.text}


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.services = {}
    
    def allow_request(self, service_name: str) -> bool:
        """
        Check if request should be allowed for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Boolean indicating if request is allowed
        """
        if service_name not in self.services:
            self.services[service_name] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        service = self.services[service_name]
        current_time = time.time()
        
        if service['state'] == 'open':
            # Check if recovery timeout has passed
            if (current_time - service['last_failure']) > self.recovery_timeout:
                service['state'] = 'half-open'
                return True
            return False
        
        return True
    
    def record_success(self, service_name: str):
        """Record a successful request."""
        if service_name in self.services:
            self.services[service_name]['failures'] = 0
            self.services[service_name]['state'] = 'closed'
    
    def record_failure(self, service_name: str):
        """Record a failed request."""
        if service_name not in self.services:
            self.services[service_name] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }
        
        service = self.services[service_name]
        service['failures'] += 1
        service['last_failure'] = time.time()
        
        if service['failures'] >= self.failure_threshold:
            service['state'] = 'open'
            logger.warning(f"Circuit breaker opened for service: {service_name}")


class RateLimiter:
    """
    Rate limiter for API Gateway.
    """
    
    def __init__(self, requests_per_minute: int = 1000):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute window
    
    def allow_request(self, request) -> bool:
        """
        Check if request should be allowed based on rate limiting.
        
        Args:
            request: Django request object
            
        Returns:
            Boolean indicating if request is allowed
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Use cache for rate limiting
        cache_key = f"rate_limit:{client_ip}"
        current_requests = cache.get(cache_key, 0)
        
        if current_requests >= self.requests_per_minute:
            return False
        
        # Increment counter
        cache.set(cache_key, current_requests + 1, self.window_size)
        return True
    
    def _get_client_ip(self, request) -> str:
        """Get client IP address from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')