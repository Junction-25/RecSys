"""
API Gateway middleware for request processing and routing.
"""
import time
import uuid
import logging
from django.utils.deprecation import MiddlewareMixin
from django.http import JsonResponse
from django.core.cache import cache

logger = logging.getLogger(__name__)


class GatewayRequestMiddleware(MiddlewareMixin):
    """
    Middleware for API Gateway request processing.
    """
    
    def process_request(self, request):
        """Process incoming requests through the gateway."""
        # Add request ID for tracing
        request.gateway_request_id = str(uuid.uuid4())
        request.gateway_start_time = time.time()
        
        # Add gateway headers
        request.META['HTTP_X_GATEWAY_REQUEST_ID'] = request.gateway_request_id
        request.META['HTTP_X_REQUEST_TIME'] = str(int(time.time()))
        
        # Log gateway request
        logger.info(
            f"Gateway request started",
            extra={
                'gateway_request_id': request.gateway_request_id,
                'method': request.method,
                'path': request.path,
                'client_ip': self._get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', 'unknown')
            }
        )
    
    def process_response(self, request, response):
        """Process responses from the gateway."""
        if hasattr(request, 'gateway_start_time'):
            duration = (time.time() - request.gateway_start_time) * 1000
            
            # Add gateway headers to response
            response['X-Gateway-Request-ID'] = getattr(request, 'gateway_request_id', 'unknown')
            response['X-Gateway-Response-Time'] = f"{duration:.2f}ms"
            response['X-Gateway-Version'] = '1.0.0'
            
            # Log gateway response
            logger.info(
                f"Gateway request completed",
                extra={
                    'gateway_request_id': getattr(request, 'gateway_request_id', 'unknown'),
                    'status_code': response.status_code,
                    'duration_ms': duration,
                    'response_size': len(response.content) if hasattr(response, 'content') else 0
                }
            )
        
        return response
    
    def _get_client_ip(self, request):
        """Get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')


class ServiceRoutingMiddleware(MiddlewareMixin):
    """
    Middleware for automatic service routing based on URL patterns.
    """
    
    def __init__(self, get_response):
        """Initialize the middleware."""
        super().__init__(get_response)
        self.service_patterns = {
            '/api/v1/properties/': 'properties',
            '/api/v1/contacts/': 'contacts',
            '/api/v1/recommendations/': 'recommendations',
            '/api/v1/ai-agents/': 'ai-agents',
            '/api/v1/analytics/': 'analytics',
            '/api/v1/quotes/': 'quotes',
        }
    
    def process_request(self, request):
        """Check if request should be routed through gateway."""
        path = request.path
        
        # Check if this is a gateway management request
        if path.startswith('/gateway/'):
            return None  # Let it pass through normally
        
        # Check if this matches a service pattern
        for pattern, service_name in self.service_patterns.items():
            if path.startswith(pattern):
                # Add service routing information
                request.target_service = service_name
                request.service_path = path
                
                logger.debug(
                    f"Request routed to service: {service_name}",
                    extra={
                        'path': path,
                        'service': service_name,
                        'method': request.method
                    }
                )
                break
        
        return None


class CircuitBreakerMiddleware(MiddlewareMixin):
    """
    Circuit breaker middleware for service resilience.
    """
    
    def __init__(self, get_response):
        """Initialize circuit breaker middleware."""
        super().__init__(get_response)
        self.failure_threshold = 5
        self.recovery_timeout = 60
    
    def process_request(self, request):
        """Check circuit breaker status for target service."""
        if hasattr(request, 'target_service'):
            service_name = request.target_service
            
            # Check circuit breaker status
            circuit_key = f"circuit_breaker:{service_name}"
            circuit_data = cache.get(circuit_key, {'failures': 0, 'state': 'closed'})
            
            if circuit_data['state'] == 'open':
                # Check if recovery timeout has passed
                last_failure = circuit_data.get('last_failure', 0)
                if (time.time() - last_failure) < self.recovery_timeout:
                    logger.warning(f"Circuit breaker open for service: {service_name}")
                    return JsonResponse(
                        {
                            'error': 'Service temporarily unavailable',
                            'service': service_name,
                            'retry_after': self.recovery_timeout
                        },
                        status=503
                    )
                else:
                    # Move to half-open state
                    circuit_data['state'] = 'half-open'
                    cache.set(circuit_key, circuit_data, 300)
        
        return None
    
    def process_response(self, request, response):
        """Update circuit breaker based on response."""
        if hasattr(request, 'target_service'):
            service_name = request.target_service
            circuit_key = f"circuit_breaker:{service_name}"
            circuit_data = cache.get(circuit_key, {'failures': 0, 'state': 'closed'})
            
            if response.status_code >= 500:
                # Record failure
                circuit_data['failures'] += 1
                circuit_data['last_failure'] = time.time()
                
                if circuit_data['failures'] >= self.failure_threshold:
                    circuit_data['state'] = 'open'
                    logger.error(f"Circuit breaker opened for service: {service_name}")
                
                cache.set(circuit_key, circuit_data, 300)
            
            elif response.status_code < 400:
                # Record success
                circuit_data['failures'] = 0
                circuit_data['state'] = 'closed'
                cache.set(circuit_key, circuit_data, 300)
        
        return response


class LoadBalancingMiddleware(MiddlewareMixin):
    """
    Load balancing middleware for distributing requests across service instances.
    """
    
    def __init__(self, get_response):
        """Initialize load balancing middleware."""
        super().__init__(get_response)
        self.service_instances = {
            'properties': ['http://localhost:8001'],
            'contacts': ['http://localhost:8002'],
            'recommendations': ['http://localhost:8003'],
            'ai-agents': ['http://localhost:8004'],
            'analytics': ['http://localhost:8005'],
            'quotes': ['http://localhost:8006']
        }
    
    def process_request(self, request):
        """Select service instance for load balancing."""
        if hasattr(request, 'target_service'):
            service_name = request.target_service
            
            if service_name in self.service_instances:
                instances = self.service_instances[service_name]
                
                # Simple round-robin selection
                instance_key = f"lb_counter:{service_name}"
                counter = cache.get(instance_key, 0)
                selected_instance = instances[counter % len(instances)]
                
                # Update counter
                cache.set(instance_key, counter + 1, 3600)
                
                # Add selected instance to request
                request.target_instance = selected_instance
                
                logger.debug(
                    f"Load balancer selected instance: {selected_instance}",
                    extra={
                        'service': service_name,
                        'instance': selected_instance,
                        'total_instances': len(instances)
                    }
                )
        
        return None