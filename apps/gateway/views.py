"""
API Gateway views for routing and service management.
"""
import asyncio
import logging
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from drf_spectacular.utils import extend_schema
from asgiref.sync import sync_to_async

from .services import APIGateway, ServiceRegistry

logger = logging.getLogger(__name__)


class GatewayRoutingView(View):
    """
    Main gateway view for routing requests to microservices.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gateway = APIGateway()
    
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        """Handle all HTTP methods."""
        return super().dispatch(request, *args, **kwargs)
    
    async def _handle_request(self, request, path):
        """Handle the actual request routing."""
        try:
            response = await self.gateway.route_request(request, path)
            return response
        except Exception as e:
            logger.error(f"Gateway routing error: {e}")
            return JsonResponse(
                {'error': 'Gateway routing failed', 'details': str(e)},
                status=500
            )
    
    def get(self, request, path=''):
        """Handle GET requests."""
        return asyncio.run(self._handle_request(request, f"/{path}"))
    
    def post(self, request, path=''):
        """Handle POST requests."""
        return asyncio.run(self._handle_request(request, f"/{path}"))
    
    def put(self, request, path=''):
        """Handle PUT requests."""
        return asyncio.run(self._handle_request(request, f"/{path}"))
    
    def patch(self, request, path=''):
        """Handle PATCH requests."""
        return asyncio.run(self._handle_request(request, f"/{path}"))
    
    def delete(self, request, path=''):
        """Handle DELETE requests."""
        return asyncio.run(self._handle_request(request, f"/{path}"))


class ServiceDiscoveryView(APIView):
    """
    Service discovery and registry management.
    """
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service_registry = ServiceRegistry()
    
    @extend_schema(
        responses={200: dict},
        tags=['gateway']
    )
    def get(self, request):
        """Get information about all registered services."""
        service_info = self.service_registry.get_service_info()
        
        return Response({
            'gateway_status': 'operational',
            'service_registry': service_info,
            'routing_info': {
                'total_routes': sum(
                    len(service['routes']) 
                    for service in self.service_registry.services.values()
                ),
                'available_services': list(self.service_registry.services.keys())
            }
        })


class HealthCheckView(APIView):
    """
    Gateway health check and service monitoring.
    """
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service_registry = ServiceRegistry()
    
    @extend_schema(
        responses={200: dict},
        tags=['gateway']
    )
    def get(self, request):
        """Get health status of gateway and all services."""
        try:
            # Check all services health
            health_results = asyncio.run(
                self.service_registry.check_all_services_health()
            )
            
            # Calculate overall health
            healthy_services = sum(
                1 for result in health_results.values() 
                if result.get('status') == 'healthy'
            )
            total_services = len(health_results)
            
            overall_status = 'healthy' if healthy_services == total_services else 'degraded'
            if healthy_services == 0:
                overall_status = 'unhealthy'
            
            return Response({
                'gateway_status': 'operational',
                'overall_service_health': overall_status,
                'healthy_services': healthy_services,
                'total_services': total_services,
                'service_health': health_results,
                'timestamp': request.META.get('HTTP_X_REQUEST_TIME', 'unknown')
            })
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response(
                {
                    'gateway_status': 'error',
                    'error': str(e),
                    'overall_service_health': 'unknown'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GatewayMetricsView(APIView):
    """
    Gateway metrics and performance monitoring.
    """
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gateway = APIGateway()
    
    @extend_schema(
        responses={200: dict},
        tags=['gateway']
    )
    def get(self, request):
        """Get gateway performance metrics."""
        try:
            # Get circuit breaker states
            circuit_breaker_states = {}
            for service_name, service_info in self.gateway.circuit_breaker.services.items():
                circuit_breaker_states[service_name] = {
                    'state': service_info['state'],
                    'failures': service_info['failures'],
                    'last_failure': service_info['last_failure']
                }
            
            # Get service registry info
            service_info = self.gateway.service_registry.get_service_info()
            
            return Response({
                'gateway_metrics': {
                    'total_services': service_info['total_services'],
                    'circuit_breaker_states': circuit_breaker_states,
                    'rate_limiter': {
                        'requests_per_minute_limit': self.gateway.rate_limiter.requests_per_minute,
                        'window_size_seconds': self.gateway.rate_limiter.window_size
                    }
                },
                'service_performance': {
                    service_name: {
                        'status': service_data['status'],
                        'response_time_ms': service_data['response_time'],
                        'last_check': service_data['last_check']
                    }
                    for service_name, service_data in service_info['services'].items()
                },
                'routing_statistics': {
                    'total_routes_configured': sum(
                        service_data['routes_count'] 
                        for service_data in service_info['services'].values()
                    )
                }
            })
            
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return Response(
                {'error': 'Failed to retrieve metrics', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ServiceProxyView(APIView):
    """
    Direct service proxy for specific service calls.
    """
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gateway = APIGateway()
    
    @extend_schema(
        parameters=[
            {'name': 'service', 'in': 'path', 'required': True, 'schema': {'type': 'string'}},
            {'name': 'path', 'in': 'path', 'required': True, 'schema': {'type': 'string'}},
        ],
        responses={200: dict},
        tags=['gateway']
    )
    async def get(self, request, service_name, path=''):
        """Proxy GET request to specific service."""
        return await self._proxy_request(request, service_name, path)
    
    async def post(self, request, service_name, path=''):
        """Proxy POST request to specific service."""
        return await self._proxy_request(request, service_name, path)
    
    async def put(self, request, service_name, path=''):
        """Proxy PUT request to specific service."""
        return await self._proxy_request(request, service_name, path)
    
    async def delete(self, request, service_name, path=''):
        """Proxy DELETE request to specific service."""
        return await self._proxy_request(request, service_name, path)
    
    async def _proxy_request(self, request, service_name, path):
        """Handle the proxy request."""
        try:
            # Check if service exists
            if service_name not in self.gateway.service_registry.services:
                return Response(
                    {'error': f'Service {service_name} not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Build full path
            full_path = f"/api/v1/{service_name}/{path}"
            
            # Route the request
            response = await self.gateway.route_request(request, full_path)
            
            return response
            
        except Exception as e:
            logger.error(f"Service proxy error: {e}")
            return Response(
                {'error': 'Service proxy failed', 'details': str(e)},
                status=status.HTTP_502_BAD_GATEWAY
            )


class LoadBalancerView(APIView):
    """
    Load balancer configuration and management.
    """
    permission_classes = [IsAuthenticated]
    
    @extend_schema(
        responses={200: dict},
        tags=['gateway']
    )
    def get(self, request):
        """Get load balancer configuration."""
        return Response({
            'load_balancer': {
                'strategy': 'round_robin',
                'health_check_interval': 30,
                'failure_threshold': 3,
                'recovery_timeout': 60
            },
            'service_instances': {
                'properties': ['http://localhost:8001'],
                'contacts': ['http://localhost:8002'],
                'recommendations': ['http://localhost:8003'],
                'ai-agents': ['http://localhost:8004'],
                'analytics': ['http://localhost:8005'],
                'quotes': ['http://localhost:8006']
            },
            'configuration': {
                'timeout_seconds': 30,
                'retry_attempts': 3,
                'circuit_breaker_enabled': True
            }
        })
    
    @extend_schema(
        request=dict,
        responses={200: dict},
        tags=['gateway']
    )
    def post(self, request):
        """Update load balancer configuration."""
        try:
            # This would update load balancer settings
            # For now, return current configuration
            return Response({
                'message': 'Load balancer configuration updated',
                'status': 'success'
            })
            
        except Exception as e:
            return Response(
                {'error': 'Failed to update configuration', 'details': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )