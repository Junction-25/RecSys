"""
URL patterns for the API Gateway.
"""
from django.urls import path, re_path
from .views import (
    GatewayRoutingView,
    ServiceDiscoveryView,
    HealthCheckView,
    GatewayMetricsView,
    ServiceProxyView,
    LoadBalancerView
)

urlpatterns = [
    # Gateway management endpoints
    path('discovery/', ServiceDiscoveryView.as_view(), name='service-discovery'),
    path('health/', HealthCheckView.as_view(), name='gateway-health'),
    path('metrics/', GatewayMetricsView.as_view(), name='gateway-metrics'),
    path('load-balancer/', LoadBalancerView.as_view(), name='load-balancer'),
    
    # Direct service proxy
    re_path(
        r'^proxy/(?P<service_name>[\w-]+)/(?P<path>.*)$',
        ServiceProxyView.as_view(),
        name='service-proxy'
    ),
    
    # Main gateway routing (catch-all)
    re_path(
        r'^route/(?P<path>.*)$',
        GatewayRoutingView.as_view(),
        name='gateway-routing'
    ),
]