"""
Core URLs for health checks and system status
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.health_check, name='health_check'),
    path('api/', views.api_health, name='api_health'),
]