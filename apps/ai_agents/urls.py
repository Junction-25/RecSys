"""
URL patterns for the AI agents app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AIAgentViewSet

router = DefaultRouter()
router.register(r'', AIAgentViewSet, basename='ai-agents')

urlpatterns = [
    path('', include(router.urls)),
]