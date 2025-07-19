"""
URLs for the recommendations app
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router and register viewsets
router = DefaultRouter()
router.register(r'', views.RecommendationViewSet, basename='recommendations')

urlpatterns = [
    path('', include(router.urls)),
]