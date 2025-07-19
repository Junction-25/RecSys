"""
URL patterns for the quotes app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import QuoteViewSet

router = DefaultRouter()
router.register(r'', QuoteViewSet)

urlpatterns = [
    path('', include(router.urls)),
]