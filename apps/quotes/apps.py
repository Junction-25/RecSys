"""
Quotes app configuration
"""
from django.apps import AppConfig


class QuotesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.quotes'
    verbose_name = 'Quotes'