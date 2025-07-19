"""
Gateway-specific settings for Smart Real Estate AI platform.
"""
from .settings import *

# Override INSTALLED_APPS to only include essential apps
INSTALLED_APPS = [
    # Django core apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',
    'django_filters',
    'drf_spectacular',
    
    # Local apps - only core and gateway
    'apps.core',
    'apps.gateway',
]

# Disable migrations for faster startup
MIGRATION_MODULES = {
    'core': None,
    'gateway': None,
}

# Simplified logging for gateway
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}