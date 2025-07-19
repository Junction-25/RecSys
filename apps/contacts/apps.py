"""
Contacts app configuration
"""
from django.apps import AppConfig


class ContactsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.contacts'
    
    def ready(self):
        """Import signals when the app is ready."""
        import apps.contacts.signals
    verbose_name = 'Contacts'