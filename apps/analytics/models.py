"""
Analytics models
"""
from django.db import models

# Placeholder model for Analytics
class AnalyticsEvent(models.Model):
    event_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.event_type