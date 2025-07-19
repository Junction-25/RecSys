"""
Quotes models
"""
from django.db import models

# Placeholder model for Quotes
class Quote(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.content[:50]