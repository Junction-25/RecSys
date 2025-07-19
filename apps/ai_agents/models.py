"""
AI Agents models
"""
from django.db import models

# Placeholder model for AI Agents
class AIAgent(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name