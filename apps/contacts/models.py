"""
Contact models for the real estate system
"""
from django.db import models
from django.core.validators import EmailValidator


class Contact(models.Model):
    """Contact model for potential buyers/renters"""
    
    # Status choices
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    CONVERTED = 'converted'
    
    STATUS_CHOICES = [
        (ACTIVE, 'Active'),
        (INACTIVE, 'Inactive'),
        (CONVERTED, 'Converted'),
    ]
    
    # Basic Information
    external_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    name = models.CharField(max_length=200)
    email = models.EmailField(validators=[EmailValidator()])
    phone = models.CharField(max_length=20, blank=True)
    
    # Budget
    budget_min = models.DecimalField(max_digits=15, decimal_places=2)
    budget_max = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Location Preferences
    preferred_locations = models.JSONField(default=list, blank=True)
    
    # Property Preferences
    property_type = models.CharField(max_length=20, default='apartment')
    desired_area_min = models.PositiveIntegerField(default=50)
    desired_area_max = models.PositiveIntegerField(default=200)
    rooms_min = models.PositiveIntegerField(default=1)
    rooms_max = models.PositiveIntegerField(default=4)
    
    # Feature Preferences
    prefers_parking = models.BooleanField(default=True)
    prefers_garden = models.BooleanField(default=False)
    prefers_balcony = models.BooleanField(default=True)
    prefers_elevator = models.BooleanField(default=False)
    prefers_furnished = models.BooleanField(default=False)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=ACTIVE)
    
    # Embeddings (for AI recommendations)
    embedding = models.BinaryField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'apps_contacts_contact'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['property_type']),
            models.Index(fields=['budget_min', 'budget_max']),
        ]
    
    def __str__(self):
        return f"{self.name} - {self.email}"