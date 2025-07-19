"""
Property models for the real estate system
"""
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Property(models.Model):
    """Property model with all necessary fields"""
    
    # Property Types
    APARTMENT = 'apartment'
    HOUSE = 'house'
    VILLA = 'villa'
    STUDIO = 'studio'
    OFFICE = 'office'
    LAND = 'land'
    COMMERCIAL = 'commercial'
    
    PROPERTY_TYPE_CHOICES = [
        (APARTMENT, 'Apartment'),
        (HOUSE, 'House'),
        (VILLA, 'Villa'),
        (STUDIO, 'Studio'),
        (OFFICE, 'Office'),
        (LAND, 'Land'),
        (COMMERCIAL, 'Commercial'),
    ]
    
    # Status choices
    AVAILABLE = 'available'
    SOLD = 'sold'
    RENTED = 'rented'
    PENDING = 'pending'
    
    STATUS_CHOICES = [
        (AVAILABLE, 'Available'),
        (SOLD, 'Sold'),
        (RENTED, 'Rented'),
        (PENDING, 'Pending'),
    ]
    
    # Basic Information
    external_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Financial
    price = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Location
    city = models.CharField(max_length=100)
    district = models.CharField(max_length=100, blank=True)
    address = models.TextField()
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    
    # Property Details
    area = models.PositiveIntegerField(help_text="Area in square meters")
    rooms = models.PositiveIntegerField(default=0)
    property_type = models.CharField(max_length=20, choices=PROPERTY_TYPE_CHOICES, default=APARTMENT)
    
    # Features
    has_parking = models.BooleanField(default=False)
    has_garden = models.BooleanField(default=False)
    has_balcony = models.BooleanField(default=False)
    has_elevator = models.BooleanField(default=False)
    is_furnished = models.BooleanField(default=False)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=AVAILABLE)
    
    # Embeddings (for AI recommendations)
    embedding = models.BinaryField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'apps_properties_property'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['city', 'property_type']),
            models.Index(fields=['price']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.city}"