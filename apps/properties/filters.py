"""
Filters for the properties app.
"""
import django_filters
from .models import Property


class PropertyFilter(django_filters.FilterSet):
    """Filter set for properties."""
    
    # Price range filters
    min_price = django_filters.NumberFilter(field_name='price', lookup_expr='gte')
    max_price = django_filters.NumberFilter(field_name='price', lookup_expr='lte')
    
    # Area range filters
    min_area = django_filters.NumberFilter(field_name='area', lookup_expr='gte')
    max_area = django_filters.NumberFilter(field_name='area', lookup_expr='lte')
    
    # Room range filters
    min_rooms = django_filters.NumberFilter(field_name='rooms', lookup_expr='gte')
    max_rooms = django_filters.NumberFilter(field_name='rooms', lookup_expr='lte')
    
    # Location filters
    city = django_filters.CharFilter(field_name='city', lookup_expr='icontains')
    district = django_filters.CharFilter(field_name='district', lookup_expr='icontains')
    
    # Feature filters
    has_parking = django_filters.BooleanFilter(field_name='has_parking')
    has_garden = django_filters.BooleanFilter(field_name='has_garden')
    has_balcony = django_filters.BooleanFilter(field_name='has_balcony')
    has_elevator = django_filters.BooleanFilter(field_name='has_elevator')
    is_furnished = django_filters.BooleanFilter(field_name='is_furnished')
    
    class Meta:
        model = Property
        fields = [
            'property_type', 'status', 'listing_type',
            'min_price', 'max_price', 'min_area', 'max_area',
            'min_rooms', 'max_rooms', 'city', 'district',
            'has_parking', 'has_garden', 'has_balcony',
            'has_elevator', 'is_furnished'
        ]