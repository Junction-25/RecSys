"""
Filters for the contacts app.
"""
import django_filters
from .models import Contact


class ContactFilter(django_filters.FilterSet):
    """Filter set for contacts."""
    
    # Budget range filters
    min_budget_min = django_filters.NumberFilter(field_name='budget_min', lookup_expr='gte')
    max_budget_min = django_filters.NumberFilter(field_name='budget_min', lookup_expr='lte')
    min_budget_max = django_filters.NumberFilter(field_name='budget_max', lookup_expr='gte')
    max_budget_max = django_filters.NumberFilter(field_name='budget_max', lookup_expr='lte')
    
    # Area range filters
    min_area_min = django_filters.NumberFilter(field_name='desired_area_min', lookup_expr='gte')
    max_area_min = django_filters.NumberFilter(field_name='desired_area_min', lookup_expr='lte')
    min_area_max = django_filters.NumberFilter(field_name='desired_area_max', lookup_expr='gte')
    max_area_max = django_filters.NumberFilter(field_name='desired_area_max', lookup_expr='lte')
    
    # Room range filters
    min_rooms_min = django_filters.NumberFilter(field_name='rooms_min', lookup_expr='gte')
    max_rooms_min = django_filters.NumberFilter(field_name='rooms_min', lookup_expr='lte')
    min_rooms_max = django_filters.NumberFilter(field_name='rooms_max', lookup_expr='gte')
    max_rooms_max = django_filters.NumberFilter(field_name='rooms_max', lookup_expr='lte')
    
    # Location filters
    preferred_location = django_filters.CharFilter(field_name='preferred_locations', lookup_expr='icontains')
    
    # Feature preference filters
    prefers_parking = django_filters.BooleanFilter(field_name='prefers_parking')
    prefers_garden = django_filters.BooleanFilter(field_name='prefers_garden')
    prefers_balcony = django_filters.BooleanFilter(field_name='prefers_balcony')
    prefers_elevator = django_filters.BooleanFilter(field_name='prefers_elevator')
    prefers_furnished = django_filters.BooleanFilter(field_name='prefers_furnished')
    
    class Meta:
        model = Contact
        fields = [
            'property_type', 'priority', 'status',
            'min_budget_min', 'max_budget_min', 'min_budget_max', 'max_budget_max',
            'min_area_min', 'max_area_min', 'min_area_max', 'max_area_max',
            'min_rooms_min', 'max_rooms_min', 'min_rooms_max', 'max_rooms_max',
            'preferred_location', 'prefers_parking', 'prefers_garden',
            'prefers_balcony', 'prefers_elevator', 'prefers_furnished'
        ]