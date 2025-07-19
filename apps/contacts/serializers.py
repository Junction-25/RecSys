"""
Serializers for the contacts app.
"""
from rest_framework import serializers
from .models import Contact


class ContactSerializer(serializers.ModelSerializer):
    """Serializer for contacts."""
    budget_range_display = serializers.ReadOnlyField()
    area_range_display = serializers.ReadOnlyField()
    rooms_range_display = serializers.ReadOnlyField()
    preferences_list = serializers.ReadOnlyField()
    
    class Meta:
        model = Contact
        fields = [
            'id', 'external_id', 'name', 'email', 'phone',
            'preferred_locations', 'budget_min', 'budget_max',
            'desired_area_min', 'desired_area_max', 'property_type',
            'rooms_min', 'rooms_max', 'prefers_parking', 'prefers_garden',
            'prefers_balcony', 'prefers_elevator', 'prefers_furnished',
            'priority', 'status', 'notes',
            'budget_range_display', 'area_range_display', 'rooms_range_display',
            'preferences_list', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class ContactCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating contacts."""
    
    class Meta:
        model = Contact
        fields = [
            'external_id', 'name', 'email', 'phone',
            'preferred_locations', 'budget_min', 'budget_max',
            'desired_area_min', 'desired_area_max', 'property_type',
            'rooms_min', 'rooms_max', 'prefers_parking', 'prefers_garden',
            'prefers_balcony', 'prefers_elevator', 'prefers_furnished',
            'priority', 'status', 'notes'
        ]
    
    def validate_external_id(self, value):
        """Validate that external_id is unique."""
        if Contact.objects.filter(external_id=value).exists():
            raise serializers.ValidationError("Contact with this external ID already exists.")
        return value
    
    def validate(self, data):
        """Validate budget and area ranges."""
        if data['budget_max'] < data['budget_min']:
            raise serializers.ValidationError("Maximum budget must be greater than minimum budget.")
        
        if data['desired_area_max'] < data['desired_area_min']:
            raise serializers.ValidationError("Maximum area must be greater than minimum area.")
        
        if data.get('rooms_min') and data.get('rooms_max'):
            if data['rooms_max'] < data['rooms_min']:
                raise serializers.ValidationError("Maximum rooms must be greater than minimum rooms.")
        
        return data


class ContactUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating contacts."""
    
    class Meta:
        model = Contact
        fields = [
            'name', 'email', 'phone', 'preferred_locations',
            'budget_min', 'budget_max', 'desired_area_min', 'desired_area_max',
            'property_type', 'rooms_min', 'rooms_max', 'prefers_parking',
            'prefers_garden', 'prefers_balcony', 'prefers_elevator',
            'prefers_furnished', 'priority', 'status', 'notes'
        ]
    
    def validate(self, data):
        """Validate budget and area ranges."""
        budget_min = data.get('budget_min', self.instance.budget_min)
        budget_max = data.get('budget_max', self.instance.budget_max)
        
        if budget_max < budget_min:
            raise serializers.ValidationError("Maximum budget must be greater than minimum budget.")
        
        area_min = data.get('desired_area_min', self.instance.desired_area_min)
        area_max = data.get('desired_area_max', self.instance.desired_area_max)
        
        if area_max < area_min:
            raise serializers.ValidationError("Maximum area must be greater than minimum area.")
        
        rooms_min = data.get('rooms_min', self.instance.rooms_min)
        rooms_max = data.get('rooms_max', self.instance.rooms_max)
        
        if rooms_min and rooms_max and rooms_max < rooms_min:
            raise serializers.ValidationError("Maximum rooms must be greater than minimum rooms.")
        
        return data