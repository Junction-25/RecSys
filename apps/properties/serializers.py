"""
Serializers for the properties app.
"""
from rest_framework import serializers
from .models import Property, PropertyImage


class PropertyImageSerializer(serializers.ModelSerializer):
    """Serializer for property images."""
    
    class Meta:
        model = PropertyImage
        fields = ['id', 'image', 'caption', 'order']


class PropertySerializer(serializers.ModelSerializer):
    """Serializer for properties."""
    images = PropertyImageSerializer(many=True, read_only=True)
    location_display = serializers.ReadOnlyField()
    features_list = serializers.ReadOnlyField()
    
    class Meta:
        model = Property
        fields = [
            'id', 'external_id', 'title', 'description',
            'address', 'city', 'district', 'latitude', 'longitude',
            'price', 'area', 'property_type', 'rooms', 'bathrooms',
            'has_parking', 'has_garden', 'has_balcony', 'has_elevator',
            'is_furnished', 'has_air_conditioning', 'has_heating',
            'status', 'listing_type', 'agent_id', 'agent_name', 'agent_contact',
            'location_display', 'features_list', 'images',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class PropertyCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating properties."""
    
    class Meta:
        model = Property
        fields = [
            'external_id', 'title', 'description',
            'address', 'city', 'district', 'latitude', 'longitude',
            'price', 'area', 'property_type', 'rooms', 'bathrooms',
            'has_parking', 'has_garden', 'has_balcony', 'has_elevator',
            'is_furnished', 'has_air_conditioning', 'has_heating',
            'status', 'listing_type', 'agent_id', 'agent_name', 'agent_contact'
        ]
    
    def validate_external_id(self, value):
        """Validate that external_id is unique."""
        if Property.objects.filter(external_id=value).exists():
            raise serializers.ValidationError("Property with this external ID already exists.")
        return value


class PropertyUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating properties."""
    
    class Meta:
        model = Property
        fields = [
            'title', 'description', 'address', 'city', 'district',
            'latitude', 'longitude', 'price', 'area', 'rooms', 'bathrooms',
            'has_parking', 'has_garden', 'has_balcony', 'has_elevator',
            'is_furnished', 'has_air_conditioning', 'has_heating',
            'status', 'listing_type', 'agent_id', 'agent_name', 'agent_contact'
        ]


class PropertyComparisonSerializer(serializers.Serializer):
    """Serializer for property comparison requests."""
    property_id_1 = serializers.CharField()
    property_id_2 = serializers.CharField()


class PropertyComparisonResponseSerializer(serializers.Serializer):
    """Serializer for property comparison responses."""
    property_1 = PropertySerializer()
    property_2 = PropertySerializer()
    differences = serializers.DictField()
    recommendation = serializers.CharField()