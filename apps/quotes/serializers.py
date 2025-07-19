"""
Serializers for the quotes app.
"""
from rest_framework import serializers
from .models import Quote


class QuoteSerializer(serializers.ModelSerializer):
    """Serializer for quotes."""
    property_title = serializers.CharField(source='property.title', read_only=True)
    contact_name = serializers.CharField(source='contact.name', read_only=True)
    comparison_property_title = serializers.CharField(source='comparison_property.title', read_only=True)
    
    class Meta:
        model = Quote
        fields = [
            'id', 'quote_number', 'quote_type', 'status',
            'property', 'contact', 'comparison_property',
            'property_title', 'contact_name', 'comparison_property_title',
            'base_price', 'additional_fees', 'total_amount',
            'pdf_file', 'notes', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'quote_number', 'created_at', 'updated_at']


class PropertyQuoteRequestSerializer(serializers.Serializer):
    """Serializer for property quote generation requests."""
    property_id = serializers.CharField()
    contact_id = serializers.CharField()
    additional_fees = serializers.DictField(
        child=serializers.DecimalField(max_digits=14, decimal_places=2),
        required=False,
        default=dict
    )
    notes = serializers.CharField(required=False, default='')


class ComparisonQuoteRequestSerializer(serializers.Serializer):
    """Serializer for property comparison quote requests."""
    property_id_1 = serializers.CharField()
    property_id_2 = serializers.CharField()
    contact_id = serializers.CharField()
    notes = serializers.CharField(required=False, default='')


class QuoteResponseSerializer(serializers.Serializer):
    """Serializer for quote generation responses."""
    quote = QuoteSerializer()
    pdf_url = serializers.URLField()
    message = serializers.CharField()