"""
Serializers for the AI agents app.
"""
from rest_framework import serializers


class PropertyAnalysisRequestSerializer(serializers.Serializer):
    """Serializer for property analysis requests."""
    property_id = serializers.CharField()


class PropertyAnalysisResponseSerializer(serializers.Serializer):
    """Serializer for property analysis responses."""
    property = serializers.DictField()
    analysis = serializers.DictField()
    generated_at = serializers.DateTimeField()


class ContactAnalysisRequestSerializer(serializers.Serializer):
    """Serializer for contact analysis requests."""
    contact_id = serializers.CharField()


class ContactAnalysisResponseSerializer(serializers.Serializer):
    """Serializer for contact analysis responses."""
    contact = serializers.DictField()
    analysis = serializers.DictField()
    generated_at = serializers.DateTimeField()


class MatchExplanationRequestSerializer(serializers.Serializer):
    """Serializer for match explanation requests."""
    property_id = serializers.CharField()
    contact_id = serializers.CharField()
    match_score = serializers.FloatField(required=False, default=0.0)


class MatchExplanationResponseSerializer(serializers.Serializer):
    """Serializer for match explanation responses."""
    property = serializers.DictField()
    contact = serializers.DictField()
    match_score = serializers.FloatField()
    explanation = serializers.CharField()
    generated_at = serializers.DateTimeField()


class PropertyDescriptionRequestSerializer(serializers.Serializer):
    """Serializer for property description generation requests."""
    property_id = serializers.CharField()


class PropertyDescriptionResponseSerializer(serializers.Serializer):
    """Serializer for property description generation responses."""
    property_id = serializers.CharField()
    property_external_id = serializers.CharField()
    original_description = serializers.CharField()
    ai_generated_description = serializers.CharField()
    generated_at = serializers.DateTimeField()