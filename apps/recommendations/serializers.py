"""
Serializers for the recommendations API.
Handles request/response validation and serialization.
"""
from rest_framework import serializers
from typing import Dict, Any, Optional


class BatchUpdateSerializer(serializers.Serializer):
    """Serializer for batch update requests."""
    property_ids = serializers.ListField(
        child=serializers.CharField(max_length=100),
        min_length=1,
        max_length=1000,
        help_text="List of property IDs to update"
    )


class BuyerSerializer(serializers.Serializer):
    """Serializer for buyer data in recommendations."""
    id = serializers.CharField(help_text="Buyer's unique identifier")
    name = serializers.CharField(help_text="Buyer's full name")
    email = serializers.EmailField(help_text="Buyer's email address")
    phone = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Buyer's phone number"
    )
    budget_min = serializers.FloatField(
        required=False,
        allow_null=True,
        help_text="Minimum budget"
    )
    budget_max = serializers.FloatField(
        required=False,
        allow_null=True,
        help_text="Maximum budget"
    )
    preferred_locations = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=list,
        help_text="List of preferred locations"
    )


class RecommendationItemSerializer(serializers.Serializer):
    """Serializer for individual recommendation items."""
    buyer_id = serializers.CharField(help_text="Buyer's unique identifier")
    score = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        help_text="Recommendation score (0-1)"
    )
    buyer = BuyerSerializer(help_text="Buyer details")
    explanation = serializers.CharField(
        allow_blank=True,
        help_text="Human-readable explanation for the recommendation"
    )


class PropertyRecommendationResponseSerializer(serializers.Serializer):
    """Serializer for property recommendation responses."""
    property_id = serializers.CharField(help_text="Property ID")
    recommendations = RecommendationItemSerializer(
        many=True,
        help_text="List of recommended buyers"
    )
    total_count = serializers.IntegerField(
        min_value=0,
        help_text="Total number of recommendations"
    )
    processing_time_ms = serializers.FloatField(
        min_value=0,
        help_text="Time taken to process the request in milliseconds"
    )
    cached = serializers.BooleanField(
        default=False,
        help_text="Whether the result was served from cache"
    )
    error = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Error message if the request failed"
    )

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the recommendation response."""
        if 'error' in data and data['error'] and data.get('total_count', 0) > 0:
            raise serializers.ValidationError(
                "Response cannot have both error and recommendations"
            )
        return data


class BatchUpdateResponseSerializer(serializers.Serializer):
    """Serializer for batch update responses."""
    status = serializers.CharField(help_text="Status of the batch update")
    data = serializers.DictField(
        child=serializers.IntegerField(),
        help_text="Update statistics"
    )


class PropertyRecommendationRequestSerializer(serializers.Serializer):
    """Serializer for property recommendation requests."""
    property_id = serializers.CharField(help_text="Property ID to get recommendations for")
    max_results = serializers.IntegerField(
        default=10,
        min_value=1,
        max_value=100,
        help_text="Maximum number of recommendations to return"
    )
    min_score = serializers.FloatField(
        default=0.3,
        min_value=0.0,
        max_value=1.0,
        help_text="Minimum recommendation score threshold"
    )


class BatchRecommendationResponseSerializer(serializers.Serializer):
    """Serializer for batch recommendation responses."""
    results = PropertyRecommendationResponseSerializer(
        many=True,
        help_text="List of property recommendations"
    )
    total_properties = serializers.IntegerField(
        min_value=0,
        help_text="Total number of properties processed"
    )
    processing_time_ms = serializers.FloatField(
        min_value=0,
        help_text="Total processing time in milliseconds"
    )


class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check responses."""
    status = serializers.CharField(help_text="Service status")
    service = serializers.CharField(help_text="Service name")
    version = serializers.CharField(help_text="Service version")
    using_gpu = serializers.BooleanField(help_text="Whether GPU is being used")