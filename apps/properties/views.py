"""
Views for the properties app.
"""
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, OpenApiParameter
from django.db.models import Q

from .models import Property
from .serializers import (
    PropertySerializer,
    PropertyCreateSerializer,
    PropertyUpdateSerializer,
    PropertyComparisonSerializer,
    PropertyComparisonResponseSerializer
)
from .filters import PropertyFilter


class PropertyViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing properties.
    """
    queryset = Property.objects.all()
    # Temporarily disable authentication for development
    permission_classes = []
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = PropertyFilter
    search_fields = ['title', 'description', 'city', 'district', 'address']
    ordering_fields = ['price', 'area', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'create':
            return PropertyCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return PropertyUpdateSerializer
        return PropertySerializer
    
    def get_queryset(self):
        """Return filtered queryset based on user permissions."""
        queryset = Property.objects.all()
        
        # Filter by status if provided
        status_param = self.request.query_params.get('status')
        if status_param:
            queryset = queryset.filter(status=status_param)
        
        return queryset
    
    @extend_schema(
        request=PropertyComparisonSerializer,
        responses={200: PropertyComparisonResponseSerializer},
        tags=['properties']
    )
    @action(detail=False, methods=['post'])
    def compare(self, request):
        """Compare two properties."""
        serializer = PropertyComparisonSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        property_id_1 = serializer.validated_data['property_id_1']
        property_id_2 = serializer.validated_data['property_id_2']
        
        try:
            # Try to get properties by UUID first, then by external_id
            try:
                property_1 = Property.objects.get(id=property_id_1)
            except (ValueError, Property.DoesNotExist):
                property_1 = Property.objects.get(external_id=property_id_1)
            
            try:
                property_2 = Property.objects.get(id=property_id_2)
            except (ValueError, Property.DoesNotExist):
                property_2 = Property.objects.get(external_id=property_id_2)
            
        except Property.DoesNotExist:
            return Response(
                {'error': 'One or both properties not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Calculate differences
        price_diff = float(property_2.price) - float(property_1.price)
        area_diff = float(property_2.area) - float(property_1.area)
        price_per_sqm_1 = float(property_1.price) / float(property_1.area)
        price_per_sqm_2 = float(property_2.price) / float(property_2.area)
        
        differences = {
            'price_difference': price_diff,
            'area_difference': area_diff,
            'price_per_sqm': {
                'property_1': round(price_per_sqm_1, 2),
                'property_2': round(price_per_sqm_2, 2)
            },
            'value_comparison': {
                'better_value': 'property_1' if price_per_sqm_1 < price_per_sqm_2 else 'property_2',
                'value_difference_percent': round(abs(price_per_sqm_2 - price_per_sqm_1) / min(price_per_sqm_1, price_per_sqm_2) * 100, 2)
            }
        }
        
        recommendation = (
            f"Property 1 offers better value per square meter"
            if price_per_sqm_1 < price_per_sqm_2
            else f"Property 2 offers better value per square meter"
        )
        
        response_data = {
            'property_1': PropertySerializer(property_1).data,
            'property_2': PropertySerializer(property_2).data,
            'differences': differences,
            'recommendation': recommendation
        }
        
        return Response(response_data)
    
    @extend_schema(
        parameters=[
            OpenApiParameter(name='city', description='Filter by city', required=False, type=str),
            OpenApiParameter(name='min_price', description='Minimum price', required=False, type=float),
            OpenApiParameter(name='max_price', description='Maximum price', required=False, type=float),
        ],
        tags=['properties']
    )
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Advanced property search."""
        queryset = self.get_queryset()
        
        # Apply filters
        city = request.query_params.get('city')
        if city:
            queryset = queryset.filter(city__icontains=city)
        
        min_price = request.query_params.get('min_price')
        if min_price:
            queryset = queryset.filter(price__gte=min_price)
        
        max_price = request.query_params.get('max_price')
        if max_price:
            queryset = queryset.filter(price__lte=max_price)
        
        min_area = request.query_params.get('min_area')
        if min_area:
            queryset = queryset.filter(area__gte=min_area)
        
        max_area = request.query_params.get('max_area')
        if max_area:
            queryset = queryset.filter(area__lte=max_area)
        
        property_type = request.query_params.get('property_type')
        if property_type:
            queryset = queryset.filter(property_type=property_type)
        
        # Paginate results
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @extend_schema(tags=['properties'])
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get property statistics."""
        queryset = self.get_queryset()
        
        stats = {
            'total_properties': queryset.count(),
            'by_status': {
                'available': queryset.filter(status=Property.AVAILABLE).count(),
                'pending': queryset.filter(status=Property.PENDING).count(),
                'sold': queryset.filter(status=Property.SOLD).count(),
                'rented': queryset.filter(status=Property.RENTED).count(),
            },
            'by_type': {
                'apartment': queryset.filter(property_type=Property.APARTMENT).count(),
                'villa': queryset.filter(property_type=Property.VILLA).count(),
                'office': queryset.filter(property_type=Property.OFFICE).count(),
                'commercial': queryset.filter(property_type=Property.COMMERCIAL).count(),
                'land': queryset.filter(property_type=Property.LAND).count(),
            },
            'by_listing_type': {
                'sale': queryset.filter(listing_type=Property.SALE).count(),
                'rent': queryset.filter(listing_type=Property.RENT).count(),
            }
        }
        
        # Calculate average prices
        available_properties = queryset.filter(status=Property.AVAILABLE)
        if available_properties.exists():
            avg_price = sum(float(p.price) for p in available_properties) / available_properties.count()
            avg_area = sum(float(p.area) for p in available_properties) / available_properties.count()
            stats['averages'] = {
                'price': round(avg_price, 2),
                'area': round(avg_area, 2),
                'price_per_sqm': round(avg_price / avg_area, 2) if avg_area > 0 else 0
            }
        
        return Response(stats)