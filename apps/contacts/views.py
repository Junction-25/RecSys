"""
Views for the contacts app.
"""
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, OpenApiParameter

from .models import Contact
from .serializers import (
    ContactSerializer,
    ContactCreateSerializer,
    ContactUpdateSerializer
)
from .filters import ContactFilter


class ContactViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing contacts.
    """
    queryset = Contact.objects.all()
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_class = ContactFilter
    search_fields = ['name', 'email', 'phone', 'preferred_locations', 'notes']
    ordering_fields = ['name', 'budget_min', 'budget_max', 'priority', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'create':
            return ContactCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return ContactUpdateSerializer
        return ContactSerializer
    
    def get_queryset(self):
        """Return filtered queryset based on user permissions."""
        queryset = Contact.objects.all()
        
        # Filter by status if provided
        status_param = self.request.query_params.get('status')
        if status_param:
            queryset = queryset.filter(status=status_param)
        
        return queryset
    
    @extend_schema(
        parameters=[
            OpenApiParameter(name='property_type', description='Filter by property type', required=False, type=str),
            OpenApiParameter(name='min_budget', description='Minimum budget', required=False, type=float),
            OpenApiParameter(name='max_budget', description='Maximum budget', required=False, type=float),
        ],
        tags=['contacts']
    )
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Advanced contact search."""
        queryset = self.get_queryset()
        
        # Apply filters
        property_type = request.query_params.get('property_type')
        if property_type:
            queryset = queryset.filter(property_type=property_type)
        
        min_budget = request.query_params.get('min_budget')
        if min_budget:
            queryset = queryset.filter(budget_min__gte=min_budget)
        
        max_budget = request.query_params.get('max_budget')
        if max_budget:
            queryset = queryset.filter(budget_max__lte=max_budget)
        
        priority = request.query_params.get('priority')
        if priority:
            queryset = queryset.filter(priority=priority)
        
        location = request.query_params.get('location')
        if location:
            queryset = queryset.filter(preferred_locations__icontains=location)
        
        # Paginate results
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @extend_schema(tags=['contacts'])
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get contact statistics."""
        queryset = self.get_queryset()
        
        stats = {
            'total_contacts': queryset.count(),
            'by_status': {
                'active': queryset.filter(status=Contact.ACTIVE).count(),
                'inactive': queryset.filter(status=Contact.INACTIVE).count(),
                'closed': queryset.filter(status=Contact.CLOSED).count(),
            },
            'by_priority': {
                'high': queryset.filter(priority=Contact.HIGH).count(),
                'medium': queryset.filter(priority=Contact.MEDIUM).count(),
                'low': queryset.filter(priority=Contact.LOW).count(),
            },
            'by_property_type': {
                'apartment': queryset.filter(property_type=Contact.APARTMENT).count(),
                'villa': queryset.filter(property_type=Contact.VILLA).count(),
                'office': queryset.filter(property_type=Contact.OFFICE).count(),
                'commercial': queryset.filter(property_type=Contact.COMMERCIAL).count(),
                'land': queryset.filter(property_type=Contact.LAND).count(),
            }
        }
        
        # Calculate average budgets
        active_contacts = queryset.filter(status=Contact.ACTIVE)
        if active_contacts.exists():
            avg_budget_min = sum(float(c.budget_min) for c in active_contacts) / active_contacts.count()
            avg_budget_max = sum(float(c.budget_max) for c in active_contacts) / active_contacts.count()
            avg_area_min = sum(float(c.desired_area_min) for c in active_contacts) / active_contacts.count()
            avg_area_max = sum(float(c.desired_area_max) for c in active_contacts) / active_contacts.count()
            
            stats['averages'] = {
                'budget_min': round(avg_budget_min, 2),
                'budget_max': round(avg_budget_max, 2),
                'budget_range': round(avg_budget_max - avg_budget_min, 2),
                'area_min': round(avg_area_min, 2),
                'area_max': round(avg_area_max, 2),
                'area_range': round(avg_area_max - avg_area_min, 2)
            }
        
        return Response(stats)
    
    @extend_schema(tags=['contacts'])
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a contact."""
        contact = self.get_object()
        contact.status = Contact.ACTIVE
        contact.save()
        
        serializer = self.get_serializer(contact)
        return Response(serializer.data)
    
    @extend_schema(tags=['contacts'])
    @action(detail=True, methods=['post'])
    def deactivate(self, request, pk=None):
        """Deactivate a contact."""
        contact = self.get_object()
        contact.status = Contact.INACTIVE
        contact.save()
        
        serializer = self.get_serializer(contact)
        return Response(serializer.data)