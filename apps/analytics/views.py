"""
Views for the analytics app.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.db.models import Count, Avg, Sum, Q
from django.utils import timezone
from datetime import timedelta

from apps.properties.models import Property
from apps.contacts.models import Contact


class AnalyticsViewSet(viewsets.ViewSet):
    """
    Analytics endpoints for business intelligence and reporting.
    """
    permission_classes = [IsAuthenticated]
    
    @extend_schema(tags=['analytics'])
    @method_decorator(cache_page(60 * 30))  # Cache for 30 minutes
    @action(detail=False, methods=['get'])
    def dashboard(self, request):
        """Get dashboard analytics data."""
        # Property statistics
        total_properties = Property.objects.count()
        available_properties = Property.objects.filter(status=Property.AVAILABLE).count()
        sold_properties = Property.objects.filter(status=Property.SOLD).count()
        rented_properties = Property.objects.filter(status=Property.RENTED).count()
        
        # Contact statistics
        total_contacts = Contact.objects.count()
        active_contacts = Contact.objects.filter(status=Contact.ACTIVE).count()
        high_priority_contacts = Contact.objects.filter(
            status=Contact.ACTIVE, 
            priority=Contact.HIGH
        ).count()
        
        # Recent activity (last 30 days)
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_properties = Property.objects.filter(created_at__gte=thirty_days_ago).count()
        recent_contacts = Contact.objects.filter(created_at__gte=thirty_days_ago).count()
        
        # Property type distribution
        property_types = Property.objects.values('property_type').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # Contact property type preferences
        contact_preferences = Contact.objects.filter(
            status=Contact.ACTIVE
        ).values('property_type').annotate(
            count=Count('id')
        ).order_by('-count')
        
        # Average prices by property type
        avg_prices = Property.objects.filter(
            status=Property.AVAILABLE
        ).values('property_type').annotate(
            avg_price=Avg('price'),
            count=Count('id')
        ).order_by('property_type')
        
        return Response({
            'overview': {
                'total_properties': total_properties,
                'available_properties': available_properties,
                'sold_properties': sold_properties,
                'rented_properties': rented_properties,
                'total_contacts': total_contacts,
                'active_contacts': active_contacts,
                'high_priority_contacts': high_priority_contacts,
            },
            'recent_activity': {
                'new_properties_30_days': recent_properties,
                'new_contacts_30_days': recent_contacts,
            },
            'distributions': {
                'property_types': list(property_types),
                'contact_preferences': list(contact_preferences),
                'average_prices': list(avg_prices),
            },
            'generated_at': timezone.now().isoformat()
        })
    
    @extend_schema(tags=['analytics'])
    @method_decorator(cache_page(60 * 60))  # Cache for 1 hour
    @action(detail=False, methods=['get'])
    def market_trends(self, request):
        """Get market trend analytics."""
        # Price trends by property type
        price_trends = {}
        for prop_type in [Property.APARTMENT, Property.VILLA, Property.OFFICE, Property.COMMERCIAL, Property.LAND]:
            properties = Property.objects.filter(
                property_type=prop_type,
                status=Property.AVAILABLE
            )
            
            if properties.exists():
                avg_price = properties.aggregate(avg_price=Avg('price'))['avg_price']
                avg_area = properties.aggregate(avg_area=Avg('area'))['avg_area']
                price_per_sqm = avg_price / avg_area if avg_area else 0
                
                price_trends[prop_type] = {
                    'average_price': round(float(avg_price), 2),
                    'average_area': round(float(avg_area), 2),
                    'price_per_sqm': round(float(price_per_sqm), 2),
                    'total_listings': properties.count()
                }
        
        # Demand analysis (active contacts by property type)
        demand_analysis = Contact.objects.filter(
            status=Contact.ACTIVE
        ).values('property_type').annotate(
            demand_count=Count('id'),
            avg_budget_min=Avg('budget_min'),
            avg_budget_max=Avg('budget_max')
        ).order_by('-demand_count')
        
        # Supply vs Demand
        supply_demand = []
        for prop_type in [Property.APARTMENT, Property.VILLA, Property.OFFICE, Property.COMMERCIAL, Property.LAND]:
            supply = Property.objects.filter(
                property_type=prop_type,
                status=Property.AVAILABLE
            ).count()
            
            demand = Contact.objects.filter(
                property_type=prop_type,
                status=Contact.ACTIVE
            ).count()
            
            ratio = demand / supply if supply > 0 else 0
            
            supply_demand.append({
                'property_type': prop_type,
                'supply': supply,
                'demand': demand,
                'demand_supply_ratio': round(ratio, 2)
            })
        
        return Response({
            'price_trends': price_trends,
            'demand_analysis': list(demand_analysis),
            'supply_demand': supply_demand,
            'generated_at': timezone.now().isoformat()
        })
    
    @extend_schema(tags=['analytics'])
    @method_decorator(cache_page(60 * 60))  # Cache for 1 hour
    @action(detail=False, methods=['get'])
    def location_insights(self, request):
        """Get location-based analytics."""
        # Properties by city
        properties_by_city = Property.objects.values('city').annotate(
            count=Count('id'),
            avg_price=Avg('price'),
            available_count=Count('id', filter=Q(status=Property.AVAILABLE))
        ).order_by('-count')
        
        # Contact preferences by location
        location_preferences = {}
        contacts = Contact.objects.filter(status=Contact.ACTIVE)
        
        for contact in contacts:
            for location in contact.preferred_locations:
                if location not in location_preferences:
                    location_preferences[location] = 0
                location_preferences[location] += 1
        
        # Sort by popularity
        sorted_preferences = sorted(
            location_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 preferred locations
        
        # Price comparison by district
        district_prices = Property.objects.filter(
            status=Property.AVAILABLE
        ).values('city', 'district').annotate(
            avg_price=Avg('price'),
            count=Count('id')
        ).order_by('city', '-avg_price')
        
        return Response({
            'properties_by_city': list(properties_by_city),
            'top_preferred_locations': [
                {'location': loc, 'preference_count': count}
                for loc, count in sorted_preferences
            ],
            'district_price_comparison': list(district_prices),
            'generated_at': timezone.now().isoformat()
        })
    
    @extend_schema(tags=['analytics'])
    @action(detail=False, methods=['get'])
    def performance_metrics(self, request):
        """Get system performance metrics."""
        # This would typically integrate with monitoring systems
        # For now, we'll return mock data
        return Response({
            'api_performance': {
                'average_response_time_ms': 145,
                'requests_per_minute': 25,
                'error_rate_percent': 0.2,
                'uptime_percent': 99.9
            },
            'recommendation_engine': {
                'total_recommendations_generated': 15420,
                'average_match_score': 0.72,
                'cache_hit_ratio': 0.85,
                'processing_time_ms': 89
            },
            'ai_agents': {
                'total_analyses_performed': 3240,
                'average_processing_time_ms': 1250,
                'success_rate_percent': 98.5,
                'cache_utilization': 0.78
            },
            'database': {
                'total_queries_per_minute': 180,
                'average_query_time_ms': 12,
                'connection_pool_usage': 0.65,
                'slow_queries_count': 2
            },
            'generated_at': timezone.now().isoformat()
        })