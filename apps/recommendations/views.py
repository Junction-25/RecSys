"""
Views for the TikTok-like recommendations app.
Clean, optimized API endpoints with intelligent caching and performance monitoring.
"""
import logging
import time
from django.core.cache import cache
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse

from .service_registry import get_recommendation_engine
from .serializers import (
    PropertyRecommendationRequestSerializer,
    PropertyRecommendationResponseSerializer,
    BatchRecommendationResponseSerializer,
)

logger = logging.getLogger(__name__)

# Cache timeout in seconds (5 minutes)
CACHE_TIMEOUT = 300


class RecommendationViewSet(viewsets.ViewSet):
    """
    TikTok-like API endpoints for property and contact recommendations.
    Features:
    - Singleton pattern for model loading
    - Intelligent caching
    - Performance monitoring
    - Efficient database queries
    """
    # Authentication disabled for development
    permission_classes = []
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None
    
    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_recommendation_engine()
        return self._engine
    
    @extend_schema(
        parameters=[
            OpenApiParameter(name='id', description='Property ID', required=True, type=str),
            OpenApiParameter(name='min_score', description='Minimum match score (default 0.3)', required=False, type=float),
            OpenApiParameter(name='max_results', description='Maximum number of results (default 10)', required=False, type=int),
            OpenApiParameter(name='use_cache', description='Use cached results if available (default true)', required=False, type=bool),
        ],
        responses={
            200: PropertyRecommendationResponseSerializer,
            400: OpenApiResponse(description='Invalid request parameters'),
            404: OpenApiResponse(description='Property not found'),
            500: OpenApiResponse(description='Internal server error')
        },
        tags=['recommendations']
    )
    @action(detail=False, methods=['get'], url_path='property/(?P<id>[^/.]+)')
    def property_recommendations(self, request, id=None):
        start_time = time.time()
        
        # Validate request parameters (property_id comes from URL, not query params)
        query_params = request.query_params.copy()
        query_params['property_id'] = id  # Add property_id from URL to validation
        
        serializer = PropertyRecommendationRequestSerializer(data=query_params)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid parameters', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        min_score = float(serializer.validated_data.get('min_score', 0.3))
        max_results = int(serializer.validated_data.get('max_results', 10))
        use_cache = serializer.validated_data.get('use_cache', True)
        
        # Generate cache key
        cache_key = f'property_recs:{id}:{min_score}:{max_results}'
        
        try:
            # Try to get from cache
            if use_cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for property recommendations: {cache_key}")
                    return Response(cached_result)
            
            # Get recommendations from engine
            result = self.engine.get_recommendations_for_property(
                property_id=id,
                min_score=min_score,
                max_results=max_results
            )
            
            if 'error' in result:
                return Response(
                    {'error': result['error']}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Format response
            response_data = {
                'property_id': str(id),
                'recommendations': result.get('recommendations', []),
                'total_count': len(result.get('recommendations', [])),
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'cached': False,
                'method': result.get('method', 'tiktok_pipeline')
            }
            
            # Cache the result
            if use_cache:
                cache.set(cache_key, response_data, timeout=CACHE_TIMEOUT)
                logger.debug(f"Cached property recommendations: {cache_key}")
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error getting property recommendations: {str(e)}", exc_info=True)
            return Response(
                {'error': 'Internal server error', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        parameters=[
            OpenApiParameter(name='property_ids', description='Comma-separated list of property IDs (optional)', required=False, type=str),
            OpenApiParameter(name='min_score', description='Minimum match score (default 0.3)', required=False, type=float),
            OpenApiParameter(name='max_results', description='Maximum number of results per property (default 10)', required=False, type=int),
            OpenApiParameter(name='use_cache', description='Use cached results if available (default true)', required=False, type=bool),
        ],
        responses={
            200: BatchRecommendationResponseSerializer,
            400: OpenApiResponse(description='Invalid request parameters'),
            500: OpenApiResponse(description='Internal server error')
        },
        tags=['recommendations']
    )
    @action(detail=False, methods=['get'], url_path='bulk')
    def bulk_recommendations(self, request):
        """Get recommendations for multiple properties at once."""
        start_time = time.time()
        
        # Parse query parameters
        property_ids_param = request.query_params.get('property_ids', '')
        min_score = float(request.query_params.get('min_score', 0.3))
        max_results = int(request.query_params.get('max_results', 10))
        use_cache = request.query_params.get('use_cache', 'true').lower() == 'true'
        
        try:
            # Get property IDs - either from parameter or all available properties
            if property_ids_param:
                property_ids = [pid.strip() for pid in property_ids_param.split(',') if pid.strip()]
            else:
                # Get all property IDs from the engine or database
                # For now, we'll use a sample set - you can modify this to get from your Property model
                from apps.properties.models import Property
                property_ids = list(Property.objects.values_list('id', flat=True)[:50])  # Limit to 50 for performance
            
            if not property_ids:
                return Response(
                    {'error': 'No property IDs provided or found'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            results = []
            processed_count = 0
            
            for property_id in property_ids:
                try:
                    # Generate cache key
                    cache_key = f'property_recs:{property_id}:{min_score}:{max_results}'
                    
                    # Try to get from cache first
                    cached_result = None
                    if use_cache:
                        cached_result = cache.get(cache_key)
                    
                    if cached_result:
                        result_data = cached_result
                        result_data['cached'] = True
                    else:
                        # Get recommendations from engine
                        result = self.engine.get_recommendations_for_property(
                            property_id=str(property_id),
                            min_score=min_score,
                            max_results=max_results
                        )
                        
                        if 'error' not in result:
                            result_data = {
                                'property_id': str(property_id),
                                'recommendations': result.get('recommendations', []),
                                'total_count': len(result.get('recommendations', [])),
                                'cached': False,
                                'method': result.get('method', 'tiktok_pipeline')
                            }
                            
                            # Cache the result
                            if use_cache:
                                cache.set(cache_key, result_data, timeout=CACHE_TIMEOUT)
                        else:
                            result_data = {
                                'property_id': str(property_id),
                                'recommendations': [],
                                'total_count': 0,
                                'cached': False,
                                'error': result['error']
                            }
                    
                    results.append(result_data)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing property {property_id}: {str(e)}")
                    results.append({
                        'property_id': str(property_id),
                        'recommendations': [],
                        'total_count': 0,
                        'cached': False,
                        'error': f'Processing error: {str(e)}'
                    })
            
            # Format response
            response_data = {
                'results': results,
                'total_properties': processed_count,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'parameters': {
                    'min_score': min_score,
                    'max_results': max_results,
                    'use_cache': use_cache
                }
            }
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error in bulk recommendations: {str(e)}", exc_info=True)
            return Response(
                {'error': 'Internal server error', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        responses={200: dict},
        tags=['recommendations']
    )
    @action(detail=False, methods=['get'])
    def health(self, request):
        """Get recommendation engine health and performance metrics."""
        try:
            stats = self.engine.get_performance_stats()
            return Response({
                'status': 'healthy',
                'service': 'TikTok-like Recommendation Engine',
                'version': '1.0.0',
                'performance_stats': stats,
                'using_gpu': False,  # Update based on actual GPU usage
                'components': {
                    'property_encoder': 'active',
                    'vector_database': 'active', 
                    'deepfm_ranker': 'active',
                    'colbert_encoder': 'active'
                }
            })
        except Exception as e:
            logger.error(f"Error getting health metrics: {e}")
            return Response(
                {'status': 'unhealthy', 'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )