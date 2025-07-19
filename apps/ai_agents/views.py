"""
Views for the AI agents app.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.utils import timezone

from .services import GeminiAgent
from .serializers import (
    PropertyAnalysisRequestSerializer,
    PropertyAnalysisResponseSerializer,
    ContactAnalysisRequestSerializer,
    ContactAnalysisResponseSerializer,
    MatchExplanationRequestSerializer,
    PropertyDescriptionRequestSerializer
)
from apps.properties.models import Property
from apps.properties.serializers import PropertySerializer
from apps.contacts.models import Contact
from apps.contacts.serializers import ContactSerializer


class AIAgentViewSet(viewsets.ViewSet):
    """
    AI Agent endpoints for intelligent property and contact analysis.
    """
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = GeminiAgent()
    
    @extend_schema(
        responses={200: PropertyAnalysisResponseSerializer},
        tags=['ai-agents']
    )
    @method_decorator(cache_page(60 * 60))  # Cache for 1 hour
    @action(detail=False, methods=['get'], url_path='analyze-property/(?P<property_id>[^/.]+)')
    def analyze_property(self, request, property_id=None):
        """Analyze a property using AI and provide market insights."""
        try:
            # Try to get property by UUID first, then by external_id
            try:
                property_obj = Property.objects.get(id=property_id)
            except (ValueError, Property.DoesNotExist):
                property_obj = Property.objects.get(external_id=property_id)
            
            # Prepare property data for AI analysis
            property_data = PropertySerializer(property_obj).data
            
            # Get AI analysis
            analysis = self.agent.analyze_property(property_data)
            
            return Response({
                'property': property_data,
                'analysis': analysis,
                'generated_at': timezone.now().isoformat()
            })
            
        except Property.DoesNotExist:
            return Response(
                {'error': 'Property not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        responses={200: ContactAnalysisResponseSerializer},
        tags=['ai-agents']
    )
    @method_decorator(cache_page(60 * 60))  # Cache for 1 hour
    @action(detail=False, methods=['get'], url_path='analyze-contact/(?P<contact_id>[^/.]+)')
    def analyze_contact(self, request, contact_id=None):
        """Analyze a contact's preferences using AI."""
        try:
            # Try to get contact by UUID first, then by external_id
            try:
                contact = Contact.objects.get(id=contact_id)
            except (ValueError, Contact.DoesNotExist):
                contact = Contact.objects.get(external_id=contact_id)
            
            # Prepare contact data for AI analysis
            contact_data = ContactSerializer(contact).data
            
            # Get AI analysis
            analysis = self.agent.analyze_contact_preferences(contact_data)
            
            return Response({
                'contact': contact_data,
                'analysis': analysis,
                'generated_at': timezone.now().isoformat()
            })
            
        except Contact.DoesNotExist:
            return Response(
                {'error': 'Contact not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        request=MatchExplanationRequestSerializer,
        tags=['ai-agents']
    )
    @action(detail=False, methods=['post'])
    def explain_match(self, request):
        """Generate an AI explanation for why a property matches a contact."""
        serializer = MatchExplanationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        property_id = serializer.validated_data['property_id']
        contact_id = serializer.validated_data['contact_id']
        match_score = serializer.validated_data.get('match_score', 0.0)
        
        try:
            # Get property and contact
            try:
                property_obj = Property.objects.get(id=property_id)
            except (ValueError, Property.DoesNotExist):
                property_obj = Property.objects.get(external_id=property_id)
            
            try:
                contact = Contact.objects.get(id=contact_id)
            except (ValueError, Contact.DoesNotExist):
                contact = Contact.objects.get(external_id=contact_id)
            
            # Prepare data for AI
            property_data = PropertySerializer(property_obj).data
            contact_data = ContactSerializer(contact).data
            
            # Generate explanation
            explanation = self.agent.generate_match_explanation(
                property_data, contact_data, match_score
            )
            
            return Response({
                'property': property_data,
                'contact': contact_data,
                'match_score': match_score,
                'explanation': explanation,
                'generated_at': timezone.now().isoformat()
            })
            
        except (Property.DoesNotExist, Contact.DoesNotExist):
            return Response(
                {'error': 'Property or contact not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        responses={200: dict},
        tags=['ai-agents']
    )
    @method_decorator(cache_page(60 * 60 * 24))  # Cache for 24 hours
    @action(detail=False, methods=['get'], url_path='generate-description/(?P<property_id>[^/.]+)')
    def generate_description(self, request, property_id=None):
        """Generate an engaging property description using AI."""
        try:
            # Try to get property by UUID first, then by external_id
            try:
                property_obj = Property.objects.get(id=property_id)
            except (ValueError, Property.DoesNotExist):
                property_obj = Property.objects.get(external_id=property_id)
            
            # Prepare property data for AI
            property_data = PropertySerializer(property_obj).data
            
            # Generate description
            description = self.agent.generate_property_description(property_data)
            
            return Response({
                'property_id': str(property_obj.id),
                'property_external_id': property_obj.external_id,
                'original_description': property_obj.description,
                'ai_generated_description': description,
                'generated_at': timezone.now().isoformat()
            })
            
        except Property.DoesNotExist:
            return Response(
                {'error': 'Property not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        tags=['ai-agents']
    )
    @action(detail=False, methods=['get'])
    def capabilities(self, request):
        """Get information about AI agent capabilities."""
        return Response({
            'capabilities': [
                {
                    'name': 'Property Analysis',
                    'description': 'Analyze properties for market position, valuation, and selling points',
                    'endpoint': '/api/v1/ai-agents/analyze-property/{property_id}/'
                },
                {
                    'name': 'Contact Analysis',
                    'description': 'Analyze contact preferences and buyer motivation',
                    'endpoint': '/api/v1/ai-agents/analyze-contact/{contact_id}/'
                },
                {
                    'name': 'Match Explanation',
                    'description': 'Generate personalized explanations for property-contact matches',
                    'endpoint': '/api/v1/ai-agents/explain-match/'
                },
                {
                    'name': 'Description Generation',
                    'description': 'Generate engaging property descriptions',
                    'endpoint': '/api/v1/ai-agents/generate-description/{property_id}/'
                }
            ],
            'ai_model': 'Google Gemini Pro',
            'features': [
                'Natural language processing',
                'Market analysis',
                'Personalized recommendations',
                'Content generation',
                'Intelligent matching explanations'
            ]
        })