"""
Views for the quotes app.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema
from django.http import HttpResponse, Http404
from django.shortcuts import get_object_or_404

from .models import Quote
from .services import QuoteGenerationService
from .serializers import (
    QuoteSerializer,
    PropertyQuoteRequestSerializer,
    ComparisonQuoteRequestSerializer,
    QuoteResponseSerializer
)


class QuoteViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing quotes and PDF generation.
    """
    queryset = Quote.objects.all()
    serializer_class = QuoteSerializer
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quote_service = QuoteGenerationService()
    
    @extend_schema(
        request=PropertyQuoteRequestSerializer,
        responses={201: QuoteResponseSerializer},
        tags=['quotes']
    )
    @action(detail=False, methods=['post'])
    def generate_property_quote(self, request):
        """Generate a property quote PDF."""
        serializer = PropertyQuoteRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            quote = self.quote_service.generate_property_quote(
                property_id=serializer.validated_data['property_id'],
                contact_id=serializer.validated_data['contact_id'],
                additional_fees=serializer.validated_data.get('additional_fees', {})
            )
            
            # Update notes if provided
            if serializer.validated_data.get('notes'):
                quote.notes = serializer.validated_data['notes']
                quote.save()
            
            pdf_url = request.build_absolute_uri(quote.pdf_file.url) if quote.pdf_file else None
            
            return Response({
                'quote': QuoteSerializer(quote).data,
                'pdf_url': pdf_url,
                'message': 'Property quote generated successfully'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @extend_schema(
        request=ComparisonQuoteRequestSerializer,
        responses={201: QuoteResponseSerializer},
        tags=['quotes']
    )
    @action(detail=False, methods=['post'])
    def generate_comparison_quote(self, request):
        """Generate a property comparison quote PDF."""
        serializer = ComparisonQuoteRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            quote = self.quote_service.generate_comparison_quote(
                property_id_1=serializer.validated_data['property_id_1'],
                property_id_2=serializer.validated_data['property_id_2'],
                contact_id=serializer.validated_data['contact_id']
            )
            
            # Update notes if provided
            if serializer.validated_data.get('notes'):
                quote.notes = serializer.validated_data['notes']
                quote.save()
            
            pdf_url = request.build_absolute_uri(quote.pdf_file.url) if quote.pdf_file else None
            
            return Response({
                'quote': QuoteSerializer(quote).data,
                'pdf_url': pdf_url,
                'message': 'Property comparison quote generated successfully'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @extend_schema(
        responses={200: 'PDF file'},
        tags=['quotes']
    )
    @action(detail=True, methods=['get'])
    def download_pdf(self, request, pk=None):
        """Download the PDF file for a quote."""
        quote = self.get_object()
        
        if not quote.pdf_file:
            raise Http404("PDF file not found")
        
        try:
            response = HttpResponse(
                quote.pdf_file.read(),
                content_type='application/pdf'
            )
            response['Content-Disposition'] = f'attachment; filename="quote_{quote.quote_number}.pdf"'
            return response
        except Exception:
            raise Http404("PDF file not accessible")
    
    @extend_schema(tags=['quotes'])
    @action(detail=True, methods=['post'])
    def mark_as_sent(self, request, pk=None):
        """Mark a quote as sent."""
        quote = self.get_object()
        quote.status = Quote.SENT
        quote.save()
        
        return Response({
            'message': 'Quote marked as sent',
            'quote': QuoteSerializer(quote).data
        })
    
    @extend_schema(tags=['quotes'])
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get quote statistics."""
        queryset = self.get_queryset()
        
        stats = {
            'total_quotes': queryset.count(),
            'by_status': {
                'draft': queryset.filter(status=Quote.DRAFT).count(),
                'generated': queryset.filter(status=Quote.GENERATED).count(),
                'sent': queryset.filter(status=Quote.SENT).count(),
            },
            'by_type': {
                'property_quotes': queryset.filter(quote_type=Quote.PROPERTY_QUOTE).count(),
                'comparison_quotes': queryset.filter(quote_type=Quote.COMPARISON_QUOTE).count(),
            }
        }
        
        # Calculate average amounts
        property_quotes = queryset.filter(quote_type=Quote.PROPERTY_QUOTE)
        if property_quotes.exists():
            avg_amount = sum(float(q.total_amount) for q in property_quotes) / property_quotes.count()
            stats['average_quote_amount'] = round(avg_amount, 2)
        
        return Response(stats)