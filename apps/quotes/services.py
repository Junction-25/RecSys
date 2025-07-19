"""
Services for PDF quote generation.
"""
import io
from decimal import Decimal
from django.conf import settings
from django.utils import timezone
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

from .models import Quote
from apps.properties.models import Property
from apps.contacts.models import Contact


class QuoteGenerationService:
    """Service for generating PDF quotes."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='PropertyTitle',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.black
        ))
    
    def generate_property_quote(self, property_id, contact_id, additional_fees=None):
        """
        Generate a property quote PDF.
        
        Args:
            property_id: Property ID (UUID or external_id)
            contact_id: Contact ID (UUID or external_id)
            additional_fees: Dictionary of additional fees
            
        Returns:
            Quote object with generated PDF
        """
        # Get property and contact
        try:
            property_obj = Property.objects.get(id=property_id)
        except (ValueError, Property.DoesNotExist):
            property_obj = Property.objects.get(external_id=property_id)
        
        try:
            contact = Contact.objects.get(id=contact_id)
        except (ValueError, Contact.DoesNotExist):
            contact = Contact.objects.get(external_id=contact_id)
        
        # Calculate total amount
        base_price = property_obj.price
        additional_fees = additional_fees or {}
        total_fees = sum(Decimal(str(fee)) for fee in additional_fees.values())
        total_amount = base_price + total_fees
        
        # Create quote record
        quote = Quote.objects.create(
            quote_type=Quote.PROPERTY_QUOTE,
            property=property_obj,
            contact=contact,
            base_price=base_price,
            additional_fees=additional_fees,
            total_amount=total_amount,
            status=Quote.GENERATED
        )
        
        # Generate PDF
        pdf_buffer = self._create_property_quote_pdf(quote)
        
        # Save PDF to quote
        quote.pdf_file.save(
            f'quote_{quote.quote_number}.pdf',
            pdf_buffer,
            save=True
        )
        
        return quote
    
    def generate_comparison_quote(self, property_id_1, property_id_2, contact_id):
        """
        Generate a property comparison quote PDF.
        
        Args:
            property_id_1: First property ID
            property_id_2: Second property ID
            contact_id: Contact ID
            
        Returns:
            Quote object with generated PDF
        """
        # Get properties and contact
        try:
            property_1 = Property.objects.get(id=property_id_1)
        except (ValueError, Property.DoesNotExist):
            property_1 = Property.objects.get(external_id=property_id_1)
        
        try:
            property_2 = Property.objects.get(id=property_id_2)
        except (ValueError, Property.DoesNotExist):
            property_2 = Property.objects.get(external_id=property_id_2)
        
        try:
            contact = Contact.objects.get(id=contact_id)
        except (ValueError, Contact.DoesNotExist):
            contact = Contact.objects.get(external_id=contact_id)
        
        # Create quote record
        quote = Quote.objects.create(
            quote_type=Quote.COMPARISON_QUOTE,
            property=property_1,
            comparison_property=property_2,
            contact=contact,
            base_price=property_1.price,
            total_amount=property_1.price,
            status=Quote.GENERATED
        )
        
        # Generate PDF
        pdf_buffer = self._create_comparison_quote_pdf(quote)
        
        # Save PDF to quote
        quote.pdf_file.save(
            f'comparison_{quote.quote_number}.pdf',
            pdf_buffer,
            save=True
        )
        
        return quote
    
    def _create_property_quote_pdf(self, quote):
        """Create PDF for property quote."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("Property Quote", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Quote information
        quote_info = [
            ['Quote Number:', quote.quote_number],
            ['Date:', timezone.now().strftime('%B %d, %Y')],
            ['Status:', quote.get_status_display()],
        ]
        
        quote_table = Table(quote_info, colWidths=[2*inch, 3*inch])
        quote_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(quote_table)
        story.append(Spacer(1, 20))
        
        # Contact information
        story.append(Paragraph("Contact Information", self.styles['SectionHeader']))
        contact_info = [
            ['Name:', quote.contact.name],
            ['Email:', quote.contact.email],
            ['Phone:', quote.contact.phone],
        ]
        
        contact_table = Table(contact_info, colWidths=[2*inch, 4*inch])
        contact_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(contact_table)
        story.append(Spacer(1, 20))
        
        # Property information
        story.append(Paragraph("Property Details", self.styles['SectionHeader']))
        property_info = [
            ['Title:', quote.property.title],
            ['Type:', quote.property.get_property_type_display()],
            ['Location:', f"{quote.property.city}, {quote.property.district}"],
            ['Address:', quote.property.address],
            ['Area:', f"{quote.property.area} m²"],
            ['Rooms:', str(quote.property.rooms) if quote.property.rooms else 'N/A'],
            ['Bathrooms:', str(quote.property.bathrooms) if quote.property.bathrooms else 'N/A'],
        ]
        
        property_table = Table(property_info, colWidths=[2*inch, 4*inch])
        property_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(property_table)
        story.append(Spacer(1, 20))
        
        # Features
        if quote.property.features_list:
            story.append(Paragraph("Property Features", self.styles['SectionHeader']))
            features_text = ", ".join(quote.property.features_list)
            story.append(Paragraph(features_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Pricing breakdown
        story.append(Paragraph("Pricing Breakdown", self.styles['SectionHeader']))
        
        pricing_data = [['Description', 'Amount']]
        pricing_data.append(['Base Price', f"${quote.base_price:,.2f}"])
        
        # Add additional fees
        for fee_name, fee_amount in quote.additional_fees.items():
            pricing_data.append([fee_name.replace('_', ' ').title(), f"${fee_amount:,.2f}"])
        
        # Add total
        pricing_data.append(['', ''])  # Empty row for spacing
        pricing_data.append(['TOTAL AMOUNT', f"${quote.total_amount:,.2f}"])
        
        pricing_table = Table(pricing_data, colWidths=[4*inch, 2*inch])
        pricing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -2), 1, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
        ]))
        story.append(pricing_table)
        story.append(Spacer(1, 30))
        
        # Notes
        if quote.notes:
            story.append(Paragraph("Additional Notes", self.styles['SectionHeader']))
            story.append(Paragraph(quote.notes, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Footer
        footer_text = "This quote is valid for 30 days from the date of generation."
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_comparison_quote_pdf(self, quote):
        """Create PDF for property comparison quote."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("Property Comparison Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Quote information
        quote_info = [
            ['Report Number:', quote.quote_number],
            ['Date:', timezone.now().strftime('%B %d, %Y')],
            ['Prepared for:', quote.contact.name],
        ]
        
        quote_table = Table(quote_info, colWidths=[2*inch, 3*inch])
        quote_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(quote_table)
        story.append(Spacer(1, 30))
        
        # Comparison table
        story.append(Paragraph("Property Comparison", self.styles['SectionHeader']))
        
        comparison_data = [
            ['Feature', 'Property 1', 'Property 2'],
            ['Title', quote.property.title, quote.comparison_property.title],
            ['Type', quote.property.get_property_type_display(), quote.comparison_property.get_property_type_display()],
            ['Location', f"{quote.property.city}, {quote.property.district}", f"{quote.comparison_property.city}, {quote.comparison_property.district}"],
            ['Price', f"${quote.property.price:,.2f}", f"${quote.comparison_property.price:,.2f}"],
            ['Area', f"{quote.property.area} m²", f"{quote.comparison_property.area} m²"],
            ['Price per m²', f"${float(quote.property.price)/float(quote.property.area):,.2f}", f"${float(quote.comparison_property.price)/float(quote.comparison_property.area):,.2f}"],
            ['Rooms', str(quote.property.rooms) if quote.property.rooms else 'N/A', str(quote.comparison_property.rooms) if quote.comparison_property.rooms else 'N/A'],
            ['Bathrooms', str(quote.property.bathrooms) if quote.property.bathrooms else 'N/A', str(quote.comparison_property.bathrooms) if quote.comparison_property.bathrooms else 'N/A'],
        ]
        
        comparison_table = Table(comparison_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(comparison_table)
        story.append(Spacer(1, 30))
        
        # Analysis
        story.append(Paragraph("Analysis", self.styles['SectionHeader']))
        
        price_diff = float(quote.comparison_property.price) - float(quote.property.price)
        area_diff = float(quote.comparison_property.area) - float(quote.property.area)
        
        analysis_text = f"""
        <b>Price Comparison:</b><br/>
        Property 2 is ${abs(price_diff):,.2f} {'more expensive' if price_diff > 0 else 'less expensive'} than Property 1.<br/><br/>
        
        <b>Area Comparison:</b><br/>
        Property 2 has {abs(area_diff):,.1f} m² {'more' if area_diff > 0 else 'less'} space than Property 1.<br/><br/>
        
        <b>Value Analysis:</b><br/>
        Based on price per square meter, {'Property 1' if float(quote.property.price)/float(quote.property.area) < float(quote.comparison_property.price)/float(quote.comparison_property.area) else 'Property 2'} offers better value for money.
        """
        
        story.append(Paragraph(analysis_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer