"""
Management command to generate embeddings for all properties and contacts.
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.recommendations.embedding_service import EmbeddingService


class Command(BaseCommand):
    """Command to generate and store embeddings for properties and contacts."""
    
    help = 'Generate vector embeddings for all properties and contacts'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of existing embeddings'
        )
        parser.add_argument(
            '--properties-only',
            action='store_true',
            help='Generate embeddings only for properties'
        )
        parser.add_argument(
            '--contacts-only',
            action='store_true',
            help='Generate embeddings only for contacts'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Batch size for processing (default: 50)'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        start_time = timezone.now()
        
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting embedding generation...')
        )
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Get initial statistics
        initial_stats = embedding_service.get_embedding_stats()
        self.stdout.write(
            f"📊 Initial stats: "
            f"{initial_stats['properties_with_embeddings']}/{initial_stats['total_properties']} properties, "
            f"{initial_stats['contacts_with_embeddings']}/{initial_stats['total_contacts']} contacts"
        )
        
        try:
            if options['properties_only']:
                self._generate_property_embeddings(embedding_service, options)
            elif options['contacts_only']:
                self._generate_contact_embeddings(embedding_service, options)
            else:
                self._generate_property_embeddings(embedding_service, options)
                self._generate_contact_embeddings(embedding_service, options)
            
            # Build FAISS indexes
            self.stdout.write("🔍 Building FAISS indexes...")
            embedding_service.build_faiss_index(rebuild=True)
            
            # Get final statistics
            final_stats = embedding_service.get_embedding_stats()
            
            # Calculate processing time
            end_time = timezone.now()
            duration = (end_time - start_time).total_seconds()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ Embedding generation completed in {duration:.2f} seconds!\n"
                    f"📈 Final stats: "
                    f"{final_stats['properties_with_embeddings']}/{final_stats['total_properties']} properties "
                    f"({final_stats['property_embedding_coverage']:.1%}), "
                    f"{final_stats['contacts_with_embeddings']}/{final_stats['total_contacts']} contacts "
                    f"({final_stats['contact_embedding_coverage']:.1%})"
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error during embedding generation: {e}")
            )
            raise
    
    def _generate_property_embeddings(self, embedding_service, options):
        """Generate embeddings for properties."""
        from apps.properties.models import Property
        
        self.stdout.write("🏠 Generating property embeddings...")
        
        # Get properties that need embeddings
        if options['force']:
            properties = Property.objects.all()
        else:
            properties = Property.objects.filter(embedding__isnull=True)
        
        total_properties = properties.count()
        if total_properties == 0:
            self.stdout.write("   No properties need embedding generation.")
            return
        
        batch_size = options['batch_size']
        processed = 0
        
        for i in range(0, total_properties, batch_size):
            batch = properties[i:i + batch_size]
            
            for property_obj in batch:
                try:
                    # Generate and store embedding
                    embedding = embedding_service.generate_property_embedding(property_obj)
                    embedding_service.store_property_embedding(property_obj, embedding)
                    processed += 1
                    
                    if processed % 10 == 0:
                        self.stdout.write(f"   Processed {processed}/{total_properties} properties...")
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(
                            f"   Warning: Failed to generate embedding for property {property_obj.external_id}: {e}"
                        )
                    )
        
        self.stdout.write(f"   ✅ Generated embeddings for {processed} properties")
    
    def _generate_contact_embeddings(self, embedding_service, options):
        """Generate embeddings for contacts."""
        from apps.contacts.models import Contact
        
        self.stdout.write("👥 Generating contact embeddings...")
        
        # Get contacts that need embeddings
        if options['force']:
            contacts = Contact.objects.all()
        else:
            contacts = Contact.objects.filter(embedding__isnull=True)
        
        total_contacts = contacts.count()
        if total_contacts == 0:
            self.stdout.write("   No contacts need embedding generation.")
            return
        
        batch_size = options['batch_size']
        processed = 0
        
        for i in range(0, total_contacts, batch_size):
            batch = contacts[i:i + batch_size]
            
            for contact in batch:
                try:
                    # Generate and store embedding
                    embedding = embedding_service.generate_contact_embedding(contact)
                    embedding_service.store_contact_embedding(contact, embedding)
                    processed += 1
                    
                    if processed % 10 == 0:
                        self.stdout.write(f"   Processed {processed}/{total_contacts} contacts...")
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(
                            f"   Warning: Failed to generate embedding for contact {contact.external_id}: {e}"
                        )
                    )
        
        self.stdout.write(f"   ✅ Generated embeddings for {processed} contacts")