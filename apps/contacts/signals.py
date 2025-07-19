"""
Django signals for automatic contact/buyer embedding generation.
Generates embeddings when contacts are created or updated and adds them to vector DB.
"""
import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache
from .models import Contact

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Contact)
def generate_contact_embedding_and_add_to_vector_db(sender, instance, created, **kwargs):
    """
    Generate embedding for contact and add to vector database.
    Uses the TikTok-like engine for embedding generation and vector storage.
    """
    try:
        # Import here to avoid circular imports
        from apps.recommendations.service_registry import get_recommendation_engine
        import pickle
        
        # Get the recommendation engine (singleton)
        engine = get_recommendation_engine()
        
        # Fit the neural binning if this is the first contact being processed
        if created and Contact.objects.count() <= 10:  # Only fit on first few contacts
            try:
                contacts = list(Contact.objects.all()[:100])  # Sample for fitting
                engine.property_encoder.neural_binning.fit_buyer_features(contacts)
                logger.info("Fitted neural binning for buyer features")
            except Exception as fit_error:
                logger.warning(f"Could not fit neural binning: {fit_error}")
        
        # Generate buyer embedding using the engine
        buyer_embedding = engine.encode_buyer(instance)
        
        # Store embedding as binary data in the model
        instance.embedding = pickle.dumps(buyer_embedding)
        
        # Save without triggering signals again
        Contact.objects.filter(pk=instance.pk).update(embedding=instance.embedding)
        
        # Add to vector database for fast ANN retrieval
        buyer_metadata = {
            'buyer_id': str(instance.id),
            'name': instance.name,
            'email': instance.email,
            'phone': instance.phone,
            'budget_min': float(instance.budget_min) if instance.budget_min else None,
            'budget_max': float(instance.budget_max) if instance.budget_max else None,
            'property_type': instance.property_type,
            'preferred_locations': instance.preferred_locations,
            'status': instance.status,
        }
        
        engine.add_buyer_to_vector_db(str(instance.id), instance, buyer_metadata)
        
        action = "Created" if created else "Updated"
        logger.info(f"{action} embedding and vector DB entry for contact {instance.id} ({instance.name})")
        
        # Clear related caches
        cache.delete_many([
            f'contact_embedding_{instance.id}',
            f'contact_recommendations_{instance.id}*',
            'property_recommendations_*',
            'batch_recommendations_*'
        ])
        
    except Exception as e:
        logger.error(f"Error generating embedding for contact {instance.id}: {e}")


@receiver(post_delete, sender=Contact)
def remove_contact_from_vector_db(sender, instance, **kwargs):
    """Remove contact from vector database and clean up caches when deleted."""
    try:
        # Import here to avoid circular imports
        from apps.recommendations.service_registry import get_recommendation_engine
        
        # Get the recommendation engine
        engine = get_recommendation_engine()
        
        # Remove from vector database
        engine.vector_db.remove_buyer(str(instance.id))
        
        # Clear related caches
        cache.delete_many([
            f'contact_embedding_{instance.id}',
            f'contact_recommendations_{instance.id}*',
            'property_recommendations_*',
            'batch_recommendations_*'
        ])
        
        logger.info(f"Removed contact {instance.id} from vector DB and cleaned up cache")
        
    except Exception as e:
        logger.error(f"Error removing contact {instance.id} from vector DB: {e}")