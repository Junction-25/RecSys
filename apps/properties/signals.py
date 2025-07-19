"""
Django signals for automatic property embedding generation.
Generates embeddings when properties are created or updated.
"""
import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache
from .models import Property

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Property)
def generate_property_embedding(sender, instance, created, **kwargs):
    """
    Generate embedding for property when created or updated.
    Uses the TikTok-like PropertyEncoder for embedding generation.
    """
    try:
        # Import here to avoid circular imports
        from apps.recommendations.service_registry import get_recommendation_engine
        import pickle
        import numpy as np
        
        # Get the recommendation engine (singleton)
        engine = get_recommendation_engine()
        
        # Fit the neural binning if this is the first property being processed
        if created and Property.objects.count() <= 10:  # Only fit on first few properties
            try:
                properties = list(Property.objects.all()[:100])  # Sample for fitting
                engine.property_encoder.neural_binning.fit_property_features(properties)
                logger.info("Fitted neural binning for property features")
            except Exception as fit_error:
                logger.warning(f"Could not fit neural binning: {fit_error}")
        
        # Generate embedding
        embedding = engine.encode_property(instance)
        
        # Store embedding as binary data
        instance.embedding = pickle.dumps(embedding)
        
        # Save without triggering signals again
        Property.objects.filter(pk=instance.pk).update(embedding=instance.embedding)
        
        logger.info(f"Generated embedding for property {instance.id} ({instance.title})")
        
        # Clear related caches
        cache.delete_many([
            f'property_embedding_{instance.id}',
            f'property_recommendations_{instance.id}*',
            'batch_recommendations_*'
        ])
        
    except Exception as e:
        logger.error(f"Error generating embedding for property {instance.id}: {e}")


@receiver(post_delete, sender=Property)
def cleanup_property_cache(sender, instance, **kwargs):
    """Clean up caches when property is deleted."""
    try:
        # Clear related caches
        cache.delete_many([
            f'property_embedding_{instance.id}',
            f'property_recommendations_{instance.id}*',
            'batch_recommendations_*'
        ])
        
        logger.info(f"Cleaned up cache for deleted property {instance.id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up cache for property {instance.id}: {e}")