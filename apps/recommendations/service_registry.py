"""
Service registry for managing singleton instances of recommendation services.
Ensures the TikTok-like engine is loaded only once and reused across requests.
"""
import logging
from django.core.cache import cache

from .engine import RecommendationEngine

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Service registry for managing singleton instances."""
    
    _recommendation_engine = None
    
    @classmethod
    def get_recommendation_engine(cls):
        """Get or create a singleton instance of RecommendationEngine."""
        if cls._recommendation_engine is None:
            logger.info("Initializing TikTok-like RecommendationEngine...")
            cls._recommendation_engine = RecommendationEngine()
            
            # Try to load saved engine state
            import os
            engine_path = os.path.join("models", "recommendation_engine")
            if os.path.exists(f"{engine_path}_vector_db.faiss"):
                try:
                    logger.info("Loading saved engine state...")
                    cls._recommendation_engine.load_engine(engine_path)
                    stats = cls._recommendation_engine.get_vector_db_stats()
                    logger.info(f"Loaded engine with {stats.get('total_buyers', 0):,} buyers in vector DB")
                except Exception as e:
                    logger.warning(f"Could not load saved engine state: {e}")
                    logger.info("Starting with fresh engine state")
            else:
                logger.info("No saved engine state found, starting fresh")
                
        return cls._recommendation_engine
    
    @classmethod
    def clear_service_cache(cls):
        """Clear cached service instance."""
        cls._recommendation_engine = None
        logger.info("Cleared recommendation engine cache")


# Global instance functions for backward compatibility
_recommendation_engine = None

def get_recommendation_engine():
    """Get or create a singleton instance of RecommendationEngine."""
    return ServiceRegistry.get_recommendation_engine()

def clear_service_cache():
    """Clear cached service instance."""
    return ServiceRegistry.clear_service_cache()
