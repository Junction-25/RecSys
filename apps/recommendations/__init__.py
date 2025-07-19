"""
Recommendations app for the Real Estate platform.
Provides property and contact recommendation services.
"""

# Import functions in a way that prevents circular imports
def get_recommendation_engine():
    from .service_registry import get_recommendation_engine as _get_recommendation_engine
    return _get_recommendation_engine()

def get_hyper_engine():
    from .service_registry import get_hyper_engine as _get_hyper_engine
    return _get_hyper_engine()

def get_embedding_service():
    from .service_registry import get_embedding_service as _get_embedding_service
    return _get_embedding_service()

def get_gemini_explainer():
    from .service_registry import get_gemini_explainer as _get_gemini_explainer
    return _get_gemini_explainer()

def get_filter_engine():
    from .service_registry import get_filter_engine as _get_filter_engine
    return _get_filter_engine()

def clear_service_cache():
    from .service_registry import clear_service_cache as _clear_service_cache
    return _clear_service_cache()

__all__ = [
    'get_recommendation_engine',
    'get_hyper_engine',
    'get_embedding_service',
    'get_gemini_explainer',
    'get_filter_engine',
    'clear_service_cache'
]