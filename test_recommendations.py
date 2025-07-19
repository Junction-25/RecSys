#!/usr/bin/env python
"""
Test the TikTok-like recommendation system with all embedded data.
"""
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.recommendations.service_registry import ServiceRegistry
from apps.properties.models import Property

def test_recommendations():
    print('🧪 Testing TikTok-like Recommendation System...')
    print('=' * 50)

    # Get the engine
    engine = ServiceRegistry.get_recommendation_engine()

    # Get vector database stats
    vector_stats = engine.get_vector_db_stats()
    print(f'📈 Vector Database Stats:')
    print(f'   • Total Buyers: {vector_stats.get("total_buyers", 0):,}')
    print(f'   • Embedding Dimension: {vector_stats.get("embedding_dim", 0)}')
    print(f'   • Index Type: {vector_stats.get("index_type", "Unknown")}')
    print(f'   • Memory Usage: {vector_stats.get("memory_usage_mb", 0):.2f} MB')
    print()

    # Test recommendations for a few properties
    test_properties = Property.objects.all()[:3]

    for prop in test_properties:
        print(f'🔍 Testing Property {prop.id}: {prop.title}')
        print(f'   📍 Location: {prop.city}')
        print(f'   💰 Price: ${prop.price:,}')
        print(f'   📐 Area: {prop.area} sqm, {prop.rooms} rooms')
        
        try:
            recommendations = engine.get_recommendations_for_property(
                property_id=prop.id,
                max_results=5
            )
            
            rec_count = len(recommendations.get('recommendations', []))
            processing_time = recommendations.get('processing_time_ms', 0)
            
            print(f'   ✅ Found {rec_count} recommendations in {processing_time}ms')
            
            for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
                buyer_name = rec.get('buyer_name', 'Unknown')
                score = rec.get('score', 0)
                print(f'      {i}. {buyer_name} (Score: {score:.3f})')
            
        except Exception as e:
            print(f'   ❌ Error: {e}')
        
        print()

    print('🎉 TikTok-like Recommendation System Test Complete!')

if __name__ == "__main__":
    test_recommendations()