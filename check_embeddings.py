#!/usr/bin/env python
"""
Check if embeddings are properly saved and can be loaded.
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.recommendations.service_registry import ServiceRegistry
from apps.contacts.models import Contact
from apps.properties.models import Property

def check_embeddings():
    """Check the current state of embeddings."""
    print("🔍 Checking Embedding Status...")
    print("=" * 50)
    
    # Check database embeddings
    contacts_with_embeddings = Contact.objects.exclude(embedding__isnull=True).count()
    total_contacts = Contact.objects.count()
    
    properties_with_embeddings = Property.objects.exclude(embedding__isnull=True).count()
    total_properties = Property.objects.count()
    
    print(f"📊 Database Embeddings:")
    print(f"   • Contacts: {contacts_with_embeddings:,}/{total_contacts:,} have embeddings")
    print(f"   • Properties: {properties_with_embeddings:,}/{total_properties:,} have embeddings")
    print()
    
    # Check saved engine files
    models_dir = "models"
    engine_path = os.path.join(models_dir, "recommendation_engine")
    
    saved_files = []
    expected_files = [
        f"{engine_path}_vector_db.faiss",
        f"{engine_path}_vector_db.pkl",
        f"{engine_path}_property_encoder.pkl",
        f"{engine_path}_deepfm_ranker.pth"
    ]
    
    print(f"💾 Saved Engine Files:")
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {os.path.basename(file_path)} ({file_size:.2f} MB)")
            saved_files.append(file_path)
        else:
            print(f"   ❌ {os.path.basename(file_path)} (missing)")
    
    print()
    
    # Test loading the engine
    print(f"🚀 Testing Engine Loading...")
    try:
        engine = ServiceRegistry.get_recommendation_engine()
        stats = engine.get_vector_db_stats()
        
        print(f"   ✅ Engine loaded successfully")
        print(f"   📈 Vector DB Stats:")
        print(f"      • Total Buyers: {stats.get('total_buyers', 0):,}")
        print(f"      • Embedding Dim: {stats.get('embedding_dim', 0)}")
        print(f"      • Index Type: {stats.get('index_type', 'Unknown')}")
        print(f"      • Memory Usage: {stats.get('memory_usage_mb', 0):.2f} MB")
        
        # Test a quick recommendation if we have data
        if total_properties > 0 and stats.get('total_buyers', 0) > 0:
            print()
            print(f"🧪 Testing Recommendations...")
            test_property = Property.objects.first()
            recommendations = engine.get_recommendations_for_property(
                property_id=test_property.id,
                max_results=3
            )
            
            rec_count = len(recommendations.get('recommendations', []))
            print(f"   ✅ Generated {rec_count} recommendations for '{test_property.title}'")
            
            for i, rec in enumerate(recommendations.get('recommendations', [])[:2], 1):
                buyer_name = rec.get('buyer_name', 'Unknown')
                score = rec.get('score', 0)
                print(f"      {i}. {buyer_name} (Score: {score:.3f})")
        
    except Exception as e:
        print(f"   ❌ Error loading engine: {e}")
    
    print()
    
    # Summary
    if len(saved_files) == len(expected_files):
        print("✅ All embeddings are properly saved and can be loaded!")
    elif len(saved_files) > 0:
        print("⚠️  Some embeddings are saved, but not all components")
        print("   Run 'python embed_all_data.py' to generate missing components")
    else:
        print("❌ No saved embeddings found")
        print("   Run 'python embed_all_data.py' to generate embeddings")

if __name__ == "__main__":
    check_embeddings()