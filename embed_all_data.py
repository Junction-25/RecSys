#!/usr/bin/env python
"""
Embed all contacts and properties using the TikTok-like recommendation engine.
This will populate the vector database with all buyer and property embeddings.
"""
import os
import django
import time
from tqdm import tqdm

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.contacts.models import Contact
from apps.properties.models import Property

def embed_all_data():
    """Embed all contacts and properties."""
    print("🚀 Starting TikTok-like Embedding Process...")
    print("=" * 60)
    
    # Get counts
    total_contacts = Contact.objects.count()
    total_properties = Property.objects.count()
    
    print(f"📊 Data Summary:")
    print(f"   • Total Contacts: {total_contacts:,}")
    print(f"   • Total Properties: {total_properties:,}")
    print(f"   • Total Items to Embed: {total_contacts + total_properties:,}")
    print()
    
    start_time = time.time()
    
    # Embed all contacts
    print("👥 Embedding Contacts (Buyers)...")
    print("-" * 40)
    
    contact_batch_size = 50
    contact_count = 0
    
    for i in tqdm(range(0, total_contacts, contact_batch_size), desc="Contact Batches"):
        contacts = Contact.objects.all()[i:i+contact_batch_size]
        
        for contact in contacts:
            try:
                # Trigger embedding generation via save signal
                contact.save()
                contact_count += 1
                
                if contact_count % 100 == 0:
                    print(f"   ✅ Embedded {contact_count:,}/{total_contacts:,} contacts")
                    
            except Exception as e:
                print(f"   ❌ Error embedding contact {contact.id}: {e}")
                continue
    
    print(f"   🎉 Completed embedding {contact_count:,} contacts!")
    print()
    
    # Embed all properties
    print("🏠 Embedding Properties...")
    print("-" * 40)
    
    property_batch_size = 25
    property_count = 0
    
    for i in tqdm(range(0, total_properties, property_batch_size), desc="Property Batches"):
        properties = Property.objects.all()[i:i+property_batch_size]
        
        for property_obj in properties:
            try:
                # Trigger embedding generation via save signal
                property_obj.save()
                property_count += 1
                
                if property_count % 50 == 0:
                    print(f"   ✅ Embedded {property_count:,}/{total_properties:,} properties")
                    
            except Exception as e:
                print(f"   ❌ Error embedding property {property_obj.id}: {e}")
                continue
    
    print(f"   🎉 Completed embedding {property_count:,} properties!")
    print()
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("🎯 Embedding Summary:")
    print("=" * 60)
    print(f"   • Contacts Embedded: {contact_count:,}/{total_contacts:,}")
    print(f"   • Properties Embedded: {property_count:,}/{total_properties:,}")
    print(f"   • Total Time: {total_time:.1f} seconds")
    print(f"   • Average Time per Item: {total_time/(contact_count + property_count):.2f}s")
    print()
    
    # Save the engine state to disk
    print("💾 Saving Engine State to Disk...")
    print("-" * 40)
    
    try:
        # Import here to avoid circular imports
        from apps.recommendations.service_registry import get_recommendation_engine
        import os
        
        engine = get_recommendation_engine()
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"   📁 Created {models_dir} directory")
        
        # Save the entire engine state using the correct method
        engine_path = os.path.join(models_dir, "recommendation_engine")
        engine.save_system(engine_path)
        print(f"   ✅ Saved engine state to {engine_path}")
        print(f"   📊 Vector DB: {engine.vector_db.get_database_stats()['total_buyers']:,} buyers")
        print()
        
    except Exception as e:
        print(f"   ❌ Error saving engine state: {e}")
        print("   ⚠️  Continuing with testing...")
    
    # Test the system
    print("🧪 Testing Recommendation System...")
    print("-" * 40)
    
    try:
        engine = get_recommendation_engine()
        
        # Get vector database stats
        vector_stats = engine.get_vector_db_stats()
        print(f"   📈 Vector Database Stats:")
        print(f"      • Total Buyers in Vector DB: {vector_stats.get('total_buyers', 0):,}")
        print(f"      • Embedding Dimension: {vector_stats.get('embedding_dim', 0)}")
        print(f"      • Index Type: {vector_stats.get('index_type', 'Unknown')}")
        print(f"      • Memory Usage: {vector_stats.get('memory_usage_mb', 0):.2f} MB")
        print()
        
        # Test a recommendation
        if total_properties > 0:
            test_property = Property.objects.first()
            print(f"   🔍 Testing recommendations for: {test_property.title}")
            
            recommendations = engine.get_recommendations_for_property(
                property_id=test_property.id,
                max_results=5
            )
            
            print(f"   📋 Found {len(recommendations.get('recommendations', []))} recommendations")
            
            for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
                buyer_name = rec.get('buyer_name', 'Unknown')
                score = rec.get('score', 0)
                print(f"      {i}. {buyer_name} (Score: {score:.3f})")
        
        print()
        print("✅ TikTok-like Recommendation System is Ready!")
        print("🚀 You can now test the API endpoints!")
        
    except Exception as e:
        print(f"   ❌ Error testing system: {e}")
        print("   ⚠️  System may need restart to load embeddings properly")

if __name__ == "__main__":
    embed_all_data()