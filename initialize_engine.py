import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.properties.models import Property
from apps.contacts.models import Contact
from apps.recommendations.engine import RecommendationEngine

def initialize_recommendation_engine():
    print("Initializing TikTok-like Recommendation Engine...")
    
    # Create engine instance
    engine = RecommendationEngine()
    
    # Get all properties and contacts
    properties = list(Property.objects.all()[:100])  # Limit for faster initialization
    contacts = list(Contact.objects.all()[:500])     # Limit for faster initialization
    
    print(f"Fitting engine with {len(properties)} properties and {len(contacts)} contacts...")
    
    # Fit the system
    engine.fit_system(properties, contacts)
    
    print("Engine initialization completed!")
    print(f"Vector DB stats: {engine.get_performance_stats()}")

if __name__ == '__main__':
    initialize_recommendation_engine()