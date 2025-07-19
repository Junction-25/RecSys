import os
import json
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from apps.properties.models import Property

def load_properties():
    with open('data/properties.json', 'r') as f:
        properties = json.load(f)
    
    for prop in properties:
        property_data = {
            'title': f"Property {prop.get('id', '')}",
            'description': f"Beautiful property at {prop.get('address', '')}",
            'address': prop.get('address', ''),
            'city': prop.get('address', '').split(',')[-1].strip() if prop.get('address') else 'Unknown',
            'district': prop.get('address', '').split(',')[0].strip() if prop.get('address') else 'Unknown',
            'price': float(prop.get('price', 0)),
            'area': int(prop.get('area_sqm', 100)),
            'rooms': int(prop.get('rooms', 3)),
            'property_type': prop.get('property_type', 'apartment'),
            'status': 'available',
            'has_parking': prop.get('has_parking', False),
            'has_garden': prop.get('has_garden', False),
            'has_balcony': prop.get('has_balcony', True),
            'has_elevator': prop.get('has_elevator', False),
            'is_furnished': prop.get('is_furnished', False),
            'latitude': prop.get('location', {}).get('lat'),
            'longitude': prop.get('location', {}).get('lon'),
        }
        
        # Create or update property using external_id
        external_id = prop.get('id')
        if external_id:
            Property.objects.update_or_create(
                external_id=str(external_id),
                defaults=property_data
            )
    
    print(f"Successfully loaded {Property.objects.count()} properties")

if __name__ == '__main__':
    print("Starting property loading process...")
    load_properties()
    print("Property loading completed!")