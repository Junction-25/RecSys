import os
import json
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()


from apps.contacts.models import Contact


def load_contacts():
    with open('data/contacts.json', 'r') as f:
        contacts = json.load(f)
    
    for contact in contacts:
        contact_data = {
            'name': contact.get('name', ''),
            'email': f"contact_{contact.get('id', '')}@example.com",
            'phone': contact.get('phone', ''),
            'budget_min': float(contact.get('min_budget', 0)) if contact.get('min_budget') is not None else 0,
            'budget_max': float(contact.get('max_budget', 1000000)) if contact.get('max_budget') is not None else 1000000,
            'preferred_locations': contact.get('preferred_locations', []),
            'property_type': contact.get('property_types', ['apartment'])[0] if contact.get('property_types') else 'apartment',
            'desired_area_min': int(contact.get('min_area_sqm', 50)) if contact.get('min_area_sqm') is not None else 50,
            'desired_area_max': int(contact.get('max_area_sqm', 200)) if contact.get('max_area_sqm') is not None else 200,
            'rooms_min': int(contact.get('min_rooms', 1)) if contact.get('min_rooms') is not None else 1,
            'rooms_max': int(contact.get('max_rooms', 4)) if contact.get('max_rooms') is not None else 4,
            'prefers_parking': True,  # Default values as they're not in the JSON
            'prefers_garden': False,
            'prefers_balcony': True,
            'prefers_elevator': False,
            'prefers_furnished': False,
            'status': 'active'
        }
        
        # Create or update contact using external_id
        external_id = contact.get('id')
        if external_id:
            Contact.objects.update_or_create(
                external_id=str(external_id),
                defaults=contact_data
            )
    
    print(f"Successfully loaded {Contact.objects.count()} contacts")

if __name__ == '__main__':
    print("Starting data loading process...")
    load_contacts()
    print("Data loading completed!")