"""
Django management command to populate database with real data from JSON files
"""
import json
import logging
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from apps.properties.models import Property
from apps.contacts.models import Contact

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Populate database with real properties and contacts data from JSON files'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--properties-file',
            type=str,
            default='data/properties.json',
            help='Path to properties JSON file'
        )
        parser.add_argument(
            '--contacts-file', 
            type=str,
            default='data/contacts.json',
            help='Path to contacts JSON file'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Batch size for bulk operations'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing data before importing'
        )
        parser.add_argument(
            '--properties-only',
            action='store_true',
            help='Import only properties'
        )
        parser.add_argument(
            '--contacts-only',
            action='store_true',
            help='Import only contacts'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting Real Data Population')
        )
        
        # Clear existing data if requested
        if options['clear_existing']:
            self.clear_existing_data()
        
        # Import data
        if not options['contacts_only']:
            self.import_properties(
                options['properties_file'], 
                options['batch_size']
            )
        
        if not options['properties_only']:
            self.import_contacts(
                options['contacts_file'],
                options['batch_size'] 
            )
        
        # Generate summary
        self.generate_summary()
        
        self.stdout.write(
            self.style.SUCCESS('🎉 Data population completed successfully!')
        )
    
    def clear_existing_data(self):
        """Clear existing data from database"""
        self.stdout.write('🗑️ Clearing existing data...')
        
        with transaction.atomic():
            properties_count = Property.objects.count()
            contacts_count = Contact.objects.count()
            
            Property.objects.all().delete()
            Contact.objects.all().delete()
            
            self.stdout.write(
                f'   Deleted {properties_count} properties and {contacts_count} contacts'
            )
    
    def import_properties(self, file_path, batch_size):
        """Import properties from JSON file"""
        self.stdout.write(f'🏠 Importing properties from {file_path}...')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                properties_data = json.load(f)
            
            self.stdout.write(f'   Found {len(properties_data)} properties to import')
            
            # Process in batches
            properties_to_create = []
            created_count = 0
            skipped_count = 0
            
            for i, prop_data in enumerate(properties_data):
                try:
                    # Transform data to match our model
                    property_obj = self.transform_property_data(prop_data)
                    properties_to_create.append(property_obj)
                    
                    # Bulk create when batch is full
                    if len(properties_to_create) >= batch_size:
                        created_batch = Property.objects.bulk_create(
                            properties_to_create,
                            ignore_conflicts=True
                        )
                        created_count += len(created_batch)
                        properties_to_create = []
                        
                        self.stdout.write(f'   Processed {i + 1}/{len(properties_data)} properties...')
                
                except Exception as e:
                    skipped_count += 1
                    logger.warning(f'Skipped property {prop_data.get("id", "unknown")}: {e}')
            
            # Create remaining properties
            if properties_to_create:
                created_batch = Property.objects.bulk_create(
                    properties_to_create,
                    ignore_conflicts=True
                )
                created_count += len(created_batch)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ Properties import completed: {created_count} created, {skipped_count} skipped'
                )
            )
            
        except FileNotFoundError:
            self.stdout.write(
                self.style.ERROR(f'❌ Properties file not found: {file_path}')
            )
        except json.JSONDecodeError as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Invalid JSON in properties file: {e}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error importing properties: {e}')
            )
    
    def import_contacts(self, file_path, batch_size):
        """Import contacts from JSON file"""
        self.stdout.write(f'👥 Importing contacts from {file_path}...')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contacts_data = json.load(f)
            
            self.stdout.write(f'   Found {len(contacts_data)} contacts to import')
            
            # Process in batches
            contacts_to_create = []
            created_count = 0
            skipped_count = 0
            
            for i, contact_data in enumerate(contacts_data):
                try:
                    # Transform data to match our model
                    contact_obj = self.transform_contact_data(contact_data)
                    contacts_to_create.append(contact_obj)
                    
                    # Bulk create when batch is full
                    if len(contacts_to_create) >= batch_size:
                        created_batch = Contact.objects.bulk_create(
                            contacts_to_create,
                            ignore_conflicts=True
                        )
                        created_count += len(created_batch)
                        contacts_to_create = []
                        
                        if (i + 1) % 5000 == 0:  # Progress every 5000 records
                            self.stdout.write(f'   Processed {i + 1}/{len(contacts_data)} contacts...')
                
                except Exception as e:
                    skipped_count += 1
                    logger.warning(f'Skipped contact {contact_data.get("id", "unknown")}: {e}')
            
            # Create remaining contacts
            if contacts_to_create:
                created_batch = Contact.objects.bulk_create(
                    contacts_to_create,
                    ignore_conflicts=True
                )
                created_count += len(created_batch)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'✅ Contacts import completed: {created_count} created, {skipped_count} skipped'
                )
            )
            
        except FileNotFoundError:
            self.stdout.write(
                self.style.ERROR(f'❌ Contacts file not found: {file_path}')
            )
        except json.JSONDecodeError as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Invalid JSON in contacts file: {e}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error importing contacts: {e}')
            )
    
    def transform_property_data(self, prop_data):
        """Transform JSON property data to Django model format"""
        # Extract location info
        location = prop_data.get('location', {})
        latitude = location.get('lat')
        longitude = location.get('lon')
        
        # Parse address to extract city and district
        address = prop_data.get('address', '')
        city, district = self.parse_address(address)
        
        # Map property type
        property_type = self.normalize_property_type(prop_data.get('property_type', ''))
        
        # Generate title from address and type
        title = self.generate_property_title(address, property_type, prop_data.get('number_of_rooms', 0))
        
        # Generate features based on property type and size
        features = self.generate_property_features(prop_data)
        
        return Property(
            external_id=str(prop_data.get('id')),
            title=title,
            description=self.generate_property_description(prop_data),
            price=Decimal(str(prop_data.get('price', 0))),
            city=city,
            district=district,
            address=address,
            latitude=latitude,
            longitude=longitude,
            area=prop_data.get('area_sqm', 0),
            rooms=prop_data.get('number_of_rooms', 0),
            property_type=property_type,
            has_parking=features.get('parking', False),
            has_garden=features.get('garden', False),
            has_balcony=features.get('balcony', False),
            has_elevator=features.get('elevator', False),
            is_furnished=features.get('furnished', False),
            status=Property.AVAILABLE,
            created_at=timezone.now()
        )
    
    def transform_contact_data(self, contact_data):
        """Transform JSON contact data to Django model format"""
        # Extract preferred locations
        preferred_locations = []
        for location in contact_data.get('preferred_locations', []):
            location_name = location.get('name', '').replace('Around ', '')
            if location_name:
                preferred_locations.append(location_name)
        
        # Get primary property type
        property_types = contact_data.get('property_types', [])
        primary_property_type = self.normalize_property_type(
            property_types[0] if property_types else 'apartment'
        )
        
        # Generate email from name
        email = self.generate_email_from_name(contact_data.get('name', ''))
        
        # Calculate desired area range
        min_area = contact_data.get('min_area_sqm', 50)
        max_area = contact_data.get('max_area_sqm', 200)
        
        # Calculate room range
        min_rooms = max(1, contact_data.get('min_rooms', 1))
        max_rooms = max(min_rooms, min_rooms + 2)  # Default range of 2 rooms
        
        # Generate preferences based on property type and budget
        preferences = self.generate_contact_preferences(contact_data)
        
        return Contact(
            external_id=str(contact_data.get('id')),
            name=contact_data.get('name', ''),
            email=email,
            phone=self.generate_phone_number(),
            budget_min=Decimal(str(contact_data.get('min_budget', 0))),
            budget_max=Decimal(str(contact_data.get('max_budget', 0))),
            preferred_locations=preferred_locations,
            property_type=primary_property_type,
            desired_area_min=min_area,
            desired_area_max=max_area,
            rooms_min=min_rooms,
            rooms_max=max_rooms,
            prefers_parking=preferences.get('parking', True),
            prefers_garden=preferences.get('garden', False),
            prefers_balcony=preferences.get('balcony', True),
            prefers_elevator=preferences.get('elevator', False),
            prefers_furnished=preferences.get('furnished', False),
            status=Contact.ACTIVE,
            created_at=timezone.now()
        )
    
    def parse_address(self, address):
        """Parse address to extract city and district"""
        if not address:
            return 'Unknown', 'Unknown'
        
        # Split address and get the last part (usually city)
        parts = address.split(', ')
        if len(parts) >= 2:
            city = parts[-1].strip()
            district = parts[-2].strip() if len(parts) > 2 else 'Center'
        else:
            # Try to extract from single string
            if 'Algiers' in address:
                city = 'Algiers'
                district = 'Center'
            elif 'Constantine' in address:
                city = 'Constantine'
                district = 'Center'
            elif 'Oran' in address:
                city = 'Oran'
                district = 'Center'
            elif 'Bab' in address:
                city = 'Bab'
                district = 'Center'
            else:
                city = 'Unknown'
                district = 'Center'
        
        return city, district
    
    def normalize_property_type(self, prop_type):
        """Normalize property type to match our model choices"""
        prop_type = prop_type.lower().strip()
        
        type_mapping = {
            'apartment': Property.APARTMENT,
            'house': Property.HOUSE,
            'villa': Property.VILLA,
            'studio': Property.STUDIO,
            'office': Property.OFFICE,
            'land': Property.LAND,
            'commercial': Property.COMMERCIAL
        }
        
        return type_mapping.get(prop_type, Property.APARTMENT)
    
    def generate_property_title(self, address, property_type, rooms):
        """Generate a descriptive title for the property"""
        # Extract key parts from address
        parts = address.split(', ')
        location = parts[-1] if parts else 'Prime Location'
        
        # Create title based on type and rooms
        if property_type == Property.APARTMENT and rooms > 0:
            return f"{rooms}BR Apartment in {location}"
        elif property_type == Property.HOUSE and rooms > 0:
            return f"{rooms}BR House in {location}"
        elif property_type == Property.VILLA:
            return f"Luxury Villa in {location}"
        elif property_type == Property.STUDIO:
            return f"Modern Studio in {location}"
        elif property_type == Property.OFFICE:
            return f"Office Space in {location}"
        elif property_type == Property.LAND:
            return f"Land Plot in {location}"
        else:
            return f"Property in {location}"
    
    def generate_property_description(self, prop_data):
        """Generate a description for the property"""
        property_type = prop_data.get('property_type', 'property')
        area = prop_data.get('area_sqm', 0)
        rooms = prop_data.get('number_of_rooms', 0)
        price = prop_data.get('price', 0)
        
        descriptions = []
        
        if property_type == 'apartment':
            descriptions.append(f"Spacious {rooms}-bedroom apartment")
        elif property_type == 'house':
            descriptions.append(f"Beautiful {rooms}-bedroom house")
        elif property_type == 'villa':
            descriptions.append("Luxury villa with premium finishes")
        elif property_type == 'studio':
            descriptions.append("Modern studio apartment")
        elif property_type == 'office':
            descriptions.append("Professional office space")
        elif property_type == 'land':
            descriptions.append("Prime land for development")
        
        descriptions.append(f"featuring {area}m² of living space")
        
        if rooms > 3:
            descriptions.append("perfect for families")
        elif rooms <= 2:
            descriptions.append("ideal for professionals or couples")
        
        if area > 200:
            descriptions.append("with generous room sizes")
        
        descriptions.append("in a desirable location")
        
        return ". ".join(descriptions) + "."
    
    def generate_property_features(self, prop_data):
        """Generate realistic features based on property data"""
        property_type = prop_data.get('property_type', '')
        area = prop_data.get('area_sqm', 0)
        price = prop_data.get('price', 0)
        rooms = prop_data.get('number_of_rooms', 0)
        
        features = {}
        
        # Parking - more likely for houses, villas, and larger apartments
        if property_type in ['house', 'villa'] or area > 100:
            features['parking'] = True
        elif property_type == 'apartment' and rooms >= 2:
            features['parking'] = True  # 70% chance
        else:
            features['parking'] = False
        
        # Garden - mainly for houses and villas
        if property_type in ['house', 'villa']:
            features['garden'] = True
        else:
            features['garden'] = False
        
        # Balcony - common for apartments
        if property_type == 'apartment' and rooms >= 2:
            features['balcony'] = True
        elif property_type in ['house', 'villa']:
            features['balcony'] = True  # Terrace/balcony
        else:
            features['balcony'] = False
        
        # Elevator - for apartments in buildings
        if property_type == 'apartment':
            features['elevator'] = True  # Assume most apartments have elevators
        else:
            features['elevator'] = False
        
        # Furnished - based on price and type
        if property_type in ['studio', 'apartment'] and price > 20000000:  # Higher-end properties
            features['furnished'] = True
        else:
            features['furnished'] = False
        
        return features
    
    def generate_contact_preferences(self, contact_data):
        """Generate realistic preferences based on contact data"""
        property_types = contact_data.get('property_types', [])
        budget_max = contact_data.get('max_budget', 0)
        min_area = contact_data.get('min_area_sqm', 0)
        
        preferences = {}
        
        # Parking preference - higher for families and higher budgets
        if budget_max > 20000000 or min_area > 80:
            preferences['parking'] = True
        else:
            preferences['parking'] = True  # Most people want parking
        
        # Garden preference - more for houses
        if 'house' in property_types or 'villa' in property_types:
            preferences['garden'] = True
        else:
            preferences['garden'] = False
        
        # Balcony preference - common desire
        preferences['balcony'] = True
        
        # Elevator preference - for older clients or higher floors
        preferences['elevator'] = True  # Generally preferred
        
        # Furnished preference - for higher budgets or offices
        if 'office' in property_types or budget_max > 25000000:
            preferences['furnished'] = True
        else:
            preferences['furnished'] = False
        
        return preferences
    
    def generate_email_from_name(self, name):
        """Generate email from contact name"""
        if not name:
            return 'contact@example.com'
        
        # Convert name to email format
        name_parts = name.lower().split()
        if len(name_parts) >= 2:
            email = f"{name_parts[0]}.{name_parts[-1]}@example.com"
        else:
            email = f"{name_parts[0]}@example.com"
        
        return email.replace(' ', '')
    
    def generate_phone_number(self):
        """Generate a placeholder phone number"""
        import random
        return f"+213{random.randint(100000000, 999999999)}"
    
    def generate_summary(self):
        """Generate import summary"""
        self.stdout.write('\n📊 Import Summary:')
        self.stdout.write('-' * 40)
        
        properties_count = Property.objects.count()
        contacts_count = Contact.objects.count()
        
        self.stdout.write(f'Properties in database: {properties_count:,}')
        self.stdout.write(f'Contacts in database: {contacts_count:,}')
        
        # Property type breakdown
        self.stdout.write('\n🏠 Property Types:')
        for prop_type, _ in Property.PROPERTY_TYPE_CHOICES:
            count = Property.objects.filter(property_type=prop_type).count()
            if count > 0:
                self.stdout.write(f'  {prop_type.title()}: {count:,}')
        
        # Price ranges
        self.stdout.write('\n💰 Price Ranges:')
        price_ranges = [
            (0, 10000000, 'Under 10M'),
            (10000000, 25000000, '10M - 25M'),
            (25000000, 50000000, '25M - 50M'),
            (50000000, 100000000, '50M - 100M'),
            (100000000, float('inf'), 'Over 100M')
        ]
        
        for min_price, max_price, label in price_ranges:
            if max_price == float('inf'):
                count = Property.objects.filter(price__gte=min_price).count()
            else:
                count = Property.objects.filter(price__gte=min_price, price__lt=max_price).count()
            if count > 0:
                self.stdout.write(f'  {label}: {count:,}')
        
        # Location breakdown
        self.stdout.write('\n📍 Top Locations:')
        from django.db.models import Count
        top_cities = Property.objects.values('city').annotate(
            count=Count('city')
        ).order_by('-count')[:5]
        
        for city_data in top_cities:
            self.stdout.write(f"  {city_data['city']}: {city_data['count']:,}")
        
        self.stdout.write('\n✅ Database population completed successfully!')