"""
Management command to seed sample data for the Smart Real Estate AI platform.
"""
import uuid
import random
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.properties.models import Property
from apps.contacts.models import Contact


class Command(BaseCommand):
    """Command to seed sample data for testing and development."""
    
    help = 'Seeds the database with sample properties and contacts'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--properties',
            type=int,
            default=10,
            help='Number of properties to create'
        )
        parser.add_argument(
            '--contacts',
            type=int,
            default=20,
            help='Number of contacts to create'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        num_properties = options['properties']
        num_contacts = options['contacts']
        
        self.stdout.write(self.style.SUCCESS(f'Creating {num_properties} properties...'))
        self._create_properties(num_properties)
        
        self.stdout.write(self.style.SUCCESS(f'Creating {num_contacts} contacts...'))
        self._create_contacts(num_contacts)
        
        self.stdout.write(self.style.SUCCESS('Sample data seeded successfully!'))
    
    def _create_properties(self, count):
        """Create sample properties."""
        # Sample data for properties
        cities = ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Batna', 'Setif']
        districts = ['Downtown', 'Hydra', 'Kouba', 'Bab Ezzouar', 'Birkhadem', 'El Biar', 'Ain Naadja']
        property_types = [Property.APARTMENT, Property.VILLA, Property.OFFICE, Property.COMMERCIAL, Property.LAND]
        statuses = [Property.AVAILABLE, Property.PENDING, Property.SOLD, Property.RENTED]
        listing_types = [Property.SALE, Property.RENT]
        
        for i in range(count):
            city = random.choice(cities)
            district = random.choice(districts)
            property_type = random.choice(property_types)
            status = random.choice(statuses)
            listing_type = random.choice(listing_types)
            
            # Generate price based on property type and listing type
            if property_type == Property.APARTMENT:
                base_price = 25000000 if listing_type == Property.SALE else 80000
            elif property_type == Property.VILLA:
                base_price = 50000000 if listing_type == Property.SALE else 150000
            elif property_type == Property.OFFICE:
                base_price = 30000000 if listing_type == Property.SALE else 100000
            elif property_type == Property.COMMERCIAL:
                base_price = 40000000 if listing_type == Property.SALE else 120000
            else:  # LAND
                base_price = 20000000 if listing_type == Property.SALE else 0
            
            price = base_price * (0.8 + random.random() * 0.4)  # ±20% variation
            
            # Generate area based on property type
            if property_type == Property.APARTMENT:
                area = random.randint(60, 200)
            elif property_type == Property.VILLA:
                area = random.randint(150, 500)
            elif property_type == Property.OFFICE:
                area = random.randint(50, 300)
            elif property_type == Property.COMMERCIAL:
                area = random.randint(100, 1000)
            else:  # LAND
                area = random.randint(200, 2000)
            
            # Generate rooms based on property type
            if property_type in [Property.APARTMENT, Property.VILLA]:
                rooms = random.randint(1, 6)
                bathrooms = random.randint(1, 3)
            elif property_type == Property.OFFICE:
                rooms = random.randint(1, 10)
                bathrooms = random.randint(1, 4)
            else:
                rooms = None
                bathrooms = None
            
            # Create the property
            property_obj = Property.objects.create(
                external_id=f'PROP-{i+1:03d}',
                title=f'{property_type.title()} in {district}, {city}',
                description=f'Beautiful {property_type} located in {district}, {city}. '
                           f'Perfect for {"living" if property_type in [Property.APARTMENT, Property.VILLA] else "business"}.',
                address=f'{random.randint(1, 100)} {district} Street',
                city=city,
                district=district,
                latitude=36.7 + random.random() * 0.5,
                longitude=3.0 + random.random() * 0.5,
                price=price,
                area=area,
                property_type=property_type,
                rooms=rooms,
                bathrooms=bathrooms,
                has_parking=random.choice([True, False]),
                has_garden=random.choice([True, False]) if property_type in [Property.VILLA, Property.APARTMENT] else False,
                has_balcony=random.choice([True, False]) if property_type == Property.APARTMENT else False,
                has_elevator=random.choice([True, False]) if property_type in [Property.APARTMENT, Property.OFFICE, Property.COMMERCIAL] else False,
                is_furnished=random.choice([True, False]) if property_type in [Property.APARTMENT, Property.VILLA, Property.OFFICE] else False,
                has_air_conditioning=random.choice([True, False]),
                has_heating=random.choice([True, False]),
                status=status,
                listing_type=listing_type,
                agent_id=f'AGENT-{random.randint(1, 5):02d}',
                agent_name=f'Agent {random.randint(1, 5)}',
                agent_contact=f'+213-5{random.randint(10000000, 99999999)}'
            )
            
            self.stdout.write(f'Created property: {property_obj.title}')
    
    def _create_contacts(self, count):
        """Create sample contacts."""
        # Sample data for contacts
        first_names = ['Ahmed', 'Mohamed', 'Fatima', 'Amina', 'Karim', 'Leila', 'Youcef', 'Meriem', 'Sofiane', 'Samira']
        last_names = ['Benali', 'Boudiaf', 'Khelifi', 'Mansouri', 'Bouzid', 'Hamidi', 'Taleb', 'Messaoudi', 'Rahmani', 'Ziani']
        cities = ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Batna', 'Setif']
        districts = ['Downtown', 'Hydra', 'Kouba', 'Bab Ezzouar', 'Birkhadem', 'El Biar', 'Ain Naadja']
        property_types = [Contact.APARTMENT, Contact.VILLA, Contact.OFFICE, Contact.COMMERCIAL, Contact.LAND]
        priorities = [Contact.LOW, Contact.MEDIUM, Contact.HIGH]
        statuses = [Contact.ACTIVE, Contact.INACTIVE, Contact.CLOSED]
        
        for i in range(count):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            property_type = random.choice(property_types)
            priority = random.choice(priorities)
            status = random.choice(statuses)
            
            # Generate preferred locations
            num_locations = random.randint(1, 3)
            preferred_locations = []
            for _ in range(num_locations):
                if random.random() < 0.7:  # 70% chance to prefer a city
                    preferred_locations.append(random.choice(cities))
                else:  # 30% chance to prefer a specific district
                    preferred_locations.append(random.choice(districts))
            
            # Generate budget range based on property type
            if property_type == Contact.APARTMENT:
                budget_min = random.randint(15000000, 25000000)
                budget_max = budget_min + random.randint(5000000, 15000000)
            elif property_type == Contact.VILLA:
                budget_min = random.randint(30000000, 50000000)
                budget_max = budget_min + random.randint(10000000, 30000000)
            elif property_type == Contact.OFFICE:
                budget_min = random.randint(20000000, 30000000)
                budget_max = budget_min + random.randint(5000000, 20000000)
            elif property_type == Contact.COMMERCIAL:
                budget_min = random.randint(25000000, 40000000)
                budget_max = budget_min + random.randint(10000000, 25000000)
            else:  # LAND
                budget_min = random.randint(10000000, 20000000)
                budget_max = budget_min + random.randint(5000000, 15000000)
            
            # Generate area range based on property type
            if property_type == Contact.APARTMENT:
                area_min = random.randint(60, 120)
                area_max = area_min + random.randint(20, 80)
            elif property_type == Contact.VILLA:
                area_min = random.randint(150, 300)
                area_max = area_min + random.randint(50, 200)
            elif property_type == Contact.OFFICE:
                area_min = random.randint(50, 150)
                area_max = area_min + random.randint(50, 150)
            elif property_type == Contact.COMMERCIAL:
                area_min = random.randint(100, 300)
                area_max = area_min + random.randint(100, 700)
            else:  # LAND
                area_min = random.randint(200, 500)
                area_max = area_min + random.randint(300, 1500)
            
            # Generate room preferences based on property type
            if property_type in [Contact.APARTMENT, Contact.VILLA]:
                rooms_min = random.randint(1, 4)
                rooms_max = rooms_min + random.randint(0, 2)
            else:
                rooms_min = None
                rooms_max = None
            
            # Create the contact
            contact = Contact.objects.create(
                external_id=f'CONTACT-{i+1:03d}',
                name=f'{first_name} {last_name}',
                email=f'{first_name.lower()}.{last_name.lower()}@example.com',
                phone=f'+213-6{random.randint(10000000, 99999999)}',
                preferred_locations=preferred_locations,
                budget_min=budget_min,
                budget_max=budget_max,
                desired_area_min=area_min,
                desired_area_max=area_max,
                property_type=property_type,
                rooms_min=rooms_min,
                rooms_max=rooms_max,
                prefers_parking=random.choice([True, False]),
                prefers_garden=random.choice([True, False]) if property_type in [Contact.VILLA, Contact.APARTMENT] else False,
                prefers_balcony=random.choice([True, False]) if property_type == Contact.APARTMENT else False,
                prefers_elevator=random.choice([True, False]) if property_type in [Contact.APARTMENT, Contact.OFFICE, Contact.COMMERCIAL] else False,
                prefers_furnished=random.choice([True, False]) if property_type in [Contact.APARTMENT, Contact.VILLA, Contact.OFFICE] else False,
                priority=priority,
                status=status,
                notes=f'Contact is looking for a {property_type} in {", ".join(preferred_locations)}.'
            )
            
            self.stdout.write(f'Created contact: {contact.name}')