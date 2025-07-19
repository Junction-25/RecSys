#!/usr/bin/env python
"""
Test script for the Django Smart Real Estate AI API.
"""
import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test data
TEST_PROPERTY = {
    "external_id": "TEST-PROP-001",
    "title": "Modern Apartment in Hydra",
    "description": "Beautiful modern apartment with stunning views",
    "address": "123 Hydra Street",
    "city": "Algiers",
    "district": "Hydra",
    "latitude": "36.7538",
    "longitude": "3.0588",
    "price": "25000000",
    "area": "120",
    "property_type": "apartment",
    "rooms": 3,
    "bathrooms": 2,
    "has_parking": True,
    "has_balcony": True,
    "has_elevator": True,
    "is_furnished": False,
    "status": "available",
    "listing_type": "sale",
    "agent_name": "Test Agent"
}

TEST_CONTACT = {
    "external_id": "TEST-CONTACT-001",
    "name": "Ahmed Benali",
    "email": "ahmed.benali@example.com",
    "phone": "+213-661234567",
    "preferred_locations": ["Algiers", "Hydra"],
    "budget_min": "20000000",
    "budget_max": "30000000",
    "desired_area_min": "100",
    "desired_area_max": "150",
    "property_type": "apartment",
    "rooms_min": 2,
    "rooms_max": 4,
    "prefers_parking": True,
    "prefers_balcony": True,
    "priority": "high",
    "status": "active"
}


class APITester:
    """Test class for API endpoints."""
    
    def __init__(self):
        self.session = requests.Session()
        self.property_id = None
        self.contact_id = None
        
    def print_header(self, title):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    def print_result(self, endpoint, response):
        """Print test result."""
        status_icon = "✅" if response.status_code < 400 else "❌"
        print(f"{status_icon} {endpoint}: {response.status_code}")
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error: {response.text}")
        else:
            try:
                data = response.json()
                if isinstance(data, dict) and 'count' in data:
                    print(f"   Count: {data['count']}")
                elif isinstance(data, list):
                    print(f"   Items: {len(data)}")
                elif isinstance(data, dict) and 'id' in data:
                    print(f"   ID: {data['id']}")
            except:
                pass
    
    def test_health_check(self):
        """Test health check endpoint."""
        self.print_header("Health Check")
        
        response = self.session.get(f"{BASE_URL}/health/")
        self.print_result("GET /health/", response)
        
        response = self.session.get(f"{BASE_URL}/health/status/")
        self.print_result("GET /health/status/", response)
    
    def test_properties_api(self):
        """Test properties API endpoints."""
        self.print_header("Properties API")
        
        # List properties
        response = self.session.get(f"{API_BASE}/properties/")
        self.print_result("GET /properties/", response)
        
        # Create property
        response = self.session.post(f"{API_BASE}/properties/", json=TEST_PROPERTY)
        self.print_result("POST /properties/", response)
        
        if response.status_code == 201:
            data = response.json()
            self.property_id = data['id']
            print(f"   Created property ID: {self.property_id}")
        
        # Get property details
        if self.property_id:
            response = self.session.get(f"{API_BASE}/properties/{self.property_id}/")
            self.print_result(f"GET /properties/{self.property_id}/", response)
        
        # Search properties
        response = self.session.get(f"{API_BASE}/properties/search/?city=Algiers")
        self.print_result("GET /properties/search/", response)
        
        # Property statistics
        response = self.session.get(f"{API_BASE}/properties/statistics/")
        self.print_result("GET /properties/statistics/", response)
    
    def test_contacts_api(self):
        """Test contacts API endpoints."""
        self.print_header("Contacts API")
        
        # List contacts
        response = self.session.get(f"{API_BASE}/contacts/")
        self.print_result("GET /contacts/", response)
        
        # Create contact
        response = self.session.post(f"{API_BASE}/contacts/", json=TEST_CONTACT)
        self.print_result("POST /contacts/", response)
        
        if response.status_code == 201:
            data = response.json()
            self.contact_id = data['id']
            print(f"   Created contact ID: {self.contact_id}")
        
        # Get contact details
        if self.contact_id:
            response = self.session.get(f"{API_BASE}/contacts/{self.contact_id}/")
            self.print_result(f"GET /contacts/{self.contact_id}/", response)
        
        # Contact statistics
        response = self.session.get(f"{API_BASE}/contacts/statistics/")
        self.print_result("GET /contacts/statistics/", response)
    
    def test_recommendations_api(self):
        """Test recommendations API endpoints."""
        self.print_header("Recommendations API")
        
        if not self.property_id or not self.contact_id:
            print("⚠️  Skipping recommendations tests - missing property or contact ID")
            return
        
        # Property recommendations
        response = self.session.get(f"{API_BASE}/recommendations/property/{self.property_id}/")
        self.print_result(f"GET /recommendations/property/{self.property_id}/", response)
        
        # Contact recommendations
        response = self.session.get(f"{API_BASE}/recommendations/contact/{self.contact_id}/")
        self.print_result(f"GET /recommendations/contact/{self.contact_id}/", response)
        
        # Comprehensive analysis
        response = self.session.get(f"{API_BASE}/recommendations/comprehensive-analysis/")
        self.print_result("GET /recommendations/comprehensive-analysis/", response)
        
        # Metrics
        response = self.session.get(f"{API_BASE}/recommendations/metrics/")
        self.print_result("GET /recommendations/metrics/", response)
        
        # Vector-based recommendations (new!)
        if self.property_id and self.contact_id:
            # Vector property recommendations
            response = self.session.get(f"{API_BASE}/recommendations/vector/property/{self.property_id}/?limit=5")
            self.print_result("GET /recommendations/vector/property/{id}/", response)
            
            # Vector contact recommendations  
            response = self.session.get(f"{API_BASE}/recommendations/vector/contact/{self.contact_id}/?limit=5")
            self.print_result("GET /recommendations/vector/contact/{id}/", response)
            
            # Compare recommendation methods
            response = self.session.get(f"{API_BASE}/recommendations/compare-methods/?property_id={self.property_id}&contact_id={self.contact_id}")
            self.print_result("GET /recommendations/compare-methods/", response)
        
        # Generate embeddings (if needed)
        response = self.session.post(f"{API_BASE}/recommendations/generate-embeddings/")
        self.print_result("POST /recommendations/generate-embeddings/", response)
    
    def test_ai_agents_api(self):
        """Test AI agents API endpoints."""
        self.print_header("AI Agents API")
        
        # Capabilities
        response = self.session.get(f"{API_BASE}/ai-agents/capabilities/")
        self.print_result("GET /ai-agents/capabilities/", response)
        
        if not self.property_id or not self.contact_id:
            print("⚠️  Skipping AI analysis tests - missing property or contact ID")
            return
        
        # Property analysis
        response = self.session.get(f"{API_BASE}/ai-agents/analyze-property/{self.property_id}/")
        self.print_result(f"GET /ai-agents/analyze-property/{self.property_id}/", response)
        
        # Contact analysis
        response = self.session.get(f"{API_BASE}/ai-agents/analyze-contact/{self.contact_id}/")
        self.print_result(f"GET /ai-agents/analyze-contact/{self.contact_id}/", response)
        
        # Generate description
        response = self.session.get(f"{API_BASE}/ai-agents/generate-description/{self.property_id}/")
        self.print_result(f"GET /ai-agents/generate-description/{self.property_id}/", response)
        
        # Match explanation
        match_data = {
            "property_id": self.property_id,
            "contact_id": self.contact_id,
            "match_score": 0.85
        }
        response = self.session.post(f"{API_BASE}/ai-agents/explain-match/", json=match_data)
        self.print_result("POST /ai-agents/explain-match/", response)
    
    def test_analytics_api(self):
        """Test analytics API endpoints."""
        self.print_header("Analytics API")
        
        # Dashboard
        response = self.session.get(f"{API_BASE}/analytics/dashboard/")
        self.print_result("GET /analytics/dashboard/", response)
        
        # Market trends
        response = self.session.get(f"{API_BASE}/analytics/market-trends/")
        self.print_result("GET /analytics/market-trends/", response)
        
        # Location insights
        response = self.session.get(f"{API_BASE}/analytics/location-insights/")
        self.print_result("GET /analytics/location-insights/", response)
        
        # Performance metrics
        response = self.session.get(f"{API_BASE}/analytics/performance-metrics/")
        self.print_result("GET /analytics/performance-metrics/", response)
    
    def test_quotes_api(self):
        """Test quotes API endpoints."""
        self.print_header("Quotes API")
        
        # List quotes
        response = self.session.get(f"{API_BASE}/quotes/")
        self.print_result("GET /quotes/", response)
        
        # Quote statistics
        response = self.session.get(f"{API_BASE}/quotes/statistics/")
        self.print_result("GET /quotes/statistics/", response)
        
        if not self.property_id or not self.contact_id:
            print("⚠️  Skipping quote generation tests - missing property or contact ID")
            return
        
        # Generate property quote
        quote_data = {
            "property_id": self.property_id,
            "contact_id": self.contact_id,
            "additional_fees": {
                "agency_fee": "750000",
                "legal_fees": "50000"
            },
            "notes": "Test quote generation"
        }
        response = self.session.post(f"{API_BASE}/quotes/generate-property-quote/", json=quote_data)
        self.print_result("POST /quotes/generate-property-quote/", response)
    
    def test_api_documentation(self):
        """Test API documentation endpoints."""
        self.print_header("API Documentation")
        
        # OpenAPI schema
        response = self.session.get(f"{API_BASE}/../schema/")
        self.print_result("GET /api/schema/", response)
        
        # Swagger UI (HTML response)
        response = self.session.get(f"{API_BASE}/../docs/")
        self.print_result("GET /api/docs/", response)
    
    def run_all_tests(self):
        """Run all API tests."""
        print(f"🚀 Starting Django API Tests")
        print(f"Base URL: {BASE_URL}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            self.test_health_check()
            self.test_properties_api()
            self.test_contacts_api()
            self.test_recommendations_api()
            self.test_ai_agents_api()
            self.test_analytics_api()
            self.test_quotes_api()
            self.test_api_documentation()
            
        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f" Test Completed in {duration:.2f} seconds")
        print(f"{'='*60}")


if __name__ == "__main__":
    tester = APITester()
    tester.run_all_tests()