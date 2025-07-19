"""
AI agent services using Gemini API.
"""
import json
import google.generativeai as genai
from django.conf import settings
from django.core.cache import cache

# Configure the Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiAgent:
    """
    AI agent powered by Google's Gemini API.
    Provides intelligent property and contact analysis.
    """
    
    def __init__(self):
        """Initialize the Gemini agent."""
        self.model = genai.GenerativeModel('gemini-pro')
    
    def analyze_property(self, property_data):
        """
        Analyze a property and provide insights.
        
        Args:
            property_data: Dictionary containing property details
            
        Returns:
            dict: Analysis results
        """
        cache_key = f"property_analysis_{property_data.get('id', '')}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        prompt = f"""
        As a real estate expert, analyze this property and provide insights:
        
        Property Details:
        - Title: {property_data.get('title')}
        - Type: {property_data.get('property_type')}
        - Location: {property_data.get('city')}, {property_data.get('district')}
        - Price: {property_data.get('price')}
        - Area: {property_data.get('area')}m²
        - Rooms: {property_data.get('rooms')}
        - Features: {', '.join(property_data.get('features_list', []))}
        
        Please provide:
        1. Market position assessment
        2. Estimated fair value range
        3. Key selling points
        4. Potential buyer profile
        5. Negotiation strategy
        
        Format your response as JSON with these exact keys: 
        market_position, value_range, selling_points, buyer_profile, negotiation_strategy
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            result = self._extract_json_from_response(response.text)
            
            # Cache the result for 24 hours
            cache.set(cache_key, result, 60 * 60 * 24)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to analyze property with AI"
            }
    
    def generate_property_description(self, property_data):
        """
        Generate an engaging property description.
        
        Args:
            property_data: Dictionary containing property details
            
        Returns:
            str: Generated description
        """
        cache_key = f"property_description_{property_data.get('id', '')}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        prompt = f"""
        Create an engaging and professional real estate listing description for this property:
        
        Property Details:
        - Title: {property_data.get('title')}
        - Type: {property_data.get('property_type')}
        - Location: {property_data.get('city')}, {property_data.get('district')}
        - Price: {property_data.get('price')}
        - Area: {property_data.get('area')}m²
        - Rooms: {property_data.get('rooms')}
        - Features: {', '.join(property_data.get('features_list', []))}
        
        The description should be compelling, highlight key features, and be approximately 150-200 words.
        Do not include the price in the description.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Cache the result for 24 hours
            cache.set(cache_key, result, 60 * 60 * 24)
            
            return result
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
    def analyze_contact_preferences(self, contact_data):
        """
        Analyze a contact's preferences and provide insights.
        
        Args:
            contact_data: Dictionary containing contact details
            
        Returns:
            dict: Analysis results
        """
        cache_key = f"contact_analysis_{contact_data.get('id', '')}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        prompt = f"""
        As a real estate expert, analyze this potential buyer's preferences:
        
        Contact Details:
        - Name: {contact_data.get('name')}
        - Preferred Locations: {', '.join(contact_data.get('preferred_locations', []))}
        - Budget Range: {contact_data.get('budget_min')} - {contact_data.get('budget_max')}
        - Desired Area: {contact_data.get('desired_area_min')} - {contact_data.get('desired_area_max')}m²
        - Property Type: {contact_data.get('property_type')}
        - Room Preference: {contact_data.get('rooms_min', 'Any')} - {contact_data.get('rooms_max', 'Any')}
        - Feature Preferences: {', '.join(contact_data.get('preferences_list', []))}
        - Priority: {contact_data.get('priority')}
        
        Please provide:
        1. Buyer motivation analysis
        2. Ideal property profile
        3. Alternative suggestions (locations, property types)
        4. Engagement strategy
        5. Potential objections and how to address them
        
        Format your response as JSON with these exact keys: 
        motivation, ideal_property, alternatives, engagement_strategy, objection_handling
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            result = self._extract_json_from_response(response.text)
            
            # Cache the result for 24 hours
            cache.set(cache_key, result, 60 * 60 * 24)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to analyze contact preferences with AI"
            }
    
    def generate_match_explanation(self, property_data, contact_data, match_score):
        """
        Generate a detailed explanation of why a property matches a contact.
        
        Args:
            property_data: Dictionary containing property details
            contact_data: Dictionary containing contact details
            match_score: The calculated match score
            
        Returns:
            str: Generated explanation
        """
        cache_key = f"match_explanation_{property_data.get('id', '')}_{contact_data.get('id', '')}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        prompt = f"""
        As a real estate expert, explain why this property is a good match for this buyer:
        
        Property Details:
        - Title: {property_data.get('title')}
        - Type: {property_data.get('property_type')}
        - Location: {property_data.get('city')}, {property_data.get('district')}
        - Price: {property_data.get('price')}
        - Area: {property_data.get('area')}m²
        - Rooms: {property_data.get('rooms')}
        - Features: {', '.join(property_data.get('features_list', []))}
        
        Buyer Preferences:
        - Preferred Locations: {', '.join(contact_data.get('preferred_locations', []))}
        - Budget Range: {contact_data.get('budget_min')} - {contact_data.get('budget_max')}
        - Desired Area: {contact_data.get('desired_area_min')} - {contact_data.get('desired_area_max')}m²
        - Property Type: {contact_data.get('property_type')}
        - Room Preference: {contact_data.get('rooms_min', 'Any')} - {contact_data.get('rooms_max', 'Any')}
        - Feature Preferences: {', '.join(contact_data.get('preferences_list', []))}
        
        Match Score: {match_score}
        
        Provide a detailed, personalized explanation (about 150 words) of why this property is a good match for this buyer.
        Focus on the strongest matching points and address any potential concerns.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Cache the result for 24 hours
            cache.set(cache_key, result, 60 * 60 * 24)
            
            return result
        except Exception as e:
            return f"Error generating match explanation: {str(e)}"
    
    def _extract_json_from_response(self, text):
        """Extract JSON from a text response."""
        try:
            # Try to parse the entire response as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the text
            try:
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # If all parsing fails, return a structured error
            return {
                "error": "Failed to parse JSON response",
                "raw_response": text
            }