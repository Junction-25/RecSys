"""
ColBERT-based Property Encoder for token-level interaction-aware embeddings.
Provides more expressiveness than standard sentence embeddings.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ColBERTEncoder:
    """
    ColBERT encoder for generating token-level interaction-aware embeddings.
    More expressive than standard sentence embeddings for nuanced matches.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize ColBERT encoder with specified model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def encode_text(self, text: str, max_length: int = 512, return_tokens: bool = False) -> np.ndarray:
        """
        Encode text using ColBERT-style token-level embeddings.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            return_tokens: If True, return all token embeddings; if False, return mean-pooled
            
        Returns:
            Token-level embeddings as numpy array (either [seq_len, 384] or [384])
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get token embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                
                # Apply ColBERT-style normalization
                token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
                
                if return_tokens:
                    # Return all token embeddings for true ColBERT
                    attention_mask = inputs['attention_mask']
                    # Remove padding tokens
                    valid_tokens = attention_mask.bool()
                    # Get only valid token embeddings
                    valid_embeddings = token_embeddings[valid_tokens]
                    return valid_embeddings.cpu().numpy()
                else:
                    # Mean pooling for single embedding (current approach)
                    attention_mask = inputs['attention_mask']
                    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.sum(attention_mask, dim=1, keepdim=True)
                    mean_pooled = summed / counts
                    return mean_pooled.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error encoding text with ColBERT: {e}")
            if return_tokens:
                return np.zeros((1, 384))  # Return single token embedding on error
            else:
                return np.zeros(384)  # Return zero vector on error
    
    def maxsim_similarity(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """
        Calculate MaxSim similarity as used in ColBERT.
        
        Args:
            query_tokens: Query token embeddings [query_len, 384]
            doc_tokens: Document token embeddings [doc_len, 384]
            
        Returns:
            MaxSim similarity score
        """
        try:
            # Calculate similarity matrix [query_len, doc_len]
            similarities = np.dot(query_tokens, doc_tokens.T)
            
            # For each query token, find max similarity with any doc token
            max_similarities = np.max(similarities, axis=1)
            
            # Average the max similarities
            maxsim_score = np.mean(max_similarities)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, (maxsim_score + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating MaxSim similarity: {e}")
            return 0.0
    
    def encode_property_description(self, property_obj, return_tokens: bool = False) -> np.ndarray:
        """
        Encode property description using ColBERT.
        
        Args:
            property_obj: Property object with description
            return_tokens: If True, return token embeddings; if False, return mean-pooled
            
        Returns:
            Property description embedding (single vector or token matrix)
        """
        # Construct rich property description
        description_parts = []
        
        if hasattr(property_obj, 'title') and property_obj.title:
            description_parts.append(f"Property: {property_obj.title}")
            
        if hasattr(property_obj, 'description') and property_obj.description:
            description_parts.append(property_obj.description)
            
        if hasattr(property_obj, 'property_type') and property_obj.property_type:
            description_parts.append(f"Type: {property_obj.property_type}")
            
        if hasattr(property_obj, 'city') and property_obj.city:
            description_parts.append(f"Location: {property_obj.city}")
            
        if hasattr(property_obj, 'area') and property_obj.area:
            description_parts.append(f"Area: {property_obj.area} square meters")
            
        if hasattr(property_obj, 'rooms') and property_obj.rooms:
            description_parts.append(f"Rooms: {property_obj.rooms}")
            
        # Add features
        features = []
        if hasattr(property_obj, 'has_parking') and property_obj.has_parking:
            features.append("parking")
        if hasattr(property_obj, 'has_garden') and property_obj.has_garden:
            features.append("garden")
        if hasattr(property_obj, 'has_balcony') and property_obj.has_balcony:
            features.append("balcony")
        if hasattr(property_obj, 'has_elevator') and property_obj.has_elevator:
            features.append("elevator")
        if hasattr(property_obj, 'is_furnished') and property_obj.is_furnished:
            features.append("furnished")
            
        if features:
            description_parts.append(f"Features: {', '.join(features)}")
        
        full_description = ". ".join(description_parts)
        return self.encode_text(full_description, return_tokens=return_tokens)
    
    def encode_buyer_preferences(self, contact_obj) -> np.ndarray:
        """
        Encode buyer preferences using ColBERT.
        
        Args:
            contact_obj: Contact object with preferences
            
        Returns:
            Buyer preferences embedding
        """
        # Construct buyer preference description
        preference_parts = []
        
        if hasattr(contact_obj, 'first_name') and contact_obj.first_name:
            preference_parts.append(f"Buyer: {contact_obj.first_name}")
            
        if hasattr(contact_obj, 'property_type') and contact_obj.property_type:
            preference_parts.append(f"Seeking: {contact_obj.property_type}")
            
        if hasattr(contact_obj, 'budget_min') and hasattr(contact_obj, 'budget_max'):
            if contact_obj.budget_min and contact_obj.budget_max:
                preference_parts.append(f"Budget: {contact_obj.budget_min} to {contact_obj.budget_max}")
                
        if hasattr(contact_obj, 'preferred_locations') and contact_obj.preferred_locations:
            if isinstance(contact_obj.preferred_locations, (list, tuple)):
                locations = ", ".join(str(loc) for loc in contact_obj.preferred_locations if loc)
            else:
                locations = str(contact_obj.preferred_locations)
            if locations:
                preference_parts.append(f"Preferred locations: {locations}")
                
        if hasattr(contact_obj, 'desired_area_min') and hasattr(contact_obj, 'desired_area_max'):
            if contact_obj.desired_area_min and contact_obj.desired_area_max:
                preference_parts.append(f"Desired area: {contact_obj.desired_area_min} to {contact_obj.desired_area_max} square meters")
                
        if hasattr(contact_obj, 'rooms_min') and hasattr(contact_obj, 'rooms_max'):
            if contact_obj.rooms_min and contact_obj.rooms_max:
                preference_parts.append(f"Rooms: {contact_obj.rooms_min} to {contact_obj.rooms_max}")
        
        # Add feature preferences
        features = []
        if hasattr(contact_obj, 'prefers_parking') and contact_obj.prefers_parking:
            features.append("parking")
        if hasattr(contact_obj, 'prefers_garden') and contact_obj.prefers_garden:
            features.append("garden")
        if hasattr(contact_obj, 'prefers_balcony') and contact_obj.prefers_balcony:
            features.append("balcony")
        if hasattr(contact_obj, 'prefers_elevator') and contact_obj.prefers_elevator:
            features.append("elevator")
        if hasattr(contact_obj, 'prefers_furnished') and contact_obj.prefers_furnished:
            features.append("furnished")
            
        if features:
            preference_parts.append(f"Prefers: {', '.join(features)}")
        
        full_preferences = ". ".join(preference_parts)
        return self.encode_text(full_preferences)
    
    def calculate_maxsim_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """
        Calculate MaxSim similarity as used in ColBERT.
        
        Args:
            query_embedding: Query (buyer) embedding
            doc_embedding: Document (property) embedding
            
        Returns:
            MaxSim similarity score
        """
        try:
            # Normalize embeddings
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)
            
            # Calculate cosine similarity (simplified MaxSim for single vectors)
            similarity = np.dot(query_norm, doc_norm)
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating MaxSim similarity: {e}")
            return 0.0