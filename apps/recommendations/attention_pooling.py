"""
Attention Pooling for location embeddings.
Assigns higher weights to more relevant locations based on context.
Uses both geographic coordinates and semantic text understanding.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import math
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class AttentionPooling:
    """
    Advanced attention pooling for location embeddings.
    Combines geographic coordinates and semantic text understanding.
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        """
        Initialize attention pooling with geographic + semantic encoding.
        
        Args:
            embedding_dim: Dimension of final location embeddings
            hidden_dim: Hidden dimension for attention computation
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize sentence transformer for semantic location encoding
        try:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_embedding_dim = 384  # all-MiniLM-L6-v2 output dimension
            logger.info("Initialized SentenceTransformer for location text encoding")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}. Using fallback.")
            self.text_encoder = None
            self.text_embedding_dim = 64
        
        # Geographic coordinate encoder
        self.geo_embedding_dim = 32
        self.geo_encoder = nn.Sequential(
            nn.Linear(2, 16),  # lat, lon -> 16
            nn.ReLU(),
            nn.Linear(16, self.geo_embedding_dim),  # 16 -> 32
            nn.Tanh()  # Normalize geographic embeddings
        )
        
        # Fusion network to combine geographic + semantic embeddings
        combined_dim = self.text_embedding_dim + self.geo_embedding_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Attention network for pooling multiple locations
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Cache for encoded locations
        self.location_cache = {}
        
        logger.info(f"Initialized AttentionPooling with geo_dim={self.geo_embedding_dim}, "
                   f"text_dim={self.text_embedding_dim}, final_dim={embedding_dim}")
        
    def encode_text_semantic(self, text: str) -> np.ndarray:
        """
        Encode location text using semantic understanding.
        
        Args:
            text: Location text (e.g., "Around Algiers-Center")
            
        Returns:
            Semantic text embedding
        """
        try:
            if self.text_encoder:
                # Use SentenceTransformer for semantic encoding
                embedding = self.text_encoder.encode(text, convert_to_numpy=True)
                return embedding
            else:
                # Fallback: simple character-based encoding
                char_values = [ord(c) for c in text.lower()[:self.text_embedding_dim]]
                char_values.extend([0] * (self.text_embedding_dim - len(char_values)))
                return np.array(char_values[:self.text_embedding_dim], dtype=np.float32) / 255.0
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            return np.zeros(self.text_embedding_dim, dtype=np.float32)
    
    def encode_geographic_coordinates(self, lat: float, lon: float) -> np.ndarray:
        """
        Encode geographic coordinates into spatial embedding.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Geographic coordinate embedding
        """
        try:
            # Normalize coordinates to [-1, 1] range
            normalized_lat = lat / 90.0  # Latitude range: -90 to 90
            normalized_lon = lon / 180.0  # Longitude range: -180 to 180
            
            coords = torch.tensor([normalized_lat, normalized_lon], dtype=torch.float32)
            
            with torch.no_grad():
                geo_embedding = self.geo_encoder(coords).numpy()
            
            return geo_embedding
        except Exception as e:
            logger.error(f"Error encoding coordinates ({lat}, {lon}): {e}")
            return np.zeros(self.geo_embedding_dim, dtype=np.float32)
    
    def encode_location_dict(self, location_dict: Dict[str, Any]) -> np.ndarray:
        """
        Encode a location dictionary with both coordinates and text.
        
        Args:
            location_dict: Dictionary with 'lat', 'lon', and 'name' keys
            
        Returns:
            Combined geographic + semantic location embedding
        """
        try:
            # Extract components
            lat = float(location_dict.get('lat', 0))
            lon = float(location_dict.get('lon', 0))
            name = str(location_dict.get('name', ''))
            
            # Create cache key
            cache_key = f"{lat}_{lon}_{name}"
            if cache_key in self.location_cache:
                return self.location_cache[cache_key]
            
            # Encode geographic coordinates
            geo_embedding = self.encode_geographic_coordinates(lat, lon)
            
            # Encode semantic text
            text_embedding = self.encode_text_semantic(name)
            
            # Combine embeddings
            combined_embedding = np.concatenate([text_embedding, geo_embedding])
            
            # Pass through fusion network
            with torch.no_grad():
                combined_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
                final_embedding = self.fusion_network(combined_tensor).numpy()
            
            # Cache result
            self.location_cache[cache_key] = final_embedding
            
            return final_embedding
            
        except Exception as e:
            logger.error(f"Error encoding location dict {location_dict}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def encode_location(self, location: Any) -> np.ndarray:
        """
        Encode a location (string or dict) into embedding.
        
        Args:
            location: Location string or dictionary
            
        Returns:
            Location embedding
        """
        try:
            if isinstance(location, dict):
                return self.encode_location_dict(location)
            elif isinstance(location, str):
                # For string locations, use semantic encoding only
                text_embedding = self.encode_text_semantic(location)
                # Pad with zeros for geographic part
                geo_padding = np.zeros(self.geo_embedding_dim, dtype=np.float32)
                combined_embedding = np.concatenate([text_embedding, geo_padding])
                
                with torch.no_grad():
                    combined_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
                    final_embedding = self.fusion_network(combined_tensor).numpy()
                
                return final_embedding
            else:
                logger.warning(f"Unknown location type: {type(location)}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error encoding location {location}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def calculate_geographic_distance(self, loc1_dict: Dict, loc2_dict: Dict) -> float:
        """
        Calculate geographic distance between two locations using Haversine formula.
        
        Args:
            loc1_dict: First location dictionary
            loc2_dict: Second location dictionary
            
        Returns:
            Distance in kilometers
        """
        try:
            lat1, lon1 = float(loc1_dict.get('lat', 0)), float(loc1_dict.get('lon', 0))
            lat2, lon2 = float(loc2_dict.get('lat', 0)), float(loc2_dict.get('lon', 0))
            
            # Haversine formula
            R = 6371  # Earth's radius in kilometers
            
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            
            a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2) * math.sin(dlon/2))
            
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating geographic distance: {e}")
            return float('inf')
    
    def encode_location_set(self, locations: List[str], context_location: Optional[str] = None) -> np.ndarray:
        """
        Encode a set of locations with attention pooling.
        
        Args:
            locations: List of location strings
            context_location: Context location for attention weighting
            
        Returns:
            Attention-pooled location embedding
        """
        try:
            if not locations:
                return np.zeros(self.embedding_dim)
            
            # Get embeddings for all locations
            location_embeddings = []
            for loc in locations:
                if loc and loc.strip():
                    emb = self.encode_location(loc.strip())
                    location_embeddings.append(emb)
            
            if not location_embeddings:
                return np.zeros(self.embedding_dim)
            
            # Convert to tensor
            embeddings_tensor = torch.tensor(np.array(location_embeddings), dtype=torch.float32)
            
            if context_location:
                # Use attention with context
                context_emb = torch.tensor(self.encode_location(context_location), dtype=torch.float32)
                
                # Calculate attention weights based on similarity to context
                similarities = torch.cosine_similarity(embeddings_tensor, context_emb.unsqueeze(0), dim=1)
                attention_weights = torch.softmax(similarities, dim=0)
                
                # Apply attention weights
                pooled_embedding = torch.sum(embeddings_tensor * attention_weights.unsqueeze(1), dim=0)
            else:
                # Use learned attention without context
                with torch.no_grad():
                    attention_weights = self.attention_net(embeddings_tensor).squeeze()
                    if attention_weights.dim() == 0:  # Single location
                        attention_weights = attention_weights.unsqueeze(0)
                    pooled_embedding = torch.sum(embeddings_tensor * attention_weights.unsqueeze(1), dim=0)
            
            return pooled_embedding.numpy()
            
        except Exception as e:
            logger.error(f"Error in attention pooling for locations: {e}")
            return np.zeros(self.embedding_dim)
    
    def calculate_location_similarity(self, loc1: str, loc2: str) -> float:
        """
        Calculate similarity between two locations.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Similarity score (0-1)
        """
        try:
            emb1 = self.encode_location(loc1)
            emb2 = self.encode_location(loc2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating location similarity: {e}")
            return 0.0
    
    def encode_property_location(self, property_obj) -> np.ndarray:
        """
        Encode property location with attention pooling.
        
        Args:
            property_obj: Property object
            
        Returns:
            Property location embedding
        """
        locations = []
        
        if hasattr(property_obj, 'city') and property_obj.city:
            locations.append(property_obj.city)
        if hasattr(property_obj, 'district') and property_obj.district:
            locations.append(property_obj.district)
        if hasattr(property_obj, 'neighborhood') and property_obj.neighborhood:
            locations.append(property_obj.neighborhood)
        if hasattr(property_obj, 'address') and property_obj.address:
            locations.append(property_obj.address)
        
        return self.encode_location_set(locations)
    
    def encode_buyer_preferred_locations(self, contact_obj, property_location: Optional[str] = None) -> np.ndarray:
        """
        Encode buyer preferred locations with attention pooling using geographic + semantic encoding.
        
        Args:
            contact_obj: Contact object
            property_location: Property location for context-aware attention
            
        Returns:
            Buyer preferred locations embedding
        """
        try:
            if not hasattr(contact_obj, 'preferred_locations') or not contact_obj.preferred_locations:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            location_embeddings = []
            
            if isinstance(contact_obj.preferred_locations, (list, tuple)):
                for loc in contact_obj.preferred_locations:
                    if isinstance(loc, dict):
                        # Use the new geographic + semantic encoding for location dictionaries
                        embedding = self.encode_location_dict(loc)
                        location_embeddings.append(embedding)
                    elif loc:
                        # Fallback for string locations
                        embedding = self.encode_location(str(loc))
                        location_embeddings.append(embedding)
            elif isinstance(contact_obj.preferred_locations, dict):
                # Single location dictionary
                embedding = self.encode_location_dict(contact_obj.preferred_locations)
                location_embeddings.append(embedding)
            else:
                # Fallback for other formats
                embedding = self.encode_location(str(contact_obj.preferred_locations))
                location_embeddings.append(embedding)
            
            if not location_embeddings:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Apply attention pooling
            embeddings_tensor = torch.tensor(np.array(location_embeddings), dtype=torch.float32)
            
            if property_location and len(location_embeddings) > 1:
                # Use context-aware attention if we have multiple locations
                context_emb = torch.tensor(self.encode_location(property_location), dtype=torch.float32)
                similarities = torch.cosine_similarity(embeddings_tensor, context_emb.unsqueeze(0), dim=1)
                attention_weights = torch.softmax(similarities, dim=0)
                pooled_embedding = torch.sum(embeddings_tensor * attention_weights.unsqueeze(1), dim=0)
            else:
                # Simple average for single location or no context
                pooled_embedding = torch.mean(embeddings_tensor, dim=0)
            
            return pooled_embedding.numpy()
            
        except Exception as e:
            logger.error(f"Error encoding buyer preferred locations: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def calculate_location_match_score(self, property_obj, contact_obj) -> float:
        """
        Calculate location match score using attention-pooled embeddings.
        
        Args:
            property_obj: Property object
            contact_obj: Contact object
            
        Returns:
            Location match score (0-1)
        """
        try:
            # Get property location embedding
            prop_location_emb = self.encode_property_location(property_obj)
            
            # Get property location string for context
            prop_location_str = None
            if hasattr(property_obj, 'city') and property_obj.city:
                prop_location_str = property_obj.city
            
            # Get buyer preferred locations embedding with property context
            buyer_locations_emb = self.encode_buyer_preferred_locations(contact_obj, prop_location_str)
            
            # Calculate similarity
            if np.allclose(prop_location_emb, 0) or np.allclose(buyer_locations_emb, 0):
                return 0.0
            
            similarity = np.dot(prop_location_emb, buyer_locations_emb) / (
                np.linalg.norm(prop_location_emb) * np.linalg.norm(buyer_locations_emb) + 1e-8
            )
            
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating location match score: {e}")
            return 0.0
    
    def train_attention_weights(self, location_pairs: List[tuple], similarity_scores: List[float]) -> None:
        """
        Train attention weights based on location similarity data.
        
        Args:
            location_pairs: List of (location1, location2) tuples
            similarity_scores: Corresponding similarity scores
        """
        try:
            # Simple training loop for attention weights
            optimizer = torch.optim.Adam(self.attention_net.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(100):  # Simple training
                total_loss = 0
                for (loc1, loc2), target_score in zip(location_pairs, similarity_scores):
                    # Get embeddings
                    emb1 = torch.tensor(self.encode_location(loc1), dtype=torch.float32)
                    emb2 = torch.tensor(self.encode_location(loc2), dtype=torch.float32)
                    
                    # Calculate attention-weighted similarity
                    combined_emb = torch.stack([emb1, emb2])
                    attention_weights = self.attention_net(combined_emb).squeeze()
                    weighted_emb = torch.sum(combined_emb * attention_weights.unsqueeze(1), dim=0)
                    
                    # Calculate predicted similarity
                    pred_similarity = torch.cosine_similarity(emb1, emb2, dim=0)
                    target = torch.tensor(target_score, dtype=torch.float32)
                    
                    # Calculate loss
                    loss = criterion(pred_similarity, target)
                    total_loss += loss.item()
                    
                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Attention training epoch {epoch}, loss: {total_loss:.4f}")
                    
        except Exception as e:
            logger.error(f"Error training attention weights: {e}")
    
    def save_state(self, filepath: str) -> None:
        """Save attention pooling state."""
        try:
            state = {
                'attention_net': self.attention_net.state_dict(),
                'geo_encoder': self.geo_encoder.state_dict(),
                'fusion_network': self.fusion_network.state_dict(),
                'location_cache': self.location_cache,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'geo_embedding_dim': self.geo_embedding_dim,
                'text_embedding_dim': self.text_embedding_dim
            }
            torch.save(state, filepath)
            logger.info(f"Saved attention pooling state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving attention pooling state: {e}")
    
    def load_state(self, filepath: str) -> None:
        """Load attention pooling state."""
        try:
            state = torch.load(filepath)
            self.attention_net.load_state_dict(state['attention_net'])
            self.geo_encoder.load_state_dict(state['geo_encoder'])
            self.fusion_network.load_state_dict(state['fusion_network'])
            self.location_cache = state.get('location_cache', {})
            self.embedding_dim = state['embedding_dim']
            self.hidden_dim = state['hidden_dim']
            self.geo_embedding_dim = state.get('geo_embedding_dim', 32)
            self.text_embedding_dim = state.get('text_embedding_dim', 384)
            logger.info(f"Loaded attention pooling state from {filepath}")
        except Exception as e:
            logger.error(f"Error loading attention pooling state: {e}")