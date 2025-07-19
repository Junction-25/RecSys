"""
Property Encoder - Stage 1 of the TikTok-like SOTA Recommender System.
Transforms heterogeneous property data into dense embedding vectors.
"""
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, List
from .colbert_encoder import ColBERTEncoder
from .neural_binning import NeuralBinning
from .attention_pooling import AttentionPooling

logger = logging.getLogger(__name__)


class PropertyEncoder:
    """
    Property Encoder that transforms heterogeneous property data into dense embeddings.
    
    Architecture:
    - Text Features: ColBERT for token-level interaction-aware embeddings
    - Locations: Attention pooling for geographic relevance
    - Categorical Features: Co-occurrence embeddings (Word2Vec-style)
    - Numerical Features: Neural binning with learnable embeddings
    """
    
    def __init__(self, 
                 text_embedding_dim: int = 384,
                 location_embedding_dim: int = 128,
                 categorical_embedding_dim: int = 64,
                 numerical_embedding_dim: int = 64,
                 final_embedding_dim: int = 512):
        """
        Initialize Property Encoder.
        
        Args:
            text_embedding_dim: Dimension for text embeddings (ColBERT)
            location_embedding_dim: Dimension for location embeddings
            categorical_embedding_dim: Dimension for categorical embeddings
            numerical_embedding_dim: Dimension for numerical embeddings
            final_embedding_dim: Final property embedding dimension
        """
        self.text_embedding_dim = text_embedding_dim
        self.location_embedding_dim = location_embedding_dim
        self.categorical_embedding_dim = categorical_embedding_dim
        self.numerical_embedding_dim = numerical_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        
        # Initialize encoders
        self.colbert_encoder = ColBERTEncoder()
        self.neural_binning = NeuralBinning(embedding_dim=numerical_embedding_dim)
        self.attention_pooling = AttentionPooling(embedding_dim=location_embedding_dim)
        
        # Categorical embeddings (Word2Vec-style)
        self.categorical_encoders = {}
        self._init_categorical_encoders()
        
        # Fusion network to combine all embeddings
        total_input_dim = (text_embedding_dim + location_embedding_dim + 
                          categorical_embedding_dim * 2 + numerical_embedding_dim * 3)
        
        self.fusion_network = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, final_embedding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(final_embedding_dim * 2, final_embedding_dim),
            torch.nn.LayerNorm(final_embedding_dim)
        )
        
        logger.info(f"Initialized PropertyEncoder with final embedding dim: {final_embedding_dim}")
    
    def _init_categorical_encoders(self):
        """Initialize categorical feature encoders."""
        # Property type encoder
        self.categorical_encoders['property_type'] = torch.nn.Embedding(
            100, self.categorical_embedding_dim  # Support up to 100 property types
        )
        
        # Status encoder
        self.categorical_encoders['status'] = torch.nn.Embedding(
            10, self.categorical_embedding_dim  # Support up to 10 status types
        )
        
        # Category to ID mappings
        self.category_to_id = {
            'property_type': {},
            'status': {}
        }
        self.next_category_ids = {
            'property_type': 0,
            'status': 0
        }
    
    def _get_category_id(self, category_type: str, category_value: str) -> int:
        """Get or create category ID."""
        if category_value not in self.category_to_id[category_type]:
            self.category_to_id[category_type][category_value] = self.next_category_ids[category_type]
            self.next_category_ids[category_type] += 1
        return self.category_to_id[category_type][category_value]
    
    def encode_text_features(self, property_obj) -> np.ndarray:
        """
        Encode text features using ColBERT.
        
        Args:
            property_obj: Property object
            
        Returns:
            Text embedding vector
        """
        return self.colbert_encoder.encode_property_description(property_obj)
    
    def encode_location_features(self, property_obj) -> np.ndarray:
        """
        Encode location features using attention pooling.
        
        Args:
            property_obj: Property object
            
        Returns:
            Location embedding vector
        """
        return self.attention_pooling.encode_property_location(property_obj)
    
    def encode_categorical_features(self, property_obj) -> np.ndarray:
        """
        Encode categorical features using co-occurrence embeddings.
        
        Args:
            property_obj: Property object
            
        Returns:
            Categorical embeddings vector
        """
        embeddings = []
        
        # Property type embedding
        if hasattr(property_obj, 'property_type') and property_obj.property_type:
            prop_type_id = self._get_category_id('property_type', str(property_obj.property_type))
            with torch.no_grad():
                prop_type_emb = self.categorical_encoders['property_type'](
                    torch.tensor(prop_type_id)
                ).numpy()
            embeddings.append(prop_type_emb)
        else:
            embeddings.append(np.zeros(self.categorical_embedding_dim))
        
        # Status embedding
        if hasattr(property_obj, 'status') and property_obj.status:
            status_id = self._get_category_id('status', str(property_obj.status))
            with torch.no_grad():
                status_emb = self.categorical_encoders['status'](
                    torch.tensor(status_id)
                ).numpy()
            embeddings.append(status_emb)
        else:
            embeddings.append(np.zeros(self.categorical_embedding_dim))
        
        return np.concatenate(embeddings)
    
    def encode_numerical_features(self, property_obj) -> np.ndarray:
        """
        Encode numerical features using neural binning.
        
        Args:
            property_obj: Property object
            
        Returns:
            Numerical embeddings vector
        """
        return self.neural_binning.encode_property_numerical(property_obj)
    
    def encode_property(self, property_obj) -> np.ndarray:
        """
        Encode a complete property into a dense embedding vector.
        
        Args:
            property_obj: Property object
            
        Returns:
            Property embedding vector
        """
        try:
            # Encode each feature type
            text_emb = self.encode_text_features(property_obj)
            location_emb = self.encode_location_features(property_obj)
            categorical_emb = self.encode_categorical_features(property_obj)
            numerical_emb = self.encode_numerical_features(property_obj)
            
            # Concatenate all embeddings
            combined_emb = np.concatenate([
                text_emb,
                location_emb,
                categorical_emb,
                numerical_emb
            ])
            
            # Pass through fusion network
            with torch.no_grad():
                combined_tensor = torch.tensor(combined_emb, dtype=torch.float32)
                final_embedding = self.fusion_network(combined_tensor).numpy()
            
            return final_embedding
            
        except Exception as e:
            logger.error(f"Error encoding property: {e}")
            return np.zeros(self.final_embedding_dim)
    
    def batch_encode_properties(self, properties: List[Any]) -> np.ndarray:
        """
        Encode multiple properties in batch.
        
        Args:
            properties: List of property objects
            
        Returns:
            Array of property embeddings
        """
        embeddings = []
        for prop in properties:
            emb = self.encode_property(prop)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def fit_encoders(self, properties: List[Any]) -> None:
        """
        Fit all encoders on property data.
        
        Args:
            properties: List of property objects for fitting
        """
        logger.info("Fitting property encoders...")
        
        # Fit neural binning
        self.neural_binning.fit_property_features(properties)
        
        # Pre-populate categorical mappings
        for prop in properties:
            if hasattr(prop, 'property_type') and prop.property_type:
                self._get_category_id('property_type', str(prop.property_type))
            if hasattr(prop, 'status') and prop.status:
                self._get_category_id('status', str(prop.status))
        
        logger.info("Property encoders fitted successfully")
    
    def calculate_property_similarity(self, prop1_embedding: np.ndarray, prop2_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two property embeddings.
        
        Args:
            prop1_embedding: First property embedding
            prop2_embedding: Second property embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Cosine similarity
            similarity = np.dot(prop1_embedding, prop2_embedding) / (
                np.linalg.norm(prop1_embedding) * np.linalg.norm(prop2_embedding) + 1e-8
            )
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            logger.error(f"Error calculating property similarity: {e}")
            return 0.0
    
    def get_similar_properties(self, target_property, all_properties: List[Any], top_k: int = 10) -> List[Dict]:
        """
        Find similar properties to a target property.
        
        Args:
            target_property: Target property object
            all_properties: List of all property objects
            top_k: Number of similar properties to return
            
        Returns:
            List of similar properties with similarity scores
        """
        try:
            target_embedding = self.encode_property(target_property)
            similarities = []
            
            for prop in all_properties:
                if prop.id == target_property.id:
                    continue
                    
                prop_embedding = self.encode_property(prop)
                similarity = self.calculate_property_similarity(target_embedding, prop_embedding)
                
                similarities.append({
                    'property': prop,
                    'similarity_score': similarity,
                    'property_id': str(prop.id)
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {e}")
            return []
    
    def save_state(self, filepath: str) -> None:
        """Save encoder state."""
        try:
            state = {
                'fusion_network': self.fusion_network.state_dict(),
                'categorical_encoders': {
                    name: encoder.state_dict() 
                    for name, encoder in self.categorical_encoders.items()
                },
                'category_to_id': self.category_to_id,
                'next_category_ids': self.next_category_ids,
                'dimensions': {
                    'text_embedding_dim': self.text_embedding_dim,
                    'location_embedding_dim': self.location_embedding_dim,
                    'categorical_embedding_dim': self.categorical_embedding_dim,
                    'numerical_embedding_dim': self.numerical_embedding_dim,
                    'final_embedding_dim': self.final_embedding_dim
                }
            }
            torch.save(state, filepath)
            
            # Save sub-encoders
            self.neural_binning.save_state(f"{filepath}_binning")
            self.attention_pooling.save_state(f"{filepath}_attention")
            
            logger.info(f"Saved PropertyEncoder state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving PropertyEncoder state: {e}")
    
    def load_state(self, filepath: str) -> None:
        """Load encoder state."""
        try:
            state = torch.load(filepath)
            self.fusion_network.load_state_dict(state['fusion_network'])
            
            for name, state_dict in state['categorical_encoders'].items():
                self.categorical_encoders[name].load_state_dict(state_dict)
            
            self.category_to_id = state['category_to_id']
            self.next_category_ids = state['next_category_ids']
            
            # Load dimensions
            dims = state['dimensions']
            self.text_embedding_dim = dims['text_embedding_dim']
            self.location_embedding_dim = dims['location_embedding_dim']
            self.categorical_embedding_dim = dims['categorical_embedding_dim']
            self.numerical_embedding_dim = dims['numerical_embedding_dim']
            self.final_embedding_dim = dims['final_embedding_dim']
            
            # Load sub-encoders
            self.neural_binning.load_state(f"{filepath}_binning")
            self.attention_pooling.load_state(f"{filepath}_attention")
            
            logger.info(f"Loaded PropertyEncoder state from {filepath}")
        except Exception as e:
            logger.error(f"Error loading PropertyEncoder state: {e}")