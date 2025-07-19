"""
Recommendation Engine - TikTok-like Architecture
Two-stage retrieval & ranking pipeline optimized for millisecond response times.

Architecture:
Property → Property Encoder → Buyer Vector DB (ANN) → Feature Store → Ranker → Ordered Buyer List

Components:
- Property Encoder: ColBERT + Attention Pooling + Pre-trained Neural Binning
- Buyer Vector DB: FAISS-based fast similarity search  
- Feature Store: Enriches candidates with interaction features
- Ranker: DeepFM for final ranking with MaxSim similarity
"""
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.core.cache import cache
from django.conf import settings

from .property_encoder import PropertyEncoder
from .vector_database import VectorDatabase
from .deepfm_ranker import DeepFMRanker
from .colbert_encoder import ColBERTEncoder

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    TikTok-like Recommendation Engine optimized for millisecond response times.
    
    Pipeline:
    Property → Property Encoder → Vector DB (ANN) → Feature Store → Ranker → Ordered Buyer List
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 vector_db_type: str = "Flat",  # Use Flat for simplicity
                 enable_caching: bool = True):
        """
        Initialize TikTok-like Recommendation Engine.
        
        Args:
            embedding_dim: Embedding dimension for all components
            vector_db_type: Vector database index type ("Flat", "IVF", "HNSW")
            enable_caching: Whether to enable result caching
        """
        self.embedding_dim = embedding_dim
        self.enable_caching = enable_caching
        
        # Initialize components
        self.property_encoder = PropertyEncoder(final_embedding_dim=embedding_dim)
        self.vector_db = VectorDatabase(embedding_dim=embedding_dim, index_type=vector_db_type)
        self.deepfm_ranker = DeepFMRanker(feature_dim=embedding_dim * 2 + 10)  # Property + Buyer + interaction features
        self.colbert_encoder = ColBERTEncoder()
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'avg_response_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'ann_retrieval_time_ms': 0.0,
            'ranking_time_ms': 0.0
        }
        
        logger.info("Initialized TikTok-like Recommendation Engine with Lightweight Encoder")
        
        # Try to load saved engine state or populate from database
        self._initialize_from_saved_state_or_database()
    
    def _initialize_from_saved_state_or_database(self):
        """Initialize engine from saved state or populate from database embeddings."""
        try:
            import os
            
            # Try to load saved engine state first
            engine_path = os.path.join("models", "recommendation_engine")
            if os.path.exists(f"{engine_path}_system_metadata.pkl"):
                logger.info("Found saved engine state, loading...")
                self.load_system(engine_path)
                return
            
            # If no saved state, populate from database embeddings
            logger.info("No saved engine state found, starting fresh")
            self._populate_vector_db_from_database()
            
        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
    
    def _populate_vector_db_from_database(self):
        """Populate vector database from database embeddings."""
        try:
            from apps.contacts.models import Contact
            import pickle
            
            # Get all contacts with embeddings
            contacts_with_embeddings = Contact.objects.filter(embedding__isnull=False)
            total_contacts = contacts_with_embeddings.count()
            
            if total_contacts == 0:
                logger.info("No contacts with embeddings found in database")
                return
            
            logger.info(f"Loading {total_contacts} buyer embeddings from database...")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            loaded_count = 0
            
            for i in range(0, total_contacts, batch_size):
                batch_contacts = contacts_with_embeddings[i:i+batch_size]
                
                for contact in batch_contacts:
                    try:
                        # Deserialize embedding from database
                        buyer_embedding = pickle.loads(contact.embedding)
                        
                        # Prepare metadata
                        buyer_metadata = {
                            'buyer_id': str(contact.id),
                            'name': contact.name,
                            'email': contact.email,
                            'phone': contact.phone,
                            'budget_min': float(contact.budget_min) if contact.budget_min else None,
                            'budget_max': float(contact.budget_max) if contact.budget_max else None,
                            'property_type': contact.property_type,
                            'preferred_locations': contact.preferred_locations,
                            'status': contact.status,
                        }
                        
                        # Add to vector database
                        self.vector_db.add_buyer_embedding(str(contact.id), buyer_embedding, buyer_metadata)
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error loading embedding for contact {contact.id}: {e}")
                        continue
            
            logger.info(f"Successfully loaded {loaded_count}/{total_contacts} buyer embeddings into vector database")
            
        except Exception as e:
            logger.error(f"Error populating vector DB from database: {e}")
    
    def encode_property(self, property_obj) -> np.ndarray:
        """
        Encode property using the Property Encoder.
        
        Args:
            property_obj: Property object
            
        Returns:
            Property embedding vector
        """
        return self.property_encoder.encode_property(property_obj)
    
    def get_property_embedding(self, property_obj) -> np.ndarray:
        """
        Get property embedding from database (pre-computed) or encode if not available.
        
        Args:
            property_obj: Property object
            
        Returns:
            Property embedding vector
        """
        try:
            import pickle
            
            # Try to load pre-computed embedding from database
            if hasattr(property_obj, 'embedding') and property_obj.embedding:
                try:
                    embedding = pickle.loads(property_obj.embedding)
                    logger.debug(f"Loaded pre-computed embedding for property {property_obj.id}")
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load pre-computed embedding for property {property_obj.id}: {e}")
            
            # Fallback to real-time encoding
            logger.debug(f"Computing embedding for property {property_obj.id} (no pre-computed embedding)")
            return self.encode_property(property_obj)
            
        except Exception as e:
            logger.error(f"Error getting property embedding: {e}")
            # Fallback to real-time encoding
            return self.encode_property(property_obj)
    
    def encode_buyer(self, contact_obj) -> np.ndarray:
        """
        Encode buyer preferences using similar architecture as properties.
        
        Args:
            contact_obj: Contact/buyer object
            
        Returns:
            Buyer embedding vector
        """
        try:
            # Use ColBERT for buyer preferences text
            buyer_text_emb = self.colbert_encoder.encode_buyer_preferences(contact_obj)
            
            # Use attention pooling for preferred locations
            buyer_location_emb = self.property_encoder.attention_pooling.encode_buyer_preferred_locations(contact_obj)
            
            # Use neural binning for numerical preferences
            buyer_numerical_emb = self.property_encoder.neural_binning.encode_buyer_numerical(contact_obj)
            
            # Encode categorical preferences
            buyer_categorical_emb = self._encode_buyer_categorical(contact_obj)
            
            # Combine all embeddings
            combined_emb = np.concatenate([
                buyer_text_emb,
                buyer_location_emb,
                buyer_numerical_emb,
                buyer_categorical_emb
            ])
            
            # Pass through fusion network (reuse property encoder's network)
            import torch
            with torch.no_grad():
                combined_tensor = torch.tensor(combined_emb, dtype=torch.float32)
                # Pad or truncate to match expected input size
                if len(combined_tensor) != self.property_encoder.fusion_network[0].in_features:
                    target_size = self.property_encoder.fusion_network[0].in_features
                    if len(combined_tensor) < target_size:
                        padding = torch.zeros(target_size - len(combined_tensor))
                        combined_tensor = torch.cat([combined_tensor, padding])
                    else:
                        combined_tensor = combined_tensor[:target_size]
                
                buyer_embedding = self.property_encoder.fusion_network(combined_tensor).numpy()
            
            return buyer_embedding
            
        except Exception as e:
            logger.error(f"Error encoding buyer: {e}")
            return np.zeros(self.embedding_dim)
    
    def _encode_buyer_categorical(self, contact_obj) -> np.ndarray:
        """Encode buyer categorical features."""
        try:
            import torch
            embeddings = []
            
            # Property type preference
            if hasattr(contact_obj, 'property_type') and contact_obj.property_type:
                prop_type_id = self.property_encoder._get_category_id('property_type', str(contact_obj.property_type))
                with torch.no_grad():
                    prop_type_emb = self.property_encoder.categorical_encoders['property_type'](
                        torch.tensor(prop_type_id)
                    ).numpy()
                embeddings.append(prop_type_emb)
            else:
                embeddings.append(np.zeros(self.property_encoder.categorical_embedding_dim))
            
            # Status preference (active buyer)
            status_emb = np.zeros(self.property_encoder.categorical_embedding_dim)
            if hasattr(contact_obj, 'status'):
                status_id = self.property_encoder._get_category_id('status', str(contact_obj.status))
                with torch.no_grad():
                    status_emb = self.property_encoder.categorical_encoders['status'](
                        torch.tensor(status_id)
                    ).numpy()
            embeddings.append(status_emb)
            
            return np.concatenate(embeddings)
            
        except Exception as e:
            logger.error(f"Error encoding buyer categorical features: {e}")
            return np.zeros(self.property_encoder.categorical_embedding_dim * 2)
    
    def add_buyer_to_vector_db(self, buyer_id: str, contact_obj, metadata: Dict[str, Any] = None):
        """
        Add a buyer to the vector database.
        
        Args:
            buyer_id: Unique buyer identifier
            contact_obj: Contact/buyer object
            metadata: Additional buyer metadata
        """
        try:
            buyer_embedding = self.encode_buyer(contact_obj)
            
            # Prepare metadata
            buyer_metadata = {
                'buyer_id': buyer_id,
                'name': getattr(contact_obj, 'name', ''),
                'email': getattr(contact_obj, 'email', ''),
                'phone': getattr(contact_obj, 'phone', ''),
                'budget_min': getattr(contact_obj, 'budget_min', None),
                'budget_max': getattr(contact_obj, 'budget_max', None),
                'property_type': getattr(contact_obj, 'property_type', ''),
                'preferred_locations': getattr(contact_obj, 'preferred_locations', []),
                'status': getattr(contact_obj, 'status', ''),
                **(metadata or {})
            }
            
            self.vector_db.add_buyer_embedding(buyer_id, buyer_embedding, buyer_metadata)
            
        except Exception as e:
            logger.error(f"Error adding buyer to vector DB: {e}")
    
    def batch_add_buyers_to_vector_db(self, buyers_data: List[Tuple[str, Any, Dict]]):
        """
        Add multiple buyers to vector database in batch.
        
        Args:
            buyers_data: List of (buyer_id, contact_obj, metadata) tuples
        """
        try:
            processed_data = []
            
            for buyer_id, contact_obj, metadata in buyers_data:
                buyer_embedding = self.encode_buyer(contact_obj)
                
                buyer_metadata = {
                    'buyer_id': buyer_id,
                    'name': getattr(contact_obj, 'name', ''),
                    'email': getattr(contact_obj, 'email', ''),
                    'phone': getattr(contact_obj, 'phone', ''),
                    'budget_min': getattr(contact_obj, 'budget_min', None),
                    'budget_max': getattr(contact_obj, 'budget_max', None),
                    'property_type': getattr(contact_obj, 'property_type', ''),
                    'preferred_locations': getattr(contact_obj, 'preferred_locations', []),
                    'status': getattr(contact_obj, 'status', ''),
                    **(metadata or {})
                }
                
                processed_data.append((buyer_id, buyer_embedding, buyer_metadata))
            
            self.vector_db.batch_add_buyers(processed_data)
            
        except Exception as e:
            logger.error(f"Error batch adding buyers to vector DB: {e}")
    
    def get_recommendations_for_property(self, 
                                       property_id: str,
                                       max_candidates: int = 100,
                                       max_results: int = 20,
                                       min_score: float = 0.3) -> Dict[str, Any]:
        """
        Get buyer recommendations for a property using the TikTok-like pipeline.
        
        Args:
            property_id: Property ID to get recommendations for
            max_candidates: Maximum candidates from ANN retrieval
            max_results: Maximum final results after ranking
            min_score: Minimum score threshold
            
        Returns:
            Recommendation results with performance metrics
        """
        start_time = time.time()
        
        # Get property object from database
        try:
            from apps.properties.models import Property
            try:
                property_obj = Property.objects.get(id=property_id)
            except (ValueError, Property.DoesNotExist):
                property_obj = Property.objects.get(external_id=property_id)
        except Property.DoesNotExist:
            return {
                'property_id': property_id,
                'recommendations': [],
                'total_count': 0,
                'error': 'Property not found',
                'processing_time_ms': (time.time() - start_time) * 1000,
                'method': 'tiktok_pipeline',
                'cached': False
            }
        
        # Check cache first
        if self.enable_caching:
            cache_key = f"sota_rec_{property_id}_{max_candidates}_{max_results}_{min_score}"
            cached_result = cache.get(cache_key)
            if cached_result:
                self.performance_stats['cache_hit_rate'] += 1
                logger.debug(f"Cache hit for property {property_id}")
                return cached_result
        
        try:
            # Stage 1: Property Embedding Retrieval (pre-computed)
            encoding_start = time.time()
            property_embedding = self.get_property_embedding(property_obj)
            encoding_time = (time.time() - encoding_start) * 1000
            
            # Stage 2: ANN Retrieval from Vector DB
            retrieval_start = time.time()
            candidates = self.vector_db.search_similar_buyers(property_embedding, k=max_candidates)
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            if not candidates:
                return {
                    'property_id': property_id,
                    'recommendations': [],
                    'total_count': 0,
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'performance_breakdown': {
                        'encoding_time_ms': encoding_time,
                        'retrieval_time_ms': retrieval_time,
                        'ranking_time_ms': 0,
                        'feature_enrichment_time_ms': 0
                    },
                    'method': 'sota_pipeline',
                    'cached': False
                }
            
            # Stage 3: Feature Store - Enrich candidates with interaction features
            enrichment_start = time.time()
            enriched_candidates = self._enrich_candidates_with_features(property_obj, candidates)
            enrichment_time = (time.time() - enrichment_start) * 1000
            
            # Stage 4: DeepFM Ranking
            ranking_start = time.time()
            ranked_candidates = self.deepfm_ranker.rank_candidates(property_embedding, enriched_candidates)
            ranking_time = (time.time() - ranking_start) * 1000
            
            # Filter by minimum score and limit results
            final_recommendations = []
            for candidate in ranked_candidates[:max_results]:
                if candidate.get('deepfm_score', 0) >= min_score:
                    final_recommendations.append({
                        'buyer_id': candidate['buyer_id'],
                        'score': candidate['deepfm_score'],
                        'similarity_score': candidate.get('similarity_score', 0),
                        'buyer': candidate['metadata'],
                        'explanation': self._generate_explanation(candidate),
                        'method': 'sota_deepfm',
                        'feature_breakdown': candidate.get('interaction_features', {})
                    })
            
            # Prepare result
            total_time = (time.time() - start_time) * 1000
            result = {
                'property_id': property_id,
                'recommendations': final_recommendations,
                'total_count': len(final_recommendations),
                'processing_time_ms': total_time,
                'performance_breakdown': {
                    'encoding_time_ms': encoding_time,
                    'retrieval_time_ms': retrieval_time,
                    'ranking_time_ms': ranking_time,
                    'feature_enrichment_time_ms': enrichment_time
                },
                'method': 'sota_pipeline',
                'cached': False,
                'candidates_retrieved': len(candidates),
                'candidates_ranked': len(ranked_candidates)
            }
            
            # Cache result
            if self.enable_caching:
                cache.set(cache_key, result, timeout=3600)  # Cache for 1 hour
            
            # Update performance stats
            self._update_performance_stats(total_time, retrieval_time, ranking_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in SOTA recommendation pipeline: {e}")
            return {
                'property_id': property_id,
                'recommendations': [],
                'total_count': 0,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'method': 'sota_pipeline',
                'cached': False
            }
    
    def _enrich_candidates_with_features(self, property_obj, candidates: List[Dict]) -> List[Dict]:
        """
        Enrich candidates with interaction features for DeepFM ranking.
        
        Args:
            property_obj: Property object
            candidates: List of candidate buyers from vector search
            
        Returns:
            Enriched candidates with interaction features
        """
        try:
            enriched = []
            
            for candidate in candidates:
                # Get buyer metadata
                buyer_metadata = candidate.get('metadata', {})
                
                # Calculate interaction features
                interaction_features = {}
                
                # Budget match
                if buyer_metadata.get('budget_min') and buyer_metadata.get('budget_max'):
                    prop_price = float(getattr(property_obj, 'price', 0))
                    budget_min = float(buyer_metadata['budget_min'])
                    budget_max = float(buyer_metadata['budget_max'])
                    
                    if budget_min <= prop_price <= budget_max:
                        interaction_features['budget_match'] = 1.0
                    else:
                        # Calculate how far outside budget
                        if prop_price < budget_min:
                            deviation = (budget_min - prop_price) / budget_min
                        else:
                            deviation = (prop_price - budget_max) / budget_max
                        interaction_features['budget_match'] = max(0.0, 1.0 - deviation)
                else:
                    interaction_features['budget_match'] = 0.5
                
                # Location match using attention pooling
                location_score = self.property_encoder.attention_pooling.calculate_location_match_score(
                    property_obj, type('obj', (), buyer_metadata)()
                )
                interaction_features['location_match'] = location_score
                
                # Property type match
                prop_type = getattr(property_obj, 'property_type', '')
                buyer_prop_type = buyer_metadata.get('property_type', '')
                interaction_features['property_type_match'] = 1.0 if prop_type == buyer_prop_type else 0.0
                
                # Area and rooms matches (simplified)
                interaction_features['area_match'] = 0.7  # Placeholder
                interaction_features['rooms_match'] = 0.6  # Placeholder
                interaction_features['features_match'] = 0.8  # Placeholder
                
                # Add enriched candidate
                enriched_candidate = candidate.copy()
                enriched_candidate['interaction_features'] = interaction_features
                # Use actual buyer embedding from vector search results
                enriched_candidate['embedding'] = candidate.get('embedding', np.zeros(self.embedding_dim))
                
                enriched.append(enriched_candidate)
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriching candidates: {e}")
            return candidates
    
    def _generate_explanation(self, candidate: Dict) -> str:
        """Generate human-readable explanation for recommendation."""
        try:
            explanations = []
            
            # Score-based explanation
            score = candidate.get('deepfm_score', 0)
            if score > 0.8:
                explanations.append("Excellent match based on AI analysis")
            elif score > 0.6:
                explanations.append("Good match with strong compatibility")
            elif score > 0.4:
                explanations.append("Moderate match with some compatibility")
            else:
                explanations.append("Basic match identified")
            
            # Feature-based explanations
            features = candidate.get('interaction_features', {})
            
            if features.get('budget_match', 0) > 0.8:
                explanations.append("Budget perfectly aligned")
            elif features.get('budget_match', 0) > 0.5:
                explanations.append("Budget reasonably compatible")
            
            if features.get('location_match', 0) > 0.7:
                explanations.append("Location preferences match well")
            
            if features.get('property_type_match', 0) == 1.0:
                explanations.append("Property type exactly matches preference")
            
            return ". ".join(explanations) + "."
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "AI-powered recommendation based on compatibility analysis."
    
    def _update_performance_stats(self, total_time: float, retrieval_time: float, ranking_time: float):
        """Update performance statistics."""
        try:
            self.performance_stats['total_requests'] += 1
            
            # Update moving averages
            alpha = 0.1  # Smoothing factor
            self.performance_stats['avg_response_time_ms'] = (
                alpha * total_time + (1 - alpha) * self.performance_stats['avg_response_time_ms']
            )
            self.performance_stats['ann_retrieval_time_ms'] = (
                alpha * retrieval_time + (1 - alpha) * self.performance_stats['ann_retrieval_time_ms']
            )
            self.performance_stats['ranking_time_ms'] = (
                alpha * ranking_time + (1 - alpha) * self.performance_stats['ranking_time_ms']
            )
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        stats['vector_db_stats'] = self.vector_db.get_database_stats()
        stats['model_info'] = self.deepfm_ranker.get_model_info()
        return stats
    
    def get_vector_db_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        return self.vector_db.get_database_stats()
    
    def fit_system(self, properties: List[Any], contacts: List[Any]):
        """
        Fit the entire system on training data.
        
        Args:
            properties: List of property objects
            contacts: List of contact objects
        """
        try:
            logger.info("Fitting SOTA Recommendation System...")
            
            # Fit property encoder
            self.property_encoder.fit_encoders(properties)
            
            # Add all buyers to vector database
            buyers_data = []
            for contact in contacts:
                buyer_id = str(getattr(contact, 'id', ''))
                if buyer_id:
                    buyers_data.append((buyer_id, contact, {}))
            
            if buyers_data:
                self.batch_add_buyers_to_vector_db(buyers_data)
            
            logger.info(f"SOTA system fitted with {len(properties)} properties and {len(contacts)} buyers")
            
        except Exception as e:
            logger.error(f"Error fitting SOTA system: {e}")
    
    def save_system(self, base_filepath: str):
        """Save the entire system state."""
        try:
            # Save individual components
            self.property_encoder.save_state(f"{base_filepath}_property_encoder")
            self.vector_db.save_index(f"{base_filepath}_vector_db")
            self.deepfm_ranker.save_model(f"{base_filepath}_deepfm_ranker.pth")
            
            # Save system metadata
            import pickle
            system_metadata = {
                'embedding_dim': self.embedding_dim,
                'performance_stats': self.performance_stats,
                'enable_caching': self.enable_caching
            }
            
            with open(f"{base_filepath}_system_metadata.pkl", 'wb') as f:
                pickle.dump(system_metadata, f)
            
            logger.info(f"Saved SOTA system to {base_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving SOTA system: {e}")
    
    def load_system(self, base_filepath: str):
        """Load the entire system state."""
        try:
            # Load individual components
            self.property_encoder.load_state(f"{base_filepath}_property_encoder")
            self.vector_db.load_index(f"{base_filepath}_vector_db")
            self.deepfm_ranker.load_model(f"{base_filepath}_deepfm_ranker.pth")
            
            # Load system metadata
            import pickle
            with open(f"{base_filepath}_system_metadata.pkl", 'rb') as f:
                system_metadata = pickle.load(f)
            
            self.embedding_dim = system_metadata['embedding_dim']
            self.performance_stats = system_metadata['performance_stats']
            self.enable_caching = system_metadata['enable_caching']
            
            logger.info(f"Loaded SOTA system from {base_filepath}")
            
        except Exception as e:
            logger.error(f"Error loading SOTA system: {e}")