"""
Vector Database - Stage 2 of the TikTok-like SOTA Recommender System.
Stores buyer embeddings and provides fast ANN (Approximate Nearest Neighbor) retrieval.
"""
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.core.cache import cache
import time

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector Database for fast similarity search using FAISS.
    Stores buyer embeddings and provides ANN retrieval for candidate generation.
    """
    
    def __init__(self, embedding_dim: int = 512, index_type: str = "IVF"):
        """
        Initialize Vector Database.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.buyer_ids = []  # Maps index positions to buyer IDs
        self.buyer_metadata = {}  # Stores buyer metadata
        self.buyer_embeddings = {}  # Store embeddings separately for retrieval
        self.is_trained = False
        
        self._init_index()
        
    def _init_index(self):
        """Initialize FAISS index based on type."""
        try:
            if self.index_type == "Flat":
                # Exact search (slower but accurate)
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
                
            elif self.index_type == "IVF":
                # Inverted File Index (faster approximate search)
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                
            elif self.index_type == "HNSW":
                # Hierarchical Navigable Small World (very fast)
                M = 32  # Number of connections
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
            logger.info(f"Initialized FAISS index: {self.index_type} with dim {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            # Fallback to flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index_type = "Flat"
    
    def add_buyer_embedding(self, buyer_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """
        Add a buyer embedding to the database.
        
        Args:
            buyer_id: Unique buyer identifier
            embedding: Buyer embedding vector
            metadata: Additional buyer metadata
        """
        try:
            # Normalize embedding for cosine similarity
            embedding = embedding.astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Add to index
            if self.index_type == "IVF" and not self.is_trained:
                # IVF index needs training
                if len(self.buyer_ids) == 0:
                    # Store first embedding for training later
                    self.buyer_ids.append(buyer_id)
                    self.buyer_metadata[buyer_id] = metadata or {}
                    return
                    
            self.index.add(embedding.reshape(1, -1))
            self.buyer_ids.append(buyer_id)
            self.buyer_metadata[buyer_id] = metadata or {}
            self.buyer_embeddings[buyer_id] = embedding  # Store embedding for retrieval
            
            logger.debug(f"Added buyer {buyer_id} to vector database")
            
        except Exception as e:
            logger.error(f"Error adding buyer embedding: {e}")
    
    def batch_add_buyers(self, buyers_data: List[Tuple[str, np.ndarray, Dict]]):
        """
        Add multiple buyer embeddings in batch.
        
        Args:
            buyers_data: List of (buyer_id, embedding, metadata) tuples
        """
        try:
            if not buyers_data:
                return
                
            # Prepare embeddings
            embeddings = []
            for buyer_id, embedding, metadata in buyers_data:
                # Normalize embedding
                embedding = embedding.astype(np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings.append(embedding)
                # Store metadata and embeddings
                self.buyer_ids.append(buyer_id)
                self.buyer_metadata[buyer_id] = metadata or {}
                self.buyer_embeddings[buyer_id] = embedding  # Store embedding for retrieval
            
            embeddings_array = np.array(embeddings)
            
            # Train index if needed
            if self.index_type == "IVF" and not self.is_trained:
                logger.info("Training IVF index...")
                self.index.train(embeddings_array)
                self.is_trained = True
            
            # Add embeddings
            self.index.add(embeddings_array)
            
            logger.info(f"Added {len(buyers_data)} buyers to vector database")
            
        except Exception as e:
            logger.error(f"Error in batch add buyers: {e}")
    
    def search_similar_buyers(self, query_embedding: np.ndarray, k: int = 50) -> List[Dict[str, Any]]:
        """
        Search for similar buyers using ANN.
        
        Args:
            query_embedding: Property embedding to search with
            k: Number of similar buyers to return
            
        Returns:
            List of similar buyers with scores and metadata
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Vector database is empty")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Search
            start_time = time.time()
            scores, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                    
                if idx >= len(self.buyer_ids):
                    continue
                    
                buyer_id = self.buyer_ids[idx]
                metadata = self.buyer_metadata.get(buyer_id, {})
                embedding = self.buyer_embeddings.get(buyer_id, np.zeros(self.embedding_dim))
                
                results.append({
                    'buyer_id': buyer_id,
                    'similarity_score': float(score),
                    'metadata': metadata,
                    'embedding': embedding,  # Include actual embedding
                    'index_position': int(idx)
                })
            
            logger.debug(f"Vector search completed in {search_time:.2f}ms, found {len(results)} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar buyers: {e}")
            return []
    
    def get_buyer_embedding(self, buyer_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific buyer.
        
        Args:
            buyer_id: Buyer identifier
            
        Returns:
            Buyer embedding or None if not found
        """
        try:
            if buyer_id not in self.buyer_ids:
                return None
                
            idx = self.buyer_ids.index(buyer_id)
            
            # FAISS doesn't directly support getting vectors by index
            # This is a limitation - in production, you'd store embeddings separately
            logger.warning("Direct embedding retrieval not implemented for FAISS")
            return None
            
        except Exception as e:
            logger.error(f"Error getting buyer embedding: {e}")
            return None
    
    def update_buyer_embedding(self, buyer_id: str, new_embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """
        Update a buyer's embedding (requires rebuilding index).
        
        Args:
            buyer_id: Buyer identifier
            new_embedding: New embedding vector
            metadata: Updated metadata
        """
        try:
            if buyer_id not in self.buyer_ids:
                logger.warning(f"Buyer {buyer_id} not found for update")
                return
                
            # Update metadata
            if metadata:
                self.buyer_metadata[buyer_id] = metadata
                
            # For FAISS, updating requires rebuilding the index
            # In production, you'd implement incremental updates
            logger.warning("Embedding update requires index rebuild - not implemented")
            
        except Exception as e:
            logger.error(f"Error updating buyer embedding: {e}")
    
    def remove_buyer(self, buyer_id: str):
        """
        Remove a buyer from the database (requires rebuilding index).
        
        Args:
            buyer_id: Buyer identifier to remove
        """
        try:
            if buyer_id not in self.buyer_ids:
                logger.warning(f"Buyer {buyer_id} not found for removal")
                return
                
            # Remove from metadata
            if buyer_id in self.buyer_metadata:
                del self.buyer_metadata[buyer_id]
                
            # Remove from buyer_ids
            self.buyer_ids.remove(buyer_id)
            
            # For FAISS, removal requires rebuilding the index
            logger.warning("Buyer removal requires index rebuild - not implemented")
            
        except Exception as e:
            logger.error(f"Error removing buyer: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Database statistics
        """
        return {
            'total_buyers': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'memory_usage_mb': self.index.ntotal * self.embedding_dim * 4 / (1024 * 1024) if self.index else 0
        }
    
    def rebuild_index(self, buyers_data: List[Tuple[str, np.ndarray, Dict]]):
        """
        Rebuild the entire index with new data.
        
        Args:
            buyers_data: List of (buyer_id, embedding, metadata) tuples
        """
        try:
            logger.info("Rebuilding vector database index...")
            
            # Reset state
            self._init_index()
            self.buyer_ids = []
            self.buyer_metadata = {}
            self.is_trained = False
            
            # Add all buyers
            self.batch_add_buyers(buyers_data)
            
            logger.info(f"Index rebuilt with {len(buyers_data)} buyers")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
    
    def save_index(self, filepath: str):
        """
        Save the vector database to disk.
        
        Args:
            filepath: Path to save the index
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                'buyer_ids': self.buyer_ids,
                'buyer_metadata': self.buyer_metadata,
                'buyer_embeddings': self.buyer_embeddings,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Saved vector database to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def load_index(self, filepath: str):
        """
        Load the vector database from disk.
        
        Args:
            filepath: Path to load the index from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            self.buyer_ids = metadata['buyer_ids']
            self.buyer_metadata = metadata['buyer_metadata']
            self.buyer_embeddings = metadata.get('buyer_embeddings', {})
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Loaded vector database from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
    
    def clear_cache(self):
        """Clear any cached data."""
        try:
            # Clear Django cache entries related to vector search
            cache_keys = [
                'vector_search_*',
                'buyer_embeddings_*',
                'similar_buyers_*'
            ]
            
            for pattern in cache_keys:
                cache.delete_many(cache.keys(pattern))
                
            logger.info("Cleared vector database cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")