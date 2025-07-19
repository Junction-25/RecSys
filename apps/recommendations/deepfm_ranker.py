"""
DeepFM Ranker - Stage 4 of the TikTok-like SOTA Recommender System.
Combines FM (Factorization Machine) and DNN for ranking candidate buyers.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import pickle

logger = logging.getLogger(__name__)


class FactorizationMachine(nn.Module):
    """
    Factorization Machine component for capturing lower-order feature interactions.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 10):
        """
        Initialize FM component.
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding dimension for interactions
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Linear terms
        self.linear = nn.Linear(input_dim, 1)
        
        # Embedding for interactions
        self.embeddings = nn.Embedding(input_dim, embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.normal_(self.linear.weight, std=0.01)
    
    def forward(self, x):
        """
        Forward pass for FM.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            FM output [batch_size, 1]
        """
        # Linear terms
        linear_part = self.linear(x)
        
        # Interaction terms
        # Create indices for embeddings
        batch_size = x.size(0)
        indices = torch.arange(self.input_dim).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        
        # Get embeddings and weight by input features
        emb = self.embeddings(indices)  # [batch_size, input_dim, embedding_dim]
        weighted_emb = emb * x.unsqueeze(-1)  # [batch_size, input_dim, embedding_dim]
        
        # Calculate interaction: 0.5 * (sum^2 - sum_of_squares)
        sum_square = torch.sum(weighted_emb, dim=1) ** 2  # [batch_size, embedding_dim]
        square_sum = torch.sum(weighted_emb ** 2, dim=1)  # [batch_size, embedding_dim]
        
        interaction_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        return linear_part + interaction_part


class DeepNeuralNetwork(nn.Module):
    """
    Deep Neural Network component for capturing higher-order feature interactions.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], dropout: float = 0.2):
        """
        Initialize DNN component.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for DNN.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            DNN output [batch_size, 1]
        """
        return self.network(x)


class DeepFM(nn.Module):
    """
    DeepFM model combining FM and DNN components.
    """
    
    def __init__(self, 
                 input_dim: int,
                 fm_embedding_dim: int = 10,
                 dnn_hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2):
        """
        Initialize DeepFM model.
        
        Args:
            input_dim: Input feature dimension
            fm_embedding_dim: FM embedding dimension
            dnn_hidden_dims: DNN hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fm = FactorizationMachine(input_dim, fm_embedding_dim)
        self.dnn = DeepNeuralNetwork(input_dim, dnn_hidden_dims, dropout)
        
        # Combination weights
        self.fm_weight = nn.Parameter(torch.tensor(0.5))
        self.dnn_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        Forward pass for DeepFM.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Ranking scores [batch_size, 1]
        """
        fm_output = self.fm(x)
        dnn_output = self.dnn(x)
        
        # Weighted combination
        output = self.fm_weight * fm_output + self.dnn_weight * dnn_output
        
        return torch.sigmoid(output)


class DeepFMRanker:
    """
    DeepFM-based ranker for buyer recommendations.
    Ranks candidate buyers based on predicted relevance scores.
    """
    
    def __init__(self, 
                 feature_dim: int = 1024,
                 fm_embedding_dim: int = 10,
                 dnn_hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize DeepFM ranker.
        
        Args:
            feature_dim: Input feature dimension
            fm_embedding_dim: FM embedding dimension
            dnn_hidden_dims: DNN hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate for training
        """
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = DeepFM(
            input_dim=feature_dim,
            fm_embedding_dim=fm_embedding_dim,
            dnn_hidden_dims=dnn_hidden_dims,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Feature scaler
        self.feature_scaler = None
        
        logger.info(f"Initialized DeepFM ranker with feature dim: {feature_dim}")
    
    def extract_features(self, property_embedding: np.ndarray, buyer_embedding: np.ndarray, 
                        interaction_features: Dict[str, float] = None) -> np.ndarray:
        """
        Extract features for DeepFM ranking.
        
        Args:
            property_embedding: Property embedding vector
            buyer_embedding: Buyer embedding vector
            interaction_features: Additional interaction features
            
        Returns:
            Combined feature vector
        """
        try:
            features = []
            
            # Property and buyer embeddings
            features.extend(property_embedding.flatten())
            features.extend(buyer_embedding.flatten())
            
            # Interaction features
            if interaction_features:
                # Budget match score
                features.append(interaction_features.get('budget_match', 0.0))
                
                # Location match score
                features.append(interaction_features.get('location_match', 0.0))
                
                # Property type match
                features.append(interaction_features.get('property_type_match', 0.0))
                
                # Area match score
                features.append(interaction_features.get('area_match', 0.0))
                
                # Rooms match score
                features.append(interaction_features.get('rooms_match', 0.0))
                
                # Feature preferences match
                features.append(interaction_features.get('features_match', 0.0))
                
                # Cosine similarity between embeddings
                cosine_sim = np.dot(property_embedding, buyer_embedding) / (
                    np.linalg.norm(property_embedding) * np.linalg.norm(buyer_embedding) + 1e-8
                )
                features.append(cosine_sim)
                
                # Cross features
                features.append(interaction_features.get('budget_match', 0.0) * cosine_sim)
                features.append(interaction_features.get('location_match', 0.0) * cosine_sim)
            
            # Pad or truncate to fixed dimension
            features_array = np.array(features, dtype=np.float32)
            if len(features_array) < self.feature_dim:
                # Pad with zeros
                padding = np.zeros(self.feature_dim - len(features_array), dtype=np.float32)
                features_array = np.concatenate([features_array, padding])
            elif len(features_array) > self.feature_dim:
                # Truncate
                features_array = features_array[:self.feature_dim]
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def rank_candidates(self, property_embedding: np.ndarray, 
                       candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimized ranking of candidate buyers using DeepFM with batch processing.
        
        Args:
            property_embedding: Property embedding vector
            candidates: List of candidate buyers with embeddings and metadata
            
        Returns:
            Ranked list of candidates with scores
        """
        start_time = time.time()
        
        try:
            if not candidates:
                return []
            
            # Limit candidates for performance (top 200 max)
            if len(candidates) > 200:
                candidates = candidates[:200]
                logger.warning(f"Limited candidates to 200 for performance")
            
            # Use fast cosine similarity for initial filtering if too many candidates
            if len(candidates) > 100:
                candidates = self._fast_prefilter(property_embedding, candidates, top_k=100)
            
            # Batch process features for better performance
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_features = []
                
                # Extract features for batch
                for candidate in batch:
                    buyer_embedding = candidate.get('embedding', np.zeros(512))
                    interaction_features = candidate.get('interaction_features', {})
                    
                    features = self.extract_features_fast(
                        property_embedding, 
                        buyer_embedding, 
                        interaction_features
                    )
                    batch_features.append(features)
                
                # Convert batch to tensor and get predictions
                if batch_features:
                    features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(self.device)
                    
                    self.model.eval()
                    with torch.no_grad():
                        batch_scores = self.model(features_tensor).cpu().numpy().flatten()
                        all_scores.extend(batch_scores)
            
            # Add scores to candidates
            for i, candidate in enumerate(candidates):
                if i < len(all_scores):
                    candidate['deepfm_score'] = float(all_scores[i])
                else:
                    candidate['deepfm_score'] = 0.0
                candidate['rank_method'] = 'deepfm_optimized'
            
            # Sort by score (descending)
            ranked_candidates = sorted(candidates, key=lambda x: x['deepfm_score'], reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Ranked {len(candidates)} candidates using optimized DeepFM in {processing_time:.2f}ms")
            return ranked_candidates
            
        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            # Fallback to cosine similarity
            return self._fallback_cosine_ranking(property_embedding, candidates)
    
    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            features: Input features [batch_size, feature_dim]
            labels: Target labels [batch_size, 1]
            
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_on_batch(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
        """
        Train on a batch of data.
        
        Args:
            training_data: List of (property_embedding, buyer_embedding, relevance_score) tuples
            
        Returns:
            Average training loss
        """
        try:
            if not training_data:
                return 0.0
            
            # Prepare features and labels
            features_list = []
            labels_list = []
            
            for prop_emb, buyer_emb, relevance in training_data:
                features = self.extract_features(prop_emb, buyer_emb)
                features_list.append(features)
                labels_list.append([relevance])
            
            # Convert to tensors
            features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(self.device)
            labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32).to(self.device)
            
            # Training step
            loss = self.train_step(features_tensor, labels_tensor)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            return 0.0
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: List of (property_embedding, buyer_embedding, relevance_score) tuples
            
        Returns:
            Evaluation metrics
        """
        try:
            if not test_data:
                return {}
            
            # Prepare features and labels
            features_list = []
            labels_list = []
            
            for prop_emb, buyer_emb, relevance in test_data:
                features = self.extract_features(prop_emb, buyer_emb)
                features_list.append(features)
                labels_list.append(relevance)
            
            # Convert to tensors
            features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(self.device)
            labels_tensor = torch.tensor(np.array(labels_list), dtype=torch.float32).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(features_tensor).cpu().numpy().flatten()
                labels_np = labels_tensor.cpu().numpy().flatten()
                
                # Calculate metrics
                mse = np.mean((predictions - labels_np) ** 2)
                mae = np.mean(np.abs(predictions - labels_np))
                
                # Correlation
                correlation = np.corrcoef(predictions, labels_np)[0, 1] if len(predictions) > 1 else 0.0
                
            return {
                'mse': float(mse),
                'mae': float(mae),
                'correlation': float(correlation),
                'num_samples': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'feature_dim': self.feature_dim,
                'feature_scaler': self.feature_scaler
            }
            torch.save(state, filepath)
            logger.info(f"Saved DeepFM model to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            state = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.feature_dim = state['feature_dim']
            self.feature_scaler = state.get('feature_scaler')
            logger.info(f"Loaded DeepFM model from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def extract_features_fast(self, property_embedding: np.ndarray, buyer_embedding: np.ndarray, 
                              interaction_features: Dict[str, float] = None) -> np.ndarray:
        """
        Optimized feature extraction with reduced complexity.
        
        Args:
            property_embedding: Property embedding vector
            buyer_embedding: Buyer embedding vector  
            interaction_features: Additional interaction features
            
        Returns:
            Optimized feature vector
        """
        try:
            features = []
            
            # 1. Most important: Cosine similarity
            cosine_sim = np.dot(property_embedding, buyer_embedding) / (
                np.linalg.norm(property_embedding) * np.linalg.norm(buyer_embedding) + 1e-8
            )
            features.append(cosine_sim)
            
            # 2. Key interaction features (if available)
            if interaction_features:
                features.extend([
                    interaction_features.get('budget_match', 0.5),
                    interaction_features.get('location_match', 0.5),
                    interaction_features.get('property_type_match', 0.5),
                    interaction_features.get('area_match', 0.5),
                    interaction_features.get('rooms_match', 0.5)
                ])
            else:
                features.extend([0.5] * 5)  # Default neutral values
            
            # 3. Cross features (most predictive)
            budget_match = interaction_features.get('budget_match', 0.5) if interaction_features else 0.5
            location_match = interaction_features.get('location_match', 0.5) if interaction_features else 0.5
            
            features.extend([
                cosine_sim * budget_match,
                cosine_sim * location_match,
            ])
            
            # 4. Simple statistical features
            features.extend([
                np.mean(property_embedding[:10]),  # Sample of property embedding
                np.mean(buyer_embedding[:10]),     # Sample of buyer embedding
            ])
            
            # Pad to fixed smaller dimension (much smaller than original)
            target_dim = min(self.feature_dim, 64)  # Use much smaller feature dimension
            features_array = np.array(features, dtype=np.float32)
            
            if len(features_array) < target_dim:
                # Pad with cosine similarity (most important feature)
                padding = np.full(target_dim - len(features_array), cosine_sim, dtype=np.float32)
                features_array = np.concatenate([features_array, padding])
            elif len(features_array) > target_dim:
                features_array = features_array[:target_dim]
            
            # Extend to full feature_dim if needed
            if len(features_array) < self.feature_dim:
                final_padding = np.zeros(self.feature_dim - len(features_array), dtype=np.float32)
                features_array = np.concatenate([features_array, final_padding])
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error in fast feature extraction: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _fast_prefilter(self, property_embedding: np.ndarray, candidates: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Fast prefiltering using cosine similarity to reduce candidates before DeepFM.
        """
        try:
            # Calculate cosine similarities
            similarities = []
            for candidate in candidates:
                buyer_embedding = candidate.get('embedding', np.zeros(512))
                cosine_sim = np.dot(property_embedding, buyer_embedding) / (
                    np.linalg.norm(property_embedding) * np.linalg.norm(buyer_embedding) + 1e-8
                )
                similarities.append((cosine_sim, candidate))
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [item[1] for item in similarities[:top_k]]
            
            logger.debug(f"Prefiltered {len(candidates)} candidates to top {len(top_candidates)} using cosine similarity")
            return top_candidates
            
        except Exception as e:
            logger.error(f"Error in prefiltering: {e}")
            return candidates[:top_k]
    
    def _fallback_cosine_ranking(self, property_embedding: np.ndarray, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback ranking using simple cosine similarity.
        """
        try:
            for candidate in candidates:
                buyer_embedding = candidate.get('embedding', np.zeros(512))
                cosine_sim = np.dot(property_embedding, buyer_embedding) / (
                    np.linalg.norm(property_embedding) * np.linalg.norm(buyer_embedding) + 1e-8
                )
                candidate['deepfm_score'] = float(cosine_sim)
                candidate['rank_method'] = 'cosine_fallback'
            
            return sorted(candidates, key=lambda x: x['deepfm_score'], reverse=True)
        except Exception as e:
            logger.error(f"Error in fallback ranking: {e}")
            return candidates
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'DeepFM_Optimized',
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }