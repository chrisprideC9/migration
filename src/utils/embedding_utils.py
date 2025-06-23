# src/utils/embedding_utils.py

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import Union, List, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_resource
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.
    
    Args:
        model_name: Name of the SentenceTransformer model to load
        
    Returns:
        Loaded SentenceTransformer model
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
        raise

def string_to_embedding(embedding_str: str) -> np.ndarray:
    """
    Convert a string representation of an embedding to a numpy array.
    
    Args:
        embedding_str: String representation of embedding (e.g., '[0.1, 0.2, 0.3]')
        
    Returns:
        Numpy array of the embedding, or empty array if conversion fails
        
    Examples:
        >>> string_to_embedding('[0.1, 0.2, 0.3]')
        array([0.1, 0.2, 0.3])
        >>> string_to_embedding('invalid')
        array([])
    """
    if not isinstance(embedding_str, str):
        return np.array([])
    
    try:
        # Remove brackets and split by comma
        cleaned = embedding_str.strip('[]')
        if not cleaned:
            return np.array([])
        
        # Convert to numpy array
        embedding = np.fromstring(cleaned, sep=',', dtype=float)
        
        # Validate the result
        if embedding.size == 0 or np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return np.array([])
        
        return embedding
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert string to embedding: {embedding_str[:50]}... Error: {e}")
        return np.array([])

def embedding_to_string(embedding: np.ndarray) -> str:
    """
    Convert a numpy array embedding to string representation.
    
    Args:
        embedding: Numpy array embedding
        
    Returns:
        String representation of the embedding
        
    Examples:
        >>> embedding_to_string(np.array([0.1, 0.2, 0.3]))
        '[0.1, 0.2, 0.3]'
    """
    try:
        if embedding.size == 0:
            return '[]'
        
        # Convert to string with reasonable precision
        values = [f"{x:.6f}" for x in embedding]
        return f"[{', '.join(values)}]"
        
    except Exception as e:
        logger.error(f"Failed to convert embedding to string: {e}")
        return '[]'

def validate_embedding(embedding: Union[str, np.ndarray], expected_dim: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate an embedding for correctness and dimensionality.
    
    Args:
        embedding: Embedding as string or numpy array
        expected_dim: Expected dimensionality (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Convert string to array if needed
        if isinstance(embedding, str):
            embedding_array = string_to_embedding(embedding)
        elif isinstance(embedding, np.ndarray):
            embedding_array = embedding
        else:
            return False, f"Invalid embedding type: {type(embedding)}"
        
        # Check if conversion was successful
        if embedding_array.size == 0:
            return False, "Empty or invalid embedding"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding_array)):
            return False, "Embedding contains NaN values"
        
        if np.any(np.isinf(embedding_array)):
            return False, "Embedding contains infinite values"
        
        # Check dimensionality if specified
        if expected_dim is not None and embedding_array.size != expected_dim:
            return False, f"Expected {expected_dim} dimensions, got {embedding_array.size}"
        
        # Check if embedding is normalized (optional validation)
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            return False, "Embedding has zero norm"
        
        return True, "Valid embedding"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def cosine_similarity_single(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score (0-1)
    """
    try:
        # Validate inputs
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        if embedding1.size != embedding2.size:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [0, 1] range (cosine similarity can be [-1, 1])
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding to unit length.
    
    Args:
        embedding: Input embedding
        
    Returns:
        Normalized embedding
    """
    try:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    except Exception as e:
        logger.error(f"Error normalizing embedding: {e}")
        return embedding

def batch_string_to_embeddings(embedding_strings: List[str]) -> List[np.ndarray]:
    """
    Convert a batch of embedding strings to numpy arrays.
    
    Args:
        embedding_strings: List of embedding strings
        
    Returns:
        List of numpy arrays
    """
    return [string_to_embedding(emb_str) for emb_str in embedding_strings]

def batch_validate_embeddings(embeddings: List[Union[str, np.ndarray]], expected_dim: Optional[int] = None) -> Tuple[List[bool], List[str]]:
    """
    Validate a batch of embeddings.
    
    Args:
        embeddings: List of embeddings
        expected_dim: Expected dimensionality
        
    Returns:
        Tuple of (validation_results, error_messages)
    """
    results = []
    messages = []
    
    for i, embedding in enumerate(embeddings):
        is_valid, message = validate_embedding(embedding, expected_dim)
        results.append(is_valid)
        messages.append(f"Embedding {i}: {message}")
    
    return results, messages

def calculate_embedding_statistics(embeddings: List[np.ndarray]) -> dict:
    """
    Calculate statistics for a collection of embeddings.
    
    Args:
        embeddings: List of numpy array embeddings
        
    Returns:
        Dictionary with embedding statistics
    """
    try:
        valid_embeddings = [emb for emb in embeddings if emb.size > 0]
        
        if not valid_embeddings:
            return {
                'count': 0,
                'valid_count': 0,
                'invalid_count': len(embeddings),
                'avg_dimension': 0,
                'dimension_consistency': False
            }
        
        # Stack embeddings for analysis
        try:
            stacked = np.stack(valid_embeddings)
            
            stats = {
                'count': len(embeddings),
                'valid_count': len(valid_embeddings),
                'invalid_count': len(embeddings) - len(valid_embeddings),
                'avg_dimension': stacked.shape[1],
                'dimension_consistency': True,
                'mean_values': np.mean(stacked, axis=0),
                'std_values': np.std(stacked, axis=0),
                'min_norm': np.min([np.linalg.norm(emb) for emb in valid_embeddings]),
                'max_norm': np.max([np.linalg.norm(emb) for emb in valid_embeddings]),
                'avg_norm': np.mean([np.linalg.norm(emb) for emb in valid_embeddings])
            }
        except ValueError:
            # Inconsistent dimensions
            dimensions = [emb.size for emb in valid_embeddings]
            stats = {
                'count': len(embeddings),
                'valid_count': len(valid_embeddings),
                'invalid_count': len(embeddings) - len(valid_embeddings),
                'avg_dimension': np.mean(dimensions),
                'dimension_consistency': False,
                'unique_dimensions': list(set(dimensions)),
                'min_dimension': min(dimensions),
                'max_dimension': max(dimensions)
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating embedding statistics: {e}")
        return {'error': str(e)}

def generate_text_embedding(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """
    Generate an embedding for a text string.
    
    Args:
        text: Input text
        model: SentenceTransformer model (loads default if None)
        
    Returns:
        Text embedding as numpy array
    """
    try:
        if model is None:
            model = load_sentence_transformer()
        
        if not text or not text.strip():
            return np.array([])
        
        embedding = model.encode(text)
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}")
        return np.array([])

def batch_generate_text_embeddings(texts: List[str], model: Optional[SentenceTransformer] = None) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of input texts
        model: SentenceTransformer model (loads default if None)
        
    Returns:
        List of text embeddings
    """
    try:
        if model is None:
            model = load_sentence_transformer()
        
        # Filter out empty texts
        valid_texts = [text if text and text.strip() else "" for text in texts]
        
        # Generate embeddings
        embeddings = model.encode(valid_texts)
        
        # Convert to list of numpy arrays
        if len(embeddings.shape) == 1:
            return [embeddings]
        else:
            return [embedding for embedding in embeddings]
        
    except Exception as e:
        logger.error(f"Error generating batch text embeddings: {e}")
        return [np.array([]) for _ in texts]

def find_most_similar_embeddings(query_embedding: np.ndarray, 
                                candidate_embeddings: List[np.ndarray], 
                                top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find the most similar embeddings to a query embedding.
    
    Args:
        query_embedding: Query embedding
        candidate_embeddings: List of candidate embeddings
        top_k: Number of top similar embeddings to return
        
    Returns:
        List of tuples (index, similarity_score) sorted by similarity
    """
    try:
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            if candidate.size > 0:
                similarity = cosine_similarity_single(query_embedding, candidate)
                similarities.append((i, similarity))
        
        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        logger.error(f"Error finding similar embeddings: {e}")
        return []