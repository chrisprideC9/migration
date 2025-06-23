# src/utils/__init__.py

"""Utility functions for URL processing, file handling, and embeddings."""

from src.utils.url_utils import (
    extract_path,
    normalize_url,
    is_valid_url,
    get_domain,
    urls_same_domain,
    clean_url_for_comparison
)
from src.utils.file_utils import (
    validate_file_size,
    get_file_extension,
    sanitize_filename,
    create_backup_filename
)
from src.utils.embedding_utils import (
    string_to_embedding,
    embedding_to_string,
    validate_embedding,
    cosine_similarity_single,
    load_sentence_transformer
)

__all__ = [
    # URL utilities
    "extract_path",
    "normalize_url", 
    "is_valid_url",
    "get_domain",
    "urls_same_domain",
    "clean_url_for_comparison",
    # File utilities
    "validate_file_size",
    "get_file_extension",
    "sanitize_filename",
    "create_backup_filename",
    # Embedding utilities
    "string_to_embedding",
    "embedding_to_string",
    "validate_embedding",
    "cosine_similarity_single",
    "load_sentence_transformer"
]