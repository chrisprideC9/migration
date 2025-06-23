# src/utils/file_utils.py

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 100) -> Tuple[bool, str]:
    """
    Validate that a file is within the maximum size limit.
    
    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum file size in megabytes
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB)"
        
        return True, f"File size is valid ({file_size_mb:.1f} MB)"
        
    except FileNotFoundError:
        return False, "File not found"
    except Exception as e:
        return False, f"Error validating file size: {str(e)}"

def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension including the dot (e.g., '.csv', '.zip')
    """
    return os.path.splitext(filename)[1].lower()

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for file system use
    """
    # Remove/replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized

def create_backup_filename(original_filename: str) -> str:
    """
    Create a backup filename with timestamp.
    
    Args:
        original_filename: Original filename
        
    Returns:
        Backup filename with timestamp
    """
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_backup_{timestamp}{ext}"

def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def get_safe_filename(base_name: str, directory: Union[str, Path], extension: str = "") -> str:
    """
    Get a safe filename that doesn't conflict with existing files.
    
    Args:
        base_name: Base name for the file
        directory: Directory where the file will be saved
        extension: File extension (including dot)
        
    Returns:
        Safe filename that doesn't exist
    """
    base_name = sanitize_filename(base_name)
    directory = Path(directory)
    
    # Add extension if provided
    if extension and not base_name.endswith(extension):
        base_name += extension
    
    filename = base_name
    counter = 1
    
    while (directory / filename).exists():
        name, ext = os.path.splitext(base_name)
        filename = f"{name}_{counter}{ext}"
        counter += 1
    
    return filename

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def read_file_safely(file_path: Union[str, Path], encoding: str = 'utf-8') -> Tuple[bool, Union[str, Exception]]:
    """
    Safely read a text file with error handling.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        Tuple of (success, content_or_error)
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return True, content
    except Exception as e:
        return False, e

def write_file_safely(file_path: Union[str, Path], content: str, encoding: str = 'utf-8', create_backup: bool = False) -> Tuple[bool, str]:
    """
    Safely write content to a file with error handling.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        create_backup: Whether to create a backup if file exists
        
    Returns:
        Tuple of (success, message)
    """
    try:
        file_path = Path(file_path)
        
        # Create backup if requested and file exists
        if create_backup and file_path.exists():
            backup_path = file_path.parent / create_backup_filename(file_path.name)
            file_path.rename(backup_path)
        
        # Ensure directory exists
        ensure_directory_exists(file_path.parent)
        
        # Write the file
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True, f"File written successfully to {file_path}"
        
    except Exception as e:
        return False, f"Error writing file: {str(e)}"

def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get comprehensive information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'extension': file_path.suffix.lower(),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': file_path.is_file(),
            'is_directory': file_path.is_dir(),
            'exists': file_path.exists(),
            'absolute_path': str(file_path.absolute()),
            'parent_directory': str(file_path.parent)
        }
    except Exception as e:
        return {
            'error': str(e),
            'exists': False
        }