"""
Storage Module

This module provides unified storage abstraction supporting:
- Local filesystem storage (default)
- AWS S3 cloud storage (optional)

The storage module allows seamless switching between local
and cloud storage without changing application code.

Usage:
    from storage import StorageManager

    storage = StorageManager()
    storage.upload_file(local_path, "documents/file.pdf")
    storage.download_file("documents/file.pdf", local_path)
"""

from .storage_manager import StorageManager, LocalStorage, S3Storage

__all__ = ["StorageManager", "LocalStorage", "S3Storage"]
