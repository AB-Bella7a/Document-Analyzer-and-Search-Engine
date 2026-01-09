"""
Configuration Module for Document Analyzer & Search Engine

This module centralizes all configuration settings for the application.
Settings are loaded from environment variables (via .env file) with sensible defaults
for local development.

Usage:
    from config import settings
    print(settings.DATA_DIR)
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


# Determine base directory at module level
_BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings have defaults suitable for local development.
    Override via .env file or environment variables for production.
    """

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    APP_NAME: str = "Document Analyzer & Search Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ==========================================================================
    # Directory Paths
    # ==========================================================================
    # Base directory is the project root
    BASE_DIR: Path = _BASE_DIR

    # Data directories for different stages of the pipeline
    # These will be set relative to BASE_DIR if not specified
    DATA_DIR: Optional[Path] = None
    RAW_DATA_DIR: Optional[Path] = None
    PROCESSED_DATA_DIR: Optional[Path] = None
    EMBEDDINGS_DIR: Optional[Path] = None
    UPLOADS_DIR: Optional[Path] = None

    # ==========================================================================
    # AWS S3 Configuration (Optional - for cloud storage)
    # ==========================================================================
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    # Enable/disable S3 storage (falls back to local if disabled or not configured)
    USE_S3_STORAGE: bool = False

    # ==========================================================================
    # Document Processing Settings
    # ==========================================================================
    # Supported file extensions
    SUPPORTED_PDF_EXTENSIONS: List[str] = [".pdf"]
    SUPPORTED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]

    # Maximum file size in bytes (default: 50MB)
    MAX_FILE_SIZE: int = 50 * 1024 * 1024

    # OCR settings
    TESSERACT_CMD: Optional[str] = None  # Path to tesseract executable if not in PATH
    OCR_LANGUAGE: str = "eng"  # Default OCR language

    # ==========================================================================
    # Embedding Model Settings
    # ==========================================================================
    # Text embedding model (sentence-transformers)
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TEXT_EMBEDDING_DIMENSION: int = 384

    # Image embedding model (CLIP)
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"
    IMAGE_EMBEDDING_DIMENSION: int = 512

    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE: int = 32

    # ==========================================================================
    # Search Settings
    # ==========================================================================
    # Default number of results to return
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100

    # FAISS index settings
    FAISS_INDEX_TYPE: str = "flat"  # Options: "flat", "ivf"
    FAISS_NLIST: int = 100  # Number of clusters for IVF index

    # ==========================================================================
    # API Settings
    # ==========================================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def set_directory_defaults(self) -> "Settings":
        """Set default directory paths relative to BASE_DIR."""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.RAW_DATA_DIR is None:
            self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        if self.PROCESSED_DATA_DIR is None:
            self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        if self.EMBEDDINGS_DIR is None:
            self.EMBEDDINGS_DIR = self.DATA_DIR / "embeddings"
        if self.UPLOADS_DIR is None:
            self.UPLOADS_DIR = self.DATA_DIR / "uploads"
        return self

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.UPLOADS_DIR,
        ]
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    def is_s3_configured(self) -> bool:
        """Check if AWS S3 is properly configured."""
        return (
            self.USE_S3_STORAGE
            and self.AWS_ACCESS_KEY_ID is not None
            and self.AWS_SECRET_ACCESS_KEY is not None
            and self.S3_BUCKET_NAME is not None
        )

    def get_supported_extensions(self) -> list:
        """Get all supported file extensions."""
        return self.SUPPORTED_PDF_EXTENSIONS + self.SUPPORTED_IMAGE_EXTENSIONS

    def is_pdf(self, filename: str) -> bool:
        """Check if a file is a PDF based on its extension."""
        return Path(filename).suffix.lower() in self.SUPPORTED_PDF_EXTENSIONS

    def is_image(self, filename: str) -> bool:
        """Check if a file is an image based on its extension."""
        return Path(filename).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS


# Create a global settings instance
# This is imported throughout the application
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()


if __name__ == "__main__":
    # Print current configuration for debugging
    print("=" * 60)
    print("Document Analyzer & Search Engine - Configuration")
    print("=" * 60)
    print(f"App Name: {settings.APP_NAME}")
    print(f"Version: {settings.APP_VERSION}")
    print(f"Debug Mode: {settings.DEBUG}")
    print()
    print("Directory Paths:")
    print(f"  Base: {settings.BASE_DIR}")
    print(f"  Data: {settings.DATA_DIR}")
    print(f"  Raw: {settings.RAW_DATA_DIR}")
    print(f"  Processed: {settings.PROCESSED_DATA_DIR}")
    print(f"  Embeddings: {settings.EMBEDDINGS_DIR}")
    print(f"  Uploads: {settings.UPLOADS_DIR}")
    print()
    print("AWS Configuration:")
    print(f"  S3 Enabled: {settings.USE_S3_STORAGE}")
    print(f"  S3 Configured: {settings.is_s3_configured()}")
    print(f"  Region: {settings.AWS_REGION}")
    print()
    print("Model Configuration:")
    print(f"  Text Model: {settings.TEXT_EMBEDDING_MODEL}")
    print(f"  Image Model: {settings.IMAGE_EMBEDDING_MODEL}")
