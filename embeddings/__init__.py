"""
Document Embeddings Module

This module handles the generation of vector embeddings for documents.
It provides functionality to:
- Generate text embeddings using sentence-transformers
- Generate image embeddings using CLIP/ViT models
- Store and load embeddings efficiently
- Support batch processing

Usage:
    from embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator()
    text_embedding = generator.embed_text("Hello world")
    image_embedding = generator.embed_image("path/to/image.png")
"""

from .embedding_generator import (
    EmbeddingGenerator,
    TextEmbedder,
    ImageEmbedder,
    DocumentEmbedding
)

__all__ = [
    "EmbeddingGenerator",
    "TextEmbedder",
    "ImageEmbedder",
    "DocumentEmbedding"
]
