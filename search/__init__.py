"""
Semantic Search Module

This module provides semantic search functionality using FAISS
for efficient vector similarity search.

Features:
- Build and manage FAISS indices
- Perform semantic search with natural language queries
- Return ranked results with similarity scores

Usage:
    from search import SemanticSearch

    search_engine = SemanticSearch()
    search_engine.build_index()
    results = search_engine.search("What is machine learning?", top_k=5)
"""

from .semantic_search import SemanticSearch, SearchResult

__all__ = ["SemanticSearch", "SearchResult"]
