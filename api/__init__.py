"""
REST API Module

This module provides a FastAPI-based REST API for the Document Analyzer.
It exposes endpoints for:
- Document upload and management
- Semantic search
- System health and status

Usage:
    # Run directly
    uvicorn api:app --reload

    # Or import
    from api import app, router
"""

from .routes import router
from .main import app

__all__ = ["app", "router"]
