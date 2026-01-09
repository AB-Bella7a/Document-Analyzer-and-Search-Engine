"""
Document Ingestion Module

This module handles the intake of documents (PDFs and images) into the system.
It provides functionality to:
- Upload documents locally
- Store document metadata
- Optionally sync documents to AWS S3

Usage:
    from ingestion import DocumentIngester

    ingester = DocumentIngester()
    result = ingester.ingest_document("path/to/document.pdf")
"""

from .ingestion import DocumentIngester, DocumentMetadata

__all__ = ["DocumentIngester", "DocumentMetadata"]
