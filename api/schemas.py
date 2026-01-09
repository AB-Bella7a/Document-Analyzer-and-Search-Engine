"""
API Request and Response Schemas

This module defines Pydantic models for API request/response validation.
Using Pydantic ensures:
- Automatic request validation
- Clear API documentation
- Type safety

These schemas are used by FastAPI to:
1. Validate incoming requests
2. Serialize outgoing responses
3. Generate OpenAPI documentation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Document Schemas
# =============================================================================

class DocumentMetadataResponse(BaseModel):
    """Response schema for document metadata."""
    document_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type (pdf/image)")
    file_extension: str = Field(..., description="File extension")
    file_size_bytes: int = Field(..., description="File size in bytes")
    upload_timestamp: str = Field(..., description="Upload timestamp (ISO format)")
    storage_location: str = Field(..., description="Local storage path")
    s3_location: Optional[str] = Field(None, description="S3 URI if synced")
    checksum: str = Field(..., description="MD5 checksum")
    status: str = Field(..., description="Processing status")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123def456",
                "original_filename": "report.pdf",
                "document_type": "pdf",
                "file_extension": ".pdf",
                "file_size_bytes": 102400,
                "upload_timestamp": "2024-01-15T10:30:00",
                "storage_location": "/data/raw/pdfs/doc_abc123def456.pdf",
                "s3_location": None,
                "checksum": "d41d8cd98f00b204e9800998ecf8427e",
                "status": "ingested"
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    success: bool = Field(..., description="Whether upload was successful")
    document_id: str = Field(..., description="Assigned document ID")
    message: str = Field(..., description="Status message")
    metadata: DocumentMetadataResponse = Field(..., description="Document metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_abc123def456",
                "message": "Document uploaded and ingested successfully",
                "metadata": {
                    "document_id": "doc_abc123def456",
                    "original_filename": "report.pdf",
                    "document_type": "pdf",
                    "file_extension": ".pdf",
                    "file_size_bytes": 102400,
                    "upload_timestamp": "2024-01-15T10:30:00",
                    "storage_location": "/data/raw/pdfs/doc_abc123def456.pdf",
                    "s3_location": None,
                    "checksum": "d41d8cd98f00b204e9800998ecf8427e",
                    "status": "ingested"
                }
            }
        }


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""
    total_count: int = Field(..., description="Total number of documents")
    documents: List[DocumentMetadataResponse] = Field(..., description="List of documents")


class ProcessedDocumentResponse(BaseModel):
    """Response schema for processed document data."""
    document_id: str
    original_filename: str
    document_type: str
    cleaned_text: str = Field(..., description="Cleaned and normalized text")
    word_count: int
    character_count: int
    page_count: int
    processing_method: str
    processing_timestamp: str
    status: str


# =============================================================================
# Search Schemas
# =============================================================================

class SearchRequest(BaseModel):
    """Request schema for semantic search."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "top_k": 5,
                "min_score": 0.3
            }
        }


class SearchResultItem(BaseModel):
    """Schema for a single search result."""
    document_id: str = Field(..., description="Document identifier")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., description="Result ranking (1-based)")
    original_filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type")
    word_count: int = Field(..., description="Word count")
    text_preview: str = Field(..., description="Text preview (first 200 chars)")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123def456",
                "similarity_score": 0.8542,
                "rank": 1,
                "original_filename": "ml_report.pdf",
                "document_type": "pdf",
                "word_count": 1500,
                "text_preview": "machine learning is a subset of artificial intelligence..."
            }
        }


class SearchResponse(BaseModel):
    """Response schema for search results."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Number of results returned")
    results: List[SearchResultItem] = Field(..., description="Search results")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning",
                "total_results": 3,
                "results": [
                    {
                        "document_id": "doc_abc123",
                        "similarity_score": 0.85,
                        "rank": 1,
                        "original_filename": "ml_intro.pdf",
                        "document_type": "pdf",
                        "word_count": 1500,
                        "text_preview": "machine learning is..."
                    }
                ]
            }
        }


# =============================================================================
# Pipeline Schemas
# =============================================================================

class ProcessDocumentRequest(BaseModel):
    """Request schema for processing a document."""
    document_id: str = Field(..., description="Document ID to process")
    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_noise: bool = Field(default=True, description="Remove OCR noise")


class ProcessDocumentResponse(BaseModel):
    """Response schema for document processing."""
    success: bool
    document_id: str
    message: str
    word_count: Optional[int] = None
    status: Optional[str] = None


class GenerateEmbeddingsRequest(BaseModel):
    """Request schema for generating embeddings."""
    document_id: str = Field(..., description="Document ID")
    include_image: bool = Field(default=True, description="Include image embedding")


class GenerateEmbeddingsResponse(BaseModel):
    """Response schema for embedding generation."""
    success: bool
    document_id: str
    message: str
    text_dimension: Optional[int] = None
    image_dimension: Optional[int] = None


class PipelineRequest(BaseModel):
    """Request schema for running full pipeline."""
    document_id: Optional[str] = Field(None, description="Specific document ID")
    process_all: bool = Field(False, description="Process all pending documents")
    rebuild_index: bool = Field(False, description="Force rebuild search index")


class PipelineResponse(BaseModel):
    """Response schema for pipeline execution."""
    success: bool
    message: str
    documents_processed: int = 0
    documents_embedded: int = 0
    index_size: int = 0


# =============================================================================
# System Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics."""
    total_documents: int = Field(..., description="Total ingested documents")
    processed_documents: int = Field(..., description="Processed documents")
    embedded_documents: int = Field(..., description="Documents with embeddings")
    index_size: int = Field(..., description="FAISS index size")
    s3_enabled: bool = Field(..., description="S3 storage enabled")
    models: Dict[str, str] = Field(..., description="Model configurations")

    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 100,
                "processed_documents": 95,
                "embedded_documents": 90,
                "index_size": 90,
                "s3_enabled": False,
                "models": {
                    "text_embedding": "all-MiniLM-L6-v2",
                    "image_embedding": "openai/clip-vit-base-patch32"
                }
            }
        }


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "NotFoundError",
                "message": "Document not found: doc_xyz123",
                "details": {"document_id": "doc_xyz123"}
            }
        }
