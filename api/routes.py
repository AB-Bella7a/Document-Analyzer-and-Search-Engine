"""
API Routes for Document Analyzer

This module defines all REST API endpoints:
- Document management (upload, list, get, delete)
- Search functionality
- Pipeline operations
- System health and stats

The API orchestrates the various pipeline components
(ingestion, preprocessing, embeddings, search) rather than
duplicating their logic.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import sys

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from api.schemas import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    ProcessedDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
    PipelineRequest,
    PipelineResponse,
    HealthResponse,
    SystemStatsResponse,
    ErrorResponse
)

# Import pipeline components
from ingestion import DocumentIngester
from preprocessing import DocumentPreprocessor
from embeddings import EmbeddingGenerator
from search import SemanticSearch

# Create router
router = APIRouter()

# Initialize pipeline components (lazy loading via properties)
_ingester: Optional[DocumentIngester] = None
_preprocessor: Optional[DocumentPreprocessor] = None
_embedding_generator: Optional[EmbeddingGenerator] = None
_search_engine: Optional[SemanticSearch] = None


def get_ingester() -> DocumentIngester:
    """Get or create document ingester instance."""
    global _ingester
    if _ingester is None:
        _ingester = DocumentIngester()
    return _ingester


def get_preprocessor() -> DocumentPreprocessor:
    """Get or create preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = DocumentPreprocessor()
    return _preprocessor


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def get_search_engine() -> SemanticSearch:
    """Get or create search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SemanticSearch()
    return _search_engine


# =============================================================================
# Health & System Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check if the API is running and healthy"
)
async def health_check():
    """
    Health check endpoint.

    Returns basic system status information.
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    tags=["System"],
    summary="System statistics",
    description="Get statistics about the document analyzer system"
)
async def get_system_stats():
    """
    Get system statistics.

    Returns counts of documents at each pipeline stage
    and configuration information.
    """
    ingester = get_ingester()
    preprocessor = get_preprocessor()
    embedding_gen = get_embedding_generator()
    search = get_search_engine()

    # Count documents at each stage
    total_docs = len(ingester.list_documents())
    processed_docs = len(preprocessor.list_processed())
    embedded_docs = len(embedding_gen.list_embeddings())

    # Get search index size
    index_stats = search.get_index_stats()
    index_size = index_stats.get("total_documents", 0)

    return SystemStatsResponse(
        total_documents=total_docs,
        processed_documents=processed_docs,
        embedded_documents=embedded_docs,
        index_size=index_size,
        s3_enabled=settings.is_s3_configured(),
        models={
            "text_embedding": settings.TEXT_EMBEDDING_MODEL,
            "image_embedding": settings.IMAGE_EMBEDDING_MODEL
        }
    )


# =============================================================================
# Document Management Endpoints
# =============================================================================

@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload a document",
    description="Upload a PDF or image file for analysis"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF or image)")
):
    """
    Upload a new document.

    Accepts PDF files and images (PNG, JPG, JPEG, TIFF, BMP, GIF).
    The document will be ingested and stored for further processing.

    Returns document metadata including the assigned document ID.
    """
    # Validate file type
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower()

    if extension not in settings.get_supported_extensions():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension}. "
                   f"Supported: {settings.get_supported_extensions()}"
        )

    # Read file contents
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}"
        )

    # Check file size
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
        )

    # Ingest document
    try:
        ingester = get_ingester()
        metadata = ingester.ingest_from_bytes(contents, filename)

        return DocumentUploadResponse(
            success=True,
            document_id=metadata.document_id,
            message="Document uploaded and ingested successfully",
            metadata=DocumentMetadataResponse(**metadata.to_dict())
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all documents",
    description="Get a list of all ingested documents"
)
async def list_documents():
    """
    List all ingested documents.

    Returns metadata for all documents in the system,
    sorted by upload timestamp (newest first).
    """
    ingester = get_ingester()
    documents = ingester.list_documents()

    return DocumentListResponse(
        total_count=len(documents),
        documents=[DocumentMetadataResponse(**doc.to_dict()) for doc in documents]
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentMetadataResponse,
    tags=["Documents"],
    summary="Get document metadata",
    description="Get metadata for a specific document"
)
async def get_document(document_id: str):
    """
    Get document metadata by ID.

    Returns the metadata for a specific ingested document.
    """
    ingester = get_ingester()
    metadata = ingester.get_metadata(document_id)

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )

    return DocumentMetadataResponse(**metadata.to_dict())


@router.get(
    "/documents/{document_id}/processed",
    response_model=ProcessedDocumentResponse,
    tags=["Documents"],
    summary="Get processed document",
    description="Get the processed (extracted and cleaned) document content"
)
async def get_processed_document(document_id: str):
    """
    Get processed document data.

    Returns the extracted and cleaned text content
    for a processed document.
    """
    preprocessor = get_preprocessor()
    processed = preprocessor.get_processed(document_id)

    if not processed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processed document not found: {document_id}. "
                   "Document may not have been processed yet."
        )

    return ProcessedDocumentResponse(
        document_id=processed.document_id,
        original_filename=processed.original_filename,
        document_type=processed.document_type,
        cleaned_text=processed.cleaned_text,
        word_count=processed.word_count,
        character_count=processed.character_count,
        page_count=processed.page_count,
        processing_method=processed.processing_method,
        processing_timestamp=processed.processing_timestamp,
        status=processed.status
    )


@router.delete(
    "/documents/{document_id}",
    tags=["Documents"],
    summary="Delete a document",
    description="Delete a document and all its processed data"
)
async def delete_document(document_id: str):
    """
    Delete a document.

    Removes the document and all associated data
    (raw file, processed data, embeddings).
    """
    ingester = get_ingester()

    if not ingester.delete_document(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )

    return {"success": True, "message": f"Document {document_id} deleted"}


# =============================================================================
# Search Endpoints
# =============================================================================

@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search",
    description="Search documents using natural language queries"
)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search.

    Uses AI embeddings to find documents semantically similar
    to the query, even if they don't contain the exact words.

    Returns ranked results with similarity scores.
    """
    search = get_search_engine()

    try:
        results = search.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=[
                SearchResultItem(
                    document_id=r.document_id,
                    similarity_score=r.similarity_score,
                    rank=r.rank,
                    original_filename=r.original_filename,
                    document_type=r.document_type,
                    word_count=r.word_count,
                    text_preview=r.text_preview
                )
                for r in results
            ]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search (GET)",
    description="Search documents using query parameters"
)
async def semantic_search_get(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=100, description="Number of results"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum score")
):
    """
    Perform semantic search using GET parameters.

    Alternative to POST endpoint for simple queries.
    """
    request = SearchRequest(query=q, top_k=top_k, min_score=min_score)
    return await semantic_search(request)


@router.get(
    "/search/similar/{document_id}",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Find similar documents",
    description="Find documents similar to a given document"
)
async def find_similar_documents(
    document_id: str,
    top_k: int = Query(5, ge=1, le=100, description="Number of results")
):
    """
    Find documents similar to a given document.

    Uses the document's embedding to find other
    semantically similar documents.
    """
    search = get_search_engine()

    try:
        results = search.search_by_document(document_id, top_k=top_k)

        return SearchResponse(
            query=f"similar_to:{document_id}",
            total_results=len(results),
            results=[
                SearchResultItem(
                    document_id=r.document_id,
                    similarity_score=r.similarity_score,
                    rank=r.rank,
                    original_filename=r.original_filename,
                    document_type=r.document_type,
                    word_count=r.word_count,
                    text_preview=r.text_preview
                )
                for r in results
            ]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# =============================================================================
# Pipeline Endpoints
# =============================================================================

@router.post(
    "/pipeline/process",
    response_model=ProcessDocumentResponse,
    tags=["Pipeline"],
    summary="Process a document",
    description="Extract and clean text from an ingested document"
)
async def process_document(request: ProcessDocumentRequest):
    """
    Process a document.

    Extracts text (from PDF or via OCR for images)
    and cleans/normalizes the content.
    """
    preprocessor = get_preprocessor()

    try:
        result = preprocessor.process_document(
            document_id=request.document_id,
            lowercase=request.lowercase,
            remove_noise=request.remove_noise
        )

        return ProcessDocumentResponse(
            success=result.status in ["success", "partial"],
            document_id=result.document_id,
            message=f"Document processed: {result.word_count} words extracted",
            word_count=result.word_count,
            status=result.status
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@router.post(
    "/pipeline/embed",
    response_model=GenerateEmbeddingsResponse,
    tags=["Pipeline"],
    summary="Generate embeddings",
    description="Generate AI embeddings for a processed document"
)
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    """
    Generate embeddings for a document.

    Creates vector embeddings for the document's text
    (and optionally images) for semantic search.
    """
    embedding_gen = get_embedding_generator()

    try:
        result = embedding_gen.generate_document_embedding(
            document_id=request.document_id,
            include_image=request.include_image
        )

        return GenerateEmbeddingsResponse(
            success=True,
            document_id=result.document_id,
            message="Embeddings generated successfully",
            text_dimension=result.text_dimension,
            image_dimension=result.image_dimension
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post(
    "/pipeline/build-index",
    tags=["Pipeline"],
    summary="Build search index",
    description="Build or rebuild the FAISS search index"
)
async def build_search_index(force_rebuild: bool = False):
    """
    Build the search index.

    Creates a FAISS index from all document embeddings
    for efficient semantic search.
    """
    search = get_search_engine()

    try:
        count = search.build_index(force_rebuild=force_rebuild)

        return {
            "success": True,
            "message": f"Search index built with {count} documents",
            "index_size": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index building failed: {str(e)}"
        )


@router.post(
    "/pipeline/run",
    response_model=PipelineResponse,
    tags=["Pipeline"],
    summary="Run full pipeline",
    description="Run the full processing pipeline for documents"
)
async def run_pipeline(request: PipelineRequest):
    """
    Run the full processing pipeline.

    Processes documents through all stages:
    1. Preprocessing (text extraction)
    2. Embedding generation
    3. Search index building

    Can process a specific document or all pending documents.
    """
    preprocessor = get_preprocessor()
    embedding_gen = get_embedding_generator()
    search = get_search_engine()

    docs_processed = 0
    docs_embedded = 0

    try:
        if request.document_id:
            # Process specific document
            try:
                preprocessor.process_document(request.document_id)
                docs_processed = 1
            except Exception:
                pass

            try:
                embedding_gen.generate_document_embedding(request.document_id)
                docs_embedded = 1
            except Exception:
                pass

        elif request.process_all:
            # Process all pending
            process_results = preprocessor.process_all_pending()
            docs_processed = len(process_results.get("successful", []))

            embed_results = embedding_gen.generate_all_embeddings()
            docs_embedded = len(embed_results.get("successful", []))

        # Build/rebuild index
        index_size = search.build_index(force_rebuild=request.rebuild_index)

        return PipelineResponse(
            success=True,
            message="Pipeline completed successfully",
            documents_processed=docs_processed,
            documents_embedded=docs_embedded,
            index_size=index_size
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )


# =============================================================================
# Full Document Processing (Upload + Pipeline)
# =============================================================================

@router.post(
    "/documents/upload-and-process",
    tags=["Documents"],
    summary="Upload and fully process document",
    description="Upload a document and run it through the full pipeline"
)
async def upload_and_process(
    file: UploadFile = File(..., description="Document file")
):
    """
    Upload and fully process a document.

    This convenience endpoint:
    1. Uploads and ingests the document
    2. Extracts and cleans text
    3. Generates embeddings
    4. Updates the search index

    Returns the document ID and processing results.
    """
    # First, upload the document
    upload_response = await upload_document(file)

    if not upload_response.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )

    document_id = upload_response.document_id
    results = {
        "document_id": document_id,
        "upload": "success",
        "processing": None,
        "embedding": None,
        "index": None
    }

    # Process the document
    try:
        preprocessor = get_preprocessor()
        processed = preprocessor.process_document(document_id)
        results["processing"] = {
            "status": processed.status,
            "word_count": processed.word_count
        }
    except Exception as e:
        results["processing"] = {"status": "failed", "error": str(e)}

    # Generate embeddings
    try:
        embedding_gen = get_embedding_generator()
        embedding = embedding_gen.generate_document_embedding(document_id)
        results["embedding"] = {
            "status": "success",
            "text_dimension": embedding.text_dimension
        }
    except Exception as e:
        results["embedding"] = {"status": "failed", "error": str(e)}

    # Update search index
    try:
        search = get_search_engine()
        index_size = search.build_index(force_rebuild=True)
        results["index"] = {"status": "success", "size": index_size}
    except Exception as e:
        results["index"] = {"status": "failed", "error": str(e)}

    return results
