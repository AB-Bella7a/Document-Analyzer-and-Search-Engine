"""
FastAPI Application for Document Analyzer

This is the main application entry point that configures
and starts the FastAPI server.

Features:
- CORS middleware for cross-origin requests
- OpenAPI documentation (Swagger UI)
- Health checks and system monitoring
- Document management and search APIs

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or using the main.py script:
    python main.py
"""

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from api.routes import router

# =============================================================================
# Create FastAPI Application
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Document Analyzer & Search Engine

A powerful document analysis system that combines:
- **Document Ingestion**: Upload and store PDFs and images
- **Text Extraction**: Extract text using pdfplumber (PDFs) and Tesseract OCR (images)
- **AI Embeddings**: Generate semantic embeddings using sentence-transformers and CLIP
- **Semantic Search**: Find documents using natural language queries

### Features
- Local-first development with optional AWS S3 storage
- Modular pipeline architecture
- RESTful API with comprehensive documentation

### Quick Start
1. Upload a document: `POST /api/v1/documents/upload`
2. Process it: `POST /api/v1/pipeline/run`
3. Search: `POST /api/v1/search`

Or use the all-in-one endpoint: `POST /api/v1/documents/upload-and-process`
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =============================================================================
# Configure CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include API Routes
# =============================================================================

# Mount all routes under /api/v1 prefix
app.include_router(router, prefix="/api/v1")

# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api", include_in_schema=False)
async def api_root():
    """API information endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }


# =============================================================================
# Application Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks.

    - Ensure directories exist
    - Log configuration
    """
    print("=" * 60)
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 60)

    # Ensure all directories exist
    settings.ensure_directories()

    print(f"Data directory: {settings.DATA_DIR}")
    print(f"S3 enabled: {settings.is_s3_configured()}")
    print(f"API running at: http://{settings.API_HOST}:{settings.API_PORT}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks.

    Clean up resources as needed.
    """
    print("Shutting down Document Analyzer...")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
