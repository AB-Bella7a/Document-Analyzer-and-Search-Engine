# Document Analyzer & Search Engine

A Python-based document analysis and semantic search system that demonstrates cloud services, data pipelines, computer vision, AI/ML, and REST APIs. This project ingests documents (PDFs and images), extracts and preprocesses content, generates AI embeddings, and provides semantic search capabilities through a REST API.

## Features

- **Document Ingestion**: Upload and manage PDFs and images with metadata tracking
- **Text Extraction**: Extract text from PDFs using pdfplumber and OCR from images using Tesseract
- **AI Embeddings**: Generate semantic embeddings using sentence-transformers and CLIP models
- **Semantic Search**: Find documents using natural language queries with FAISS vector search
- **REST API**: Full-featured FastAPI endpoints for all operations
- **Cloud Storage**: Optional AWS S3 integration for scalable storage
- **Modular Architecture**: Clean separation of concerns for maintainability

### Pipeline Stages

1. **Ingestion**: Documents are uploaded, validated, and stored with metadata
2. **Preprocessing**: Text is extracted (PDF/OCR) and cleaned/normalized
3. **Embedding**: AI models generate vector representations for semantic understanding
4. **Search**: FAISS indexes enable fast similarity search with natural language queries

## Project Structure

```
Document-Analyzer-and-Search-Engine/
├── api/                    # FastAPI REST API
│   ├── __init__.py
│   ├── main.py            # Application entry point
│   ├── routes.py          # API endpoints
│   └── schemas.py         # Request/response models
├── ingestion/             # Document ingestion pipeline
│   ├── __init__.py
│   └── ingestion.py       # File validation, storage, metadata
├── preprocessing/         # Text extraction and cleaning
│   ├── __init__.py
│   └── preprocessor.py    # PDF/OCR extraction, text cleaning
├── embeddings/            # AI embedding generation
│   ├── __init__.py
│   └── embedding_generator.py  # Text and image embeddings
├── search/                # Semantic search
│   ├── __init__.py
│   └── semantic_search.py # FAISS indexing and search
├── storage/               # Storage abstraction
│   ├── __init__.py
│   └── storage_manager.py # Local and S3 storage
├── scripts/               # Utility scripts
│   ├── run_pipeline.py    # Run full processing pipeline
│   ├── ingest_documents.py # Batch document ingestion
│   └── search_cli.py      # Command-line search interface
├── tests/                 # Test suite
│   ├── conftest.py        # Test fixtures
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_search.py
│   └── test_api.py
├── data/                  # Data directories (auto-created)
│   ├── raw/               # Ingested documents
│   ├── processed/         # Extracted text
│   ├── embeddings/        # Vector embeddings
│   └── uploads/           # Temporary uploads
├── config.py              # Configuration management
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR (for image text extraction)
- pip (Python package manager)

### Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Setup Project

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Document-Analyzer-and-Search-Engine.git
cd Document-Analyzer-and-Search-Engine
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings (optional for local development)
```

5. **Verify installation:**
```bash
python main.py status
```

## Quick Start

### 1. Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### 2. Upload a Document

Using curl:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

Or using the API docs at `http://localhost:8000/docs`.

### 3. Process the Document

Run the pipeline to extract text and generate embeddings:
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{"process_all": true}'
```

### 4. Search Documents

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "top_k": 5}'
```

## Usage

### Command Line Interface

The project includes a main entry point with multiple commands:

```bash
# Start API server (default)
python main.py

# Run processing pipeline
python main.py pipeline

# Interactive search
python main.py search

# Ingest documents
python main.py ingest ./documents/

# Show system status
python main.py status
```

### Utility Scripts

**Ingest documents from a directory:**
```bash
python scripts/ingest_documents.py ./documents/ --recursive --process
```

**Run the full pipeline:**
```bash
python scripts/run_pipeline.py --rebuild
```

**Interactive search:**
```bash
python scripts/search_cli.py
```

**Single query search:**
```bash
python scripts/search_cli.py "What is machine learning?"
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/stats` | GET | System statistics |
| `/api/v1/documents` | GET | List all documents |
| `/api/v1/documents/upload` | POST | Upload a document |
| `/api/v1/documents/{id}` | GET | Get document metadata |
| `/api/v1/documents/{id}` | DELETE | Delete a document |
| `/api/v1/documents/{id}/processed` | GET | Get processed content |
| `/api/v1/search` | POST | Semantic search |
| `/api/v1/search` | GET | Search (query params) |
| `/api/v1/search/similar/{id}` | GET | Find similar documents |
| `/api/v1/pipeline/process` | POST | Process a document |
| `/api/v1/pipeline/embed` | POST | Generate embeddings |
| `/api/v1/pipeline/build-index` | POST | Build search index |
| `/api/v1/pipeline/run` | POST | Run full pipeline |
| `/api/v1/documents/upload-and-process` | POST | Upload and fully process |

### Python API

```python
from ingestion import DocumentIngester
from preprocessing import DocumentPreprocessor
from embeddings import EmbeddingGenerator
from search import SemanticSearch

# Ingest a document
ingester = DocumentIngester()
metadata = ingester.ingest_document("document.pdf")

# Process the document
preprocessor = DocumentPreprocessor()
processed = preprocessor.process_document(metadata.document_id)

# Generate embeddings
embedding_gen = EmbeddingGenerator()
embedding = embedding_gen.generate_document_embedding(metadata.document_id)

# Search
search_engine = SemanticSearch()
search_engine.build_index()
results = search_engine.search("your query", top_k=5)

for result in results:
    print(f"{result.rank}. {result.original_filename}: {result.similarity_score:.3f}")
```

## Configuration

Configuration is managed through environment variables (`.env` file) and the `config.py` module.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `True` | Enable debug mode |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `USE_S3_STORAGE` | `False` | Enable AWS S3 storage |
| `AWS_ACCESS_KEY_ID` | - | AWS access key (if using S3) |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key (if using S3) |
| `S3_BUCKET_NAME` | - | S3 bucket name |
| `TEXT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Text embedding model |
| `IMAGE_EMBEDDING_MODEL` | `openai/clip-vit-base-patch32` | Image embedding model |
| `TESSERACT_CMD` | - | Path to Tesseract executable |

### AWS S3 Setup (Optional)

To enable S3 storage:

1. Create an S3 bucket in AWS
2. Configure credentials in `.env`:
```env
USE_S3_STORAGE=True
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

## AI Models

### Text Embeddings

- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension**: 384
- **Use case**: Semantic text search

Alternative models:
- `all-mpnet-base-v2`: Higher quality, slower
- `paraphrase-MiniLM-L6-v2`: Good for paraphrase detection

### Image Embeddings

- **Model**: `openai/clip-vit-base-patch32` (CLIP)
- **Dimension**: 512
- **Use case**: Image understanding and cross-modal search

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_ingestion.py

# Run with coverage
pytest tests/ --cov=.
```

## Extending the System

### Adding a New Document Type

1. Update `config.py` with new supported extensions
2. Add extraction logic in `preprocessing/preprocessor.py`
3. Update validation in `ingestion/ingestion.py`

### Fine-tuning Embeddings

The embedding module is structured to support fine-tuning:

```python
from embeddings import TextEmbedder

# Custom model
embedder = TextEmbedder(model_name="your-finetuned-model")
```

### Custom Storage Backend

Implement the `StorageBackend` interface:

```python
from storage.storage_manager import StorageBackend

class CustomStorage(StorageBackend):
    def upload_file(self, local_path, remote_path, metadata=None):
        # Your implementation
        pass
    # ... implement other methods
```

## Performance Considerations

- **Batch Processing**: Use batch methods for multiple documents
- **Index Type**: Use IVF index for large document collections (>10k)
- **Embedding Cache**: Embeddings are persisted to avoid regeneration
- **Lazy Loading**: Models are loaded on first use

## Troubleshooting

### Common Issues

**Tesseract not found:**
```bash
# Check if installed
tesseract --version

# Set path in .env
TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

**CUDA/GPU issues:**
The system uses CPU by default. For GPU acceleration, install PyTorch with CUDA support.

**Memory issues with large files:**
Adjust `MAX_FILE_SIZE` in configuration or process files in chunks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for text embeddings
- [CLIP](https://openai.com/research/clip) for image understanding
- [FAISS](https://faiss.ai/) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image text extraction
