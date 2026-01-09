"""
Tests for Document Ingestion Module

Tests cover:
- File validation
- Document type detection
- Metadata generation
- Local storage
- Batch ingestion
"""

import pytest
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion import DocumentIngester, DocumentMetadata


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata with required fields."""
        metadata = DocumentMetadata(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            file_extension=".pdf",
            file_size_bytes=1024,
            upload_timestamp="2024-01-01T00:00:00",
            storage_location="/path/to/file.pdf"
        )

        assert metadata.document_id == "doc_test123"
        assert metadata.original_filename == "test.pdf"
        assert metadata.document_type == "pdf"
        assert metadata.status == "ingested"

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = DocumentMetadata(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            file_extension=".pdf",
            file_size_bytes=1024,
            upload_timestamp="2024-01-01T00:00:00",
            storage_location="/path/to/file.pdf"
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["document_id"] == "doc_test123"
        assert "checksum" in data

    def test_metadata_json_serialization(self):
        """Test JSON serialization/deserialization."""
        metadata = DocumentMetadata(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            file_extension=".pdf",
            file_size_bytes=1024,
            upload_timestamp="2024-01-01T00:00:00",
            storage_location="/path/to/file.pdf"
        )

        json_str = metadata.to_json()
        restored = DocumentMetadata.from_json(json_str)

        assert restored.document_id == metadata.document_id
        assert restored.original_filename == metadata.original_filename


class TestDocumentIngester:
    """Tests for DocumentIngester class."""

    def test_ingester_initialization(self, test_data_dir, mock_settings):
        """Test ingester initializes correctly."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        assert ingester.raw_dir.exists()
        assert (ingester.raw_dir / "pdfs").exists()
        assert (ingester.raw_dir / "images").exists()
        assert (ingester.raw_dir / "metadata").exists()

    def test_document_type_detection_pdf(self, test_data_dir, mock_settings):
        """Test PDF detection."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        assert ingester._get_document_type("test.pdf") == "pdf"
        assert ingester._get_document_type("TEST.PDF") == "pdf"

    def test_document_type_detection_image(self, test_data_dir, mock_settings):
        """Test image detection."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        assert ingester._get_document_type("test.png") == "image"
        assert ingester._get_document_type("test.jpg") == "image"
        assert ingester._get_document_type("test.jpeg") == "image"

    def test_document_type_detection_unsupported(self, test_data_dir, mock_settings):
        """Test unsupported file type raises error."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        with pytest.raises(ValueError):
            ingester._get_document_type("test.xyz")

    def test_generate_document_id(self, test_data_dir, mock_settings):
        """Test document ID generation."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        id1 = ingester._generate_document_id()
        id2 = ingester._generate_document_id()

        assert id1.startswith("doc_")
        assert id2.startswith("doc_")
        assert id1 != id2  # IDs should be unique

    def test_validate_file_not_found(self, test_data_dir, mock_settings):
        """Test validation fails for non-existent file."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        with pytest.raises(FileNotFoundError):
            ingester._validate_file(Path("/nonexistent/file.pdf"))

    def test_validate_empty_file(self, test_data_dir, mock_settings):
        """Test validation fails for empty file."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        # Create empty file
        empty_file = test_data_dir / "empty.pdf"
        empty_file.touch()

        with pytest.raises(ValueError, match="empty"):
            ingester._validate_file(empty_file)

    def test_ingest_pdf(self, test_data_dir, sample_pdf, mock_settings):
        """Test ingesting a PDF file."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        metadata = ingester.ingest_document(sample_pdf)

        assert metadata.document_id.startswith("doc_")
        assert metadata.document_type == "pdf"
        assert metadata.original_filename == "sample.pdf"
        assert metadata.status == "ingested"
        assert Path(metadata.storage_location).exists()

    def test_ingest_image(self, test_data_dir, sample_image, mock_settings):
        """Test ingesting an image file."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        metadata = ingester.ingest_document(sample_image)

        assert metadata.document_id.startswith("doc_")
        assert metadata.document_type == "image"
        assert metadata.original_filename == "sample.png"

    def test_ingest_from_bytes(self, test_data_dir, mock_settings):
        """Test ingesting from bytes."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        # Create simple content
        content = b"%PDF-1.4\n%Test PDF content"

        metadata = ingester.ingest_from_bytes(content, "test_bytes.pdf")

        assert metadata.document_id.startswith("doc_")
        assert metadata.document_type == "pdf"

    def test_get_metadata(self, test_data_dir, sample_pdf, mock_settings):
        """Test retrieving document metadata."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        # Ingest a document
        metadata = ingester.ingest_document(sample_pdf)

        # Retrieve metadata
        retrieved = ingester.get_metadata(metadata.document_id)

        assert retrieved is not None
        assert retrieved.document_id == metadata.document_id
        assert retrieved.original_filename == metadata.original_filename

    def test_get_metadata_not_found(self, test_data_dir, mock_settings):
        """Test retrieving non-existent metadata returns None."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        result = ingester.get_metadata("nonexistent_doc")
        assert result is None

    def test_list_documents(self, test_data_dir, sample_pdf, mock_settings):
        """Test listing all documents."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        # Ingest a document
        ingester.ingest_document(sample_pdf)

        # List documents
        documents = ingester.list_documents()

        assert len(documents) == 1
        assert documents[0].original_filename == "sample.pdf"

    def test_delete_document(self, test_data_dir, sample_pdf, mock_settings):
        """Test deleting a document."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        # Ingest a document
        metadata = ingester.ingest_document(sample_pdf)

        # Verify it exists
        assert ingester.get_metadata(metadata.document_id) is not None

        # Delete it
        result = ingester.delete_document(metadata.document_id)

        assert result is True
        assert ingester.get_metadata(metadata.document_id) is None

    def test_batch_ingestion(self, test_data_dir, sample_pdf, sample_image, mock_settings):
        """Test batch ingestion of multiple files."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        files = [sample_pdf, sample_image]
        results = ingester.ingest_batch(files)

        assert len(results["successful"]) == 2
        assert len(results["failed"]) == 0

    def test_checksum_calculation(self, test_data_dir, sample_pdf, mock_settings):
        """Test file checksum is calculated correctly."""
        ingester = DocumentIngester(raw_dir=test_data_dir / "raw")

        checksum1 = ingester._calculate_checksum(sample_pdf)
        checksum2 = ingester._calculate_checksum(sample_pdf)

        assert checksum1 == checksum2
        assert len(checksum1) == 32  # MD5 hash length
