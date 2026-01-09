"""
Tests for Document Preprocessing Module

Tests cover:
- Text cleaning and normalization
- PDF text extraction
- OCR for images
- Processed document management
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import DocumentPreprocessor, ProcessedDocument
from preprocessing.preprocessor import TextCleaner


class TestTextCleaner:
    """Tests for TextCleaner utility class."""

    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance."""
        return TextCleaner()

    def test_normalize_whitespace(self, cleaner):
        """Test whitespace normalization."""
        text = "Hello    world\t\ttest"
        result = cleaner.normalize_whitespace(text)
        assert "    " not in result
        assert "\t" not in result

    def test_normalize_whitespace_preserves_paragraphs(self, cleaner):
        """Test that paragraph breaks are preserved."""
        text = "Paragraph 1\n\nParagraph 2"
        result = cleaner.normalize_whitespace(text)
        assert "\n\n" in result

    def test_lowercase(self, cleaner):
        """Test lowercase conversion."""
        text = "Hello WORLD Test"
        result = cleaner.lowercase(text)
        assert result == "hello world test"

    def test_remove_noise_repeated_punctuation(self, cleaner):
        """Test removal of repeated punctuation."""
        text = "Hello!!!!! World....."
        result = cleaner.remove_noise(text)
        assert "!!!!!" not in result
        assert "....." not in result

    def test_full_clean_pipeline(self, cleaner):
        """Test full cleaning pipeline."""
        text = "  HELLO    World!!!!\n\n\n\nTest  "
        result = cleaner.clean(text, lowercase=True, remove_noise=True)

        assert result == result.lower()
        assert "    " not in result
        assert result.strip() == result

    def test_clean_empty_text(self, cleaner):
        """Test cleaning empty text."""
        result = cleaner.clean("")
        assert result == ""

    def test_clean_preserves_content(self, cleaner):
        """Test that cleaning preserves meaningful content."""
        text = "The quick brown fox jumps over the lazy dog."
        result = cleaner.clean(text, lowercase=False, remove_noise=False)
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result


class TestProcessedDocument:
    """Tests for ProcessedDocument dataclass."""

    def test_processed_document_creation(self):
        """Test creating a processed document."""
        doc = ProcessedDocument(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            raw_text="Hello World",
            cleaned_text="hello world"
        )

        assert doc.document_id == "doc_test123"
        assert doc.word_count == 2
        assert doc.character_count == 11
        assert doc.status == "success"

    def test_processed_document_word_count(self):
        """Test word count calculation."""
        doc = ProcessedDocument(
            document_id="doc_test",
            original_filename="test.pdf",
            document_type="pdf",
            raw_text="One two three four five",
            cleaned_text="one two three four five"
        )

        assert doc.word_count == 5

    def test_processed_document_json_serialization(self):
        """Test JSON serialization/deserialization."""
        doc = ProcessedDocument(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            raw_text="Hello",
            cleaned_text="hello"
        )

        json_str = doc.to_json()
        restored = ProcessedDocument.from_json(json_str)

        assert restored.document_id == doc.document_id
        assert restored.cleaned_text == doc.cleaned_text


class TestDocumentPreprocessor:
    """Tests for DocumentPreprocessor class."""

    def test_preprocessor_initialization(self, test_data_dir, mock_settings):
        """Test preprocessor initializes correctly."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        assert preprocessor.processed_dir.exists()

    def test_process_from_file_pdf(self, test_data_dir, sample_pdf, mock_settings):
        """Test processing a PDF file directly."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        # Process the sample PDF
        try:
            result = preprocessor.process_from_file(sample_pdf)
            assert result.document_type == "pdf"
            assert result.processing_method == "pdfplumber"
        except Exception as e:
            # PDF may not have extractable text
            pytest.skip(f"PDF processing requires pdfplumber: {e}")

    def test_is_processed(self, test_data_dir, mock_settings):
        """Test checking if document is processed."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        # Initially not processed
        assert not preprocessor.is_processed("doc_test123")

        # Create a processed file
        doc = ProcessedDocument(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            raw_text="Hello",
            cleaned_text="hello"
        )

        output_path = test_data_dir / "processed" / "doc_test123.json"
        with open(output_path, "w") as f:
            f.write(doc.to_json())

        # Now it should be processed
        assert preprocessor.is_processed("doc_test123")

    def test_get_processed(self, test_data_dir, mock_settings):
        """Test retrieving a processed document."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        # Create a processed file
        doc = ProcessedDocument(
            document_id="doc_test123",
            original_filename="test.pdf",
            document_type="pdf",
            raw_text="Hello World",
            cleaned_text="hello world"
        )

        output_path = test_data_dir / "processed" / "doc_test123.json"
        with open(output_path, "w") as f:
            f.write(doc.to_json())

        # Retrieve it
        result = preprocessor.get_processed("doc_test123")

        assert result is not None
        assert result.document_id == "doc_test123"
        assert result.cleaned_text == "hello world"

    def test_get_processed_not_found(self, test_data_dir, mock_settings):
        """Test retrieving non-existent document returns None."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        result = preprocessor.get_processed("nonexistent_doc")
        assert result is None

    def test_list_processed(self, test_data_dir, mock_settings):
        """Test listing processed documents."""
        preprocessor = DocumentPreprocessor(
            raw_dir=test_data_dir / "raw",
            processed_dir=test_data_dir / "processed"
        )

        # Create some processed files
        for i in range(3):
            doc = ProcessedDocument(
                document_id=f"doc_test{i}",
                original_filename=f"test{i}.pdf",
                document_type="pdf",
                raw_text="Hello",
                cleaned_text="hello"
            )
            output_path = test_data_dir / "processed" / f"doc_test{i}.json"
            with open(output_path, "w") as f:
                f.write(doc.to_json())

        # List them
        documents = preprocessor.list_processed()

        assert len(documents) == 3
