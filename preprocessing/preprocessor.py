"""
Document Preprocessing Pipeline

This module implements the document preprocessing pipeline which:
1. Detects document type (PDF or image)
2. Extracts text content using appropriate methods:
   - PDFs: pdfplumber for text extraction
   - Images: pytesseract (Tesseract OCR) for optical character recognition
3. Cleans and normalizes extracted text
4. Saves processed output as structured JSON files

The preprocessing stage takes ingested documents and prepares them for
embedding generation.

Example:
    >>> from preprocessing import DocumentPreprocessor
    >>>
    >>> preprocessor = DocumentPreprocessor()
    >>> result = preprocessor.process_document("doc_abc123")
    >>> print(result.cleaned_text[:100])
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field
import unicodedata

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from ingestion import DocumentIngester, DocumentMetadata


@dataclass
class ProcessedDocument:
    """
    Processed document data structure.

    This dataclass stores the results of document preprocessing including
    extracted text, cleaned text, and processing metadata.

    Attributes:
        document_id: Unique document identifier (matches ingestion ID)
        original_filename: Original filename from ingestion
        document_type: Type of document ('pdf' or 'image')
        raw_text: Original extracted text before cleaning
        cleaned_text: Normalized and cleaned text
        page_count: Number of pages (for PDFs) or 1 for images
        word_count: Number of words in cleaned text
        character_count: Number of characters in cleaned text
        processing_timestamp: When preprocessing was completed
        processing_method: Method used ('pdfplumber', 'tesseract')
        extraction_metadata: Additional extraction info (pages, confidence, etc.)
        status: Processing status ('success', 'partial', 'failed')
        error_message: Error details if processing failed
    """
    document_id: str
    original_filename: str
    document_type: str
    raw_text: str
    cleaned_text: str
    page_count: int = 1
    word_count: int = 0
    character_count: int = 0
    processing_timestamp: str = ""
    processing_method: str = ""
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None

    def __post_init__(self):
        """Calculate statistics after initialization."""
        if not self.processing_timestamp:
            self.processing_timestamp = datetime.utcnow().isoformat()
        if self.cleaned_text:
            self.word_count = len(self.cleaned_text.split())
            self.character_count = len(self.cleaned_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedDocument":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessedDocument":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class TextCleaner:
    """
    Text cleaning and normalization utilities.

    This class provides methods to clean and standardize extracted text,
    making it suitable for embedding generation and search.
    """

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters to their closest ASCII representation.

        This helps standardize characters like curly quotes, em-dashes, etc.
        """
        # Normalize to NFKC form (compatibility decomposition)
        text = unicodedata.normalize("NFKC", text)
        return text

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove non-printable control characters."""
        # Keep newlines and tabs, remove other control chars
        return "".join(
            char for char in text
            if unicodedata.category(char) != "Cc" or char in "\n\t"
        )

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Standardize whitespace in text.

        - Converts all whitespace to single spaces
        - Removes leading/trailing whitespace
        - Preserves paragraph breaks (double newlines)
        """
        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Preserve paragraph breaks by replacing with placeholder
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Remove spaces at beginning/end of lines
        text = re.sub(r" *\n *", "\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def remove_noise(text: str) -> str:
        """
        Remove common OCR noise and artifacts.

        This handles issues like:
        - Repeated punctuation
        - Stray characters
        - Common OCR errors
        """
        # Remove repeated punctuation (more than 3)
        text = re.sub(r"([.!?,;:])\1{3,}", r"\1\1\1", text)

        # Remove lines that are just punctuation/symbols
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Keep line if it has at least some alphabetic characters
            if re.search(r"[a-zA-Z]", line) or not line.strip():
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Remove excessive newlines (more than 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def clean(
        self,
        text: str,
        lowercase: bool = True,
        remove_noise: bool = True
    ) -> str:
        """
        Apply full cleaning pipeline to text.

        Args:
            text: Raw text to clean
            lowercase: Whether to convert to lowercase
            remove_noise: Whether to remove OCR noise

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Step 1: Normalize unicode
        text = self.normalize_unicode(text)

        # Step 2: Remove control characters
        text = self.remove_control_characters(text)

        # Step 3: Normalize whitespace
        text = self.normalize_whitespace(text)

        # Step 4: Remove noise if requested
        if remove_noise:
            text = self.remove_noise(text)

        # Step 5: Lowercase if requested
        if lowercase:
            text = self.lowercase(text)

        return text


class PDFExtractor:
    """
    PDF text extraction using pdfplumber.

    pdfplumber provides excellent text extraction with layout preservation
    and handles complex PDF structures well.
    """

    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with 'text', 'page_count', and 'pages' (per-page text)

        Raises:
            ImportError: If pdfplumber is not installed
            Exception: If PDF extraction fails
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install with: pip install pdfplumber"
            )

        result = {
            "text": "",
            "page_count": 0,
            "pages": []  # Text for each page
        }

        try:
            with pdfplumber.open(file_path) as pdf:
                result["page_count"] = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    # Extract text from page
                    page_text = page.extract_text() or ""

                    result["pages"].append({
                        "page_number": i + 1,
                        "text": page_text,
                        "width": page.width,
                        "height": page.height
                    })

                # Combine all pages
                result["text"] = "\n\n".join(
                    p["text"] for p in result["pages"] if p["text"]
                )

            print(f"[Preprocessing] Extracted {result['page_count']} pages from PDF")
            return result

        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")


class ImageOCRExtractor:
    """
    Image text extraction using Tesseract OCR via pytesseract.

    Tesseract is an open-source OCR engine that works well for
    printed text in images.
    """

    def __init__(self):
        """Initialize OCR extractor and verify Tesseract installation."""
        # Set Tesseract command path if configured
        if settings.TESSERACT_CMD:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with 'text', 'confidence', and 'image_info'

        Raises:
            ImportError: If pytesseract or Pillow is not installed
            Exception: If OCR extraction fails
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "pytesseract and Pillow are required for OCR. "
                "Install with: pip install pytesseract Pillow"
            )

        result = {
            "text": "",
            "confidence": 0.0,
            "image_info": {}
        }

        try:
            # Open and process image
            with Image.open(file_path) as img:
                # Store image metadata
                result["image_info"] = {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height
                }

                # Convert to RGB if necessary (OCR works best with RGB)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Perform OCR with detailed data
                ocr_data = pytesseract.image_to_data(
                    img,
                    lang=settings.OCR_LANGUAGE,
                    output_type=pytesseract.Output.DICT
                )

                # Extract text
                result["text"] = pytesseract.image_to_string(
                    img,
                    lang=settings.OCR_LANGUAGE
                )

                # Calculate average confidence (exclude -1 values which indicate no text)
                confidences = [
                    c for c in ocr_data["conf"] if c != -1
                ]
                if confidences:
                    result["confidence"] = sum(confidences) / len(confidences)

            print(f"[Preprocessing] OCR completed with {result['confidence']:.1f}% confidence")
            return result

        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")


class DocumentPreprocessor:
    """
    Document Preprocessing Pipeline

    This class orchestrates the preprocessing of documents:
    1. Loads document metadata from ingestion
    2. Extracts text using appropriate method (PDF or OCR)
    3. Cleans and normalizes the text
    4. Saves processed output as JSON

    Example:
        >>> preprocessor = DocumentPreprocessor()
        >>>
        >>> # Process a single document
        >>> result = preprocessor.process_document("doc_abc123")
        >>>
        >>> # Process all unprocessed documents
        >>> results = preprocessor.process_all_pending()
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None
    ):
        """
        Initialize the document preprocessor.

        Args:
            raw_dir: Directory containing raw documents (default from config)
            processed_dir: Directory for processed output (default from config)
        """
        self.raw_dir = raw_dir or settings.RAW_DATA_DIR
        self.processed_dir = processed_dir or settings.PROCESSED_DATA_DIR

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        self.ocr_extractor = ImageOCRExtractor()
        self.text_cleaner = TextCleaner()

        # Initialize ingester for metadata access
        self.ingester = DocumentIngester(raw_dir=self.raw_dir)

    def _extract_text(
        self,
        file_path: Path,
        document_type: str
    ) -> Dict[str, Any]:
        """
        Extract text from a document using appropriate method.

        Args:
            file_path: Path to the document
            document_type: 'pdf' or 'image'

        Returns:
            Extraction result dictionary
        """
        if document_type == "pdf":
            return self.pdf_extractor.extract(file_path)
        else:
            return self.ocr_extractor.extract(file_path)

    def process_document(
        self,
        document_id: str,
        lowercase: bool = True,
        remove_noise: bool = True
    ) -> ProcessedDocument:
        """
        Process a single document.

        Args:
            document_id: The document's unique identifier
            lowercase: Whether to convert text to lowercase
            remove_noise: Whether to remove OCR noise

        Returns:
            ProcessedDocument with extracted and cleaned text

        Raises:
            ValueError: If document not found
            Exception: If processing fails
        """
        print(f"[Preprocessing] Processing document: {document_id}")

        # Step 1: Load document metadata
        metadata = self.ingester.get_metadata(document_id)
        if not metadata:
            raise ValueError(f"Document not found: {document_id}")

        # Step 2: Verify file exists
        file_path = Path(metadata.storage_location)
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        # Step 3: Extract text
        print(f"[Preprocessing] Extracting text from {metadata.document_type}...")
        try:
            extraction_result = self._extract_text(file_path, metadata.document_type)
            raw_text = extraction_result.get("text", "")
            processing_method = (
                "pdfplumber" if metadata.document_type == "pdf" else "tesseract"
            )
        except Exception as e:
            # Return failed result
            return ProcessedDocument(
                document_id=document_id,
                original_filename=metadata.original_filename,
                document_type=metadata.document_type,
                raw_text="",
                cleaned_text="",
                processing_method="",
                status="failed",
                error_message=str(e)
            )

        # Step 4: Clean text
        print("[Preprocessing] Cleaning and normalizing text...")
        cleaned_text = self.text_cleaner.clean(
            raw_text,
            lowercase=lowercase,
            remove_noise=remove_noise
        )

        # Step 5: Create processed document
        page_count = extraction_result.get("page_count", 1)

        processed = ProcessedDocument(
            document_id=document_id,
            original_filename=metadata.original_filename,
            document_type=metadata.document_type,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            page_count=page_count,
            processing_method=processing_method,
            extraction_metadata=extraction_result,
            status="success" if cleaned_text else "partial"
        )

        # Step 6: Save processed output
        self._save_processed(processed)

        print(f"[Preprocessing] Complete: {processed.word_count} words extracted")
        return processed

    def process_from_file(
        self,
        file_path: Union[str, Path],
        lowercase: bool = True,
        remove_noise: bool = True
    ) -> ProcessedDocument:
        """
        Process a document directly from file path (bypasses ingestion).

        Useful for quick testing or processing files not yet ingested.

        Args:
            file_path: Path to the document file
            lowercase: Whether to convert text to lowercase
            remove_noise: Whether to remove OCR noise

        Returns:
            ProcessedDocument with extracted and cleaned text
        """
        file_path = Path(file_path)

        # Determine document type
        if settings.is_pdf(file_path.name):
            document_type = "pdf"
        elif settings.is_image(file_path.name):
            document_type = "image"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Extract text
        extraction_result = self._extract_text(file_path, document_type)
        raw_text = extraction_result.get("text", "")

        # Clean text
        cleaned_text = self.text_cleaner.clean(
            raw_text,
            lowercase=lowercase,
            remove_noise=remove_noise
        )

        # Create result (with placeholder document ID)
        return ProcessedDocument(
            document_id=f"temp_{file_path.stem}",
            original_filename=file_path.name,
            document_type=document_type,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            page_count=extraction_result.get("page_count", 1),
            processing_method="pdfplumber" if document_type == "pdf" else "tesseract",
            extraction_metadata=extraction_result,
            status="success" if cleaned_text else "partial"
        )

    def _save_processed(self, processed: ProcessedDocument) -> Path:
        """
        Save processed document to JSON file.

        Args:
            processed: ProcessedDocument to save

        Returns:
            Path to saved file
        """
        output_path = self.processed_dir / f"{processed.document_id}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed.to_json())

        print(f"[Preprocessing] Saved: {output_path}")
        return output_path

    def get_processed(self, document_id: str) -> Optional[ProcessedDocument]:
        """
        Load a processed document by ID.

        Args:
            document_id: The document's unique identifier

        Returns:
            ProcessedDocument if found, None otherwise
        """
        processed_path = self.processed_dir / f"{document_id}.json"

        if not processed_path.exists():
            return None

        with open(processed_path, "r", encoding="utf-8") as f:
            return ProcessedDocument.from_json(f.read())

    def is_processed(self, document_id: str) -> bool:
        """Check if a document has been processed."""
        return (self.processed_dir / f"{document_id}.json").exists()

    def process_all_pending(
        self,
        lowercase: bool = True,
        remove_noise: bool = True
    ) -> Dict[str, Any]:
        """
        Process all documents that haven't been processed yet.

        Returns:
            Dictionary with 'successful' and 'failed' lists
        """
        results = {
            "successful": [],
            "failed": []
        }

        # Get all ingested documents
        documents = self.ingester.list_documents()

        for doc in documents:
            # Skip if already processed
            if self.is_processed(doc.document_id):
                print(f"[Preprocessing] Skipping (already processed): {doc.document_id}")
                continue

            try:
                processed = self.process_document(
                    doc.document_id,
                    lowercase=lowercase,
                    remove_noise=remove_noise
                )
                results["successful"].append({
                    "document_id": doc.document_id,
                    "word_count": processed.word_count,
                    "status": processed.status
                })
            except Exception as e:
                results["failed"].append({
                    "document_id": doc.document_id,
                    "error": str(e)
                })

        print(f"[Preprocessing] Batch complete: {len(results['successful'])} success, "
              f"{len(results['failed'])} failed")

        return results

    def list_processed(self) -> List[ProcessedDocument]:
        """
        List all processed documents.

        Returns:
            List of ProcessedDocument objects
        """
        documents = []

        for json_file in self.processed_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                documents.append(ProcessedDocument.from_json(f.read()))

        # Sort by processing timestamp (newest first)
        documents.sort(key=lambda x: x.processing_timestamp, reverse=True)

        return documents


# =============================================================================
# Standalone execution for testing
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document Preprocessing Pipeline")
    parser.add_argument("--document-id", "-d", help="Process a specific document by ID")
    parser.add_argument("--file", "-f", help="Process a file directly (for testing)")
    parser.add_argument("--all", "-a", action="store_true", help="Process all pending documents")
    parser.add_argument("--list", "-l", action="store_true", help="List all processed documents")
    args = parser.parse_args()

    preprocessor = DocumentPreprocessor()

    if args.list:
        print("\n=== Processed Documents ===")
        documents = preprocessor.list_processed()
        if not documents:
            print("No processed documents found.")
        for doc in documents:
            print(f"  {doc.document_id}: {doc.original_filename}")
            print(f"    Words: {doc.word_count}, Status: {doc.status}")

    elif args.document_id:
        print(f"\n=== Processing Document: {args.document_id} ===")
        result = preprocessor.process_document(args.document_id)
        print(f"Status: {result.status}")
        print(f"Words: {result.word_count}")
        print(f"Preview: {result.cleaned_text[:200]}...")

    elif args.file:
        print(f"\n=== Processing File: {args.file} ===")
        result = preprocessor.process_from_file(args.file)
        print(f"Status: {result.status}")
        print(f"Words: {result.word_count}")
        print(f"Preview: {result.cleaned_text[:200]}...")

    elif args.all:
        print("\n=== Processing All Pending Documents ===")
        results = preprocessor.process_all_pending()
        print(f"\nResults: {len(results['successful'])} successful, "
              f"{len(results['failed'])} failed")

    else:
        print("Usage:")
        print("  python preprocessor.py --document-id <id>  # Process specific document")
        print("  python preprocessor.py --file <path>       # Process file directly")
        print("  python preprocessor.py --all               # Process all pending")
        print("  python preprocessor.py --list              # List processed documents")
