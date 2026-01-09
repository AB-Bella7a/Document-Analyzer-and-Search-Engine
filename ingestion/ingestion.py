"""
Document Ingestion Pipeline

This module implements the document ingestion pipeline for the Document Analyzer.
It handles:
1. Accepting documents (PDFs and images) from various sources
2. Validating file types and sizes
3. Storing documents in the raw data directory
4. Creating and saving document metadata
5. Optionally syncing to AWS S3

The ingestion stage is designed to be runnable independently and produces
a consistent structure of raw files and metadata.

Example:
    >>> from ingestion import DocumentIngester
    >>>
    >>> ingester = DocumentIngester()
    >>> result = ingester.ingest_document("/path/to/document.pdf")
    >>> print(result.document_id)
    'doc_abc123...'
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict
import uuid

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


@dataclass
class DocumentMetadata:
    """
    Metadata for an ingested document.

    This dataclass stores all relevant information about a document
    that has been ingested into the system.

    Attributes:
        document_id: Unique identifier for the document
        original_filename: Original name of the uploaded file
        document_type: Type of document ('pdf' or 'image')
        file_extension: File extension (e.g., '.pdf', '.png')
        file_size_bytes: Size of the file in bytes
        upload_timestamp: When the document was ingested (ISO format)
        storage_location: Path to the stored file (local or S3)
        s3_location: S3 URI if synced to cloud (optional)
        checksum: MD5 hash of the file for integrity verification
        status: Current status ('ingested', 'processing', 'completed', 'error')
    """
    document_id: str
    original_filename: str
    document_type: str
    file_extension: str
    file_size_bytes: int
    upload_timestamp: str
    storage_location: str
    s3_location: Optional[str] = None
    checksum: str = ""
    status: str = "ingested"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create metadata from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentMetadata":
        """Create metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


class DocumentIngester:
    """
    Document Ingestion Pipeline

    This class handles the ingestion of documents into the system.
    It validates, stores, and catalogs documents for further processing.

    Attributes:
        raw_dir: Directory for storing raw documents
        use_s3: Whether to sync documents to S3

    Example:
        >>> ingester = DocumentIngester()
        >>>
        >>> # Ingest a single document
        >>> metadata = ingester.ingest_document("report.pdf")
        >>>
        >>> # Ingest multiple documents
        >>> results = ingester.ingest_batch(["doc1.pdf", "image1.png"])
    """

    def __init__(self, raw_dir: Optional[Path] = None, use_s3: bool = None):
        """
        Initialize the document ingester.

        Args:
            raw_dir: Directory to store raw documents (uses config default if None)
            use_s3: Whether to enable S3 sync (uses config default if None)
        """
        self.raw_dir = raw_dir or settings.RAW_DATA_DIR
        self.use_s3 = use_s3 if use_s3 is not None else settings.is_s3_configured()

        # Ensure the raw directory exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (self.raw_dir / "pdfs").mkdir(exist_ok=True)
        (self.raw_dir / "images").mkdir(exist_ok=True)
        (self.raw_dir / "metadata").mkdir(exist_ok=True)

        # Initialize S3 client if needed
        self.s3_client = None
        if self.use_s3:
            self._init_s3_client()

    def _init_s3_client(self):
        """Initialize AWS S3 client using boto3."""
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            print(f"[Ingestion] S3 client initialized for bucket: {settings.S3_BUCKET_NAME}")
        except Exception as e:
            print(f"[Ingestion] Warning: Could not initialize S3 client: {e}")
            self.use_s3 = False

    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        # Use UUID4 for uniqueness, with a 'doc_' prefix for clarity
        return f"doc_{uuid.uuid4().hex[:12]}"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file for integrity verification."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_document_type(self, filename: str) -> str:
        """
        Determine document type from filename.

        Returns:
            'pdf' for PDF files, 'image' for image files

        Raises:
            ValueError: If file type is not supported
        """
        extension = Path(filename).suffix.lower()

        if extension in settings.SUPPORTED_PDF_EXTENSIONS:
            return "pdf"
        elif extension in settings.SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {settings.get_supported_extensions()}"
            )

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate a file before ingestion.

        Checks:
        - File exists
        - File is not empty
        - File size is within limits
        - File type is supported

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes. "
                f"Maximum allowed: {settings.MAX_FILE_SIZE} bytes"
            )

        # This will raise ValueError if type is not supported
        self._get_document_type(file_path.name)

    def _store_locally(self, source_path: Path, document_id: str, doc_type: str) -> Path:
        """
        Store a document in the local raw directory.

        Args:
            source_path: Path to the source file
            document_id: Unique document identifier
            doc_type: Type of document ('pdf' or 'image')

        Returns:
            Path to the stored file
        """
        # Determine target subdirectory
        subdir = "pdfs" if doc_type == "pdf" else "images"
        target_dir = self.raw_dir / subdir

        # Create filename with document ID for uniqueness
        extension = source_path.suffix.lower()
        target_filename = f"{document_id}{extension}"
        target_path = target_dir / target_filename

        # Copy file to storage location
        shutil.copy2(source_path, target_path)

        return target_path

    def _sync_to_s3(self, local_path: Path, document_id: str) -> Optional[str]:
        """
        Upload a file to AWS S3.

        Args:
            local_path: Path to the local file
            document_id: Document ID for S3 key generation

        Returns:
            S3 URI if successful, None otherwise
        """
        if not self.use_s3 or not self.s3_client:
            return None

        try:
            # Generate S3 key
            s3_key = f"raw/{document_id}/{local_path.name}"

            # Upload file
            self.s3_client.upload_file(
                str(local_path),
                settings.S3_BUCKET_NAME,
                s3_key
            )

            s3_uri = f"s3://{settings.S3_BUCKET_NAME}/{s3_key}"
            print(f"[Ingestion] Uploaded to S3: {s3_uri}")
            return s3_uri

        except Exception as e:
            print(f"[Ingestion] Warning: S3 upload failed: {e}")
            return None

    def _save_metadata(self, metadata: DocumentMetadata) -> Path:
        """
        Save document metadata to a JSON file.

        Args:
            metadata: DocumentMetadata object to save

        Returns:
            Path to the saved metadata file
        """
        metadata_dir = self.raw_dir / "metadata"
        metadata_path = metadata_dir / f"{metadata.document_id}.json"

        with open(metadata_path, "w") as f:
            f.write(metadata.to_json())

        return metadata_path

    def ingest_document(
        self,
        file_path: Union[str, Path],
        sync_to_s3: bool = True
    ) -> DocumentMetadata:
        """
        Ingest a single document into the system.

        This method:
        1. Validates the file
        2. Generates a unique document ID
        3. Calculates file checksum
        4. Stores the file locally
        5. Optionally syncs to S3
        6. Creates and saves metadata

        Args:
            file_path: Path to the document file
            sync_to_s3: Whether to sync this file to S3 (if S3 is enabled)

        Returns:
            DocumentMetadata object with all document information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file validation fails

        Example:
            >>> ingester = DocumentIngester()
            >>> metadata = ingester.ingest_document("report.pdf")
            >>> print(f"Ingested: {metadata.document_id}")
        """
        file_path = Path(file_path)

        # Step 1: Validate the file
        print(f"[Ingestion] Validating file: {file_path.name}")
        self._validate_file(file_path)

        # Step 2: Generate document ID
        document_id = self._generate_document_id()
        print(f"[Ingestion] Generated ID: {document_id}")

        # Step 3: Determine document type
        doc_type = self._get_document_type(file_path.name)

        # Step 4: Calculate checksum
        checksum = self._calculate_checksum(file_path)

        # Step 5: Store file locally
        stored_path = self._store_locally(file_path, document_id, doc_type)
        print(f"[Ingestion] Stored at: {stored_path}")

        # Step 6: Sync to S3 if enabled
        s3_location = None
        if sync_to_s3 and self.use_s3:
            s3_location = self._sync_to_s3(stored_path, document_id)

        # Step 7: Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            original_filename=file_path.name,
            document_type=doc_type,
            file_extension=file_path.suffix.lower(),
            file_size_bytes=file_path.stat().st_size,
            upload_timestamp=datetime.utcnow().isoformat(),
            storage_location=str(stored_path),
            s3_location=s3_location,
            checksum=checksum,
            status="ingested"
        )

        # Step 8: Save metadata
        metadata_path = self._save_metadata(metadata)
        print(f"[Ingestion] Metadata saved: {metadata_path}")

        return metadata

    def ingest_batch(
        self,
        file_paths: List[Union[str, Path]],
        sync_to_s3: bool = True,
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents in batch.

        Args:
            file_paths: List of paths to document files
            sync_to_s3: Whether to sync files to S3
            continue_on_error: If True, continue processing on errors

        Returns:
            Dictionary with 'successful' and 'failed' lists

        Example:
            >>> results = ingester.ingest_batch(["doc1.pdf", "doc2.png"])
            >>> print(f"Success: {len(results['successful'])}")
            >>> print(f"Failed: {len(results['failed'])}")
        """
        results = {
            "successful": [],
            "failed": []
        }

        for file_path in file_paths:
            try:
                metadata = self.ingest_document(file_path, sync_to_s3)
                results["successful"].append({
                    "file": str(file_path),
                    "document_id": metadata.document_id,
                    "metadata": metadata.to_dict()
                })
            except Exception as e:
                error_info = {
                    "file": str(file_path),
                    "error": str(e)
                }
                results["failed"].append(error_info)
                print(f"[Ingestion] Error processing {file_path}: {e}")

                if not continue_on_error:
                    break

        print(f"[Ingestion] Batch complete: {len(results['successful'])} success, "
              f"{len(results['failed'])} failed")

        return results

    def ingest_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        sync_to_s3: bool = True
    ) -> DocumentMetadata:
        """
        Ingest a document from bytes (useful for API uploads).

        Args:
            file_bytes: Raw file content as bytes
            filename: Original filename (used for type detection)
            sync_to_s3: Whether to sync to S3

        Returns:
            DocumentMetadata object

        Example:
            >>> with open("document.pdf", "rb") as f:
            ...     data = f.read()
            >>> metadata = ingester.ingest_from_bytes(data, "document.pdf")
        """
        # Save bytes to a temporary file in uploads directory
        temp_path = settings.UPLOADS_DIR / filename

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        try:
            # Use standard ingestion process
            metadata = self.ingest_document(temp_path, sync_to_s3)
            return metadata
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve metadata for a document by ID.

        Args:
            document_id: The document's unique identifier

        Returns:
            DocumentMetadata if found, None otherwise
        """
        metadata_path = self.raw_dir / "metadata" / f"{document_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return DocumentMetadata.from_json(f.read())

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all ingested documents.

        Returns:
            List of DocumentMetadata objects for all ingested documents
        """
        metadata_dir = self.raw_dir / "metadata"
        documents = []

        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                documents.append(DocumentMetadata.from_json(f.read()))

        # Sort by upload timestamp (newest first)
        documents.sort(key=lambda x: x.upload_timestamp, reverse=True)

        return documents

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its metadata.

        Args:
            document_id: The document's unique identifier

        Returns:
            True if deleted successfully, False if not found
        """
        metadata = self.get_metadata(document_id)

        if not metadata:
            return False

        # Delete the raw file
        raw_file = Path(metadata.storage_location)
        if raw_file.exists():
            raw_file.unlink()

        # Delete metadata
        metadata_path = self.raw_dir / "metadata" / f"{document_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()

        print(f"[Ingestion] Deleted document: {document_id}")
        return True


# =============================================================================
# Standalone execution for testing
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("files", nargs="*", help="Files to ingest")
    parser.add_argument("--list", action="store_true", help="List all ingested documents")
    args = parser.parse_args()

    ingester = DocumentIngester()

    if args.list:
        print("\n=== Ingested Documents ===")
        documents = ingester.list_documents()
        if not documents:
            print("No documents found.")
        for doc in documents:
            print(f"  {doc.document_id}: {doc.original_filename} ({doc.document_type})")
    elif args.files:
        print(f"\n=== Ingesting {len(args.files)} file(s) ===")
        results = ingester.ingest_batch(args.files)
        print(f"\nResults: {len(results['successful'])} successful, {len(results['failed'])} failed")
    else:
        print("Usage: python ingestion.py [files...] or python ingestion.py --list")
