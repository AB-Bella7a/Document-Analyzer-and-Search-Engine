#!/usr/bin/env python3
"""
Document Ingestion Script

This script provides a convenient way to ingest documents into the system.
It supports:
- Ingesting single files
- Batch ingestion from a directory
- Filtering by file type
- Automatic pipeline processing (optional)

Usage:
    # Ingest a single file
    python scripts/ingest_documents.py path/to/document.pdf

    # Ingest all documents from a directory
    python scripts/ingest_documents.py path/to/docs/ --recursive

    # Ingest and process (run full pipeline)
    python scripts/ingest_documents.py path/to/doc.pdf --process

    # Filter by file type
    python scripts/ingest_documents.py path/to/docs/ --type pdf
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from ingestion import DocumentIngester


def find_documents(
    path: Path,
    recursive: bool = False,
    file_type: str = None
) -> List[Path]:
    """
    Find documents in a path.

    Args:
        path: File or directory path
        recursive: Search subdirectories
        file_type: Filter by type ('pdf', 'image', or None for all)

    Returns:
        List of document paths
    """
    documents = []

    # Determine extensions to look for
    if file_type == "pdf":
        extensions = settings.SUPPORTED_PDF_EXTENSIONS
    elif file_type == "image":
        extensions = settings.SUPPORTED_IMAGE_EXTENSIONS
    else:
        extensions = settings.get_supported_extensions()

    if path.is_file():
        # Single file
        if path.suffix.lower() in extensions:
            documents.append(path)
        else:
            print(f"Warning: Unsupported file type: {path}")
    elif path.is_dir():
        # Directory
        pattern = "**/*" if recursive else "*"
        for ext in extensions:
            documents.extend(path.glob(f"{pattern}{ext}"))
            # Also check uppercase extensions
            documents.extend(path.glob(f"{pattern}{ext.upper()}"))
    else:
        print(f"Error: Path not found: {path}")

    # Remove duplicates and sort
    documents = sorted(set(documents))

    return documents


def ingest_documents(
    documents: List[Path],
    process: bool = False,
    sync_s3: bool = False
) -> dict:
    """
    Ingest a list of documents.

    Args:
        documents: List of document paths
        process: Run full pipeline after ingestion
        sync_s3: Sync to S3 if configured

    Returns:
        Results dictionary
    """
    if not documents:
        print("No documents to ingest.")
        return {"success": [], "failed": []}

    print(f"\n{'='*60}")
    print(f"  Ingesting {len(documents)} document(s)")
    print(f"{'='*60}\n")

    # Initialize ingester
    ingester = DocumentIngester()

    # Ingest documents
    results = ingester.ingest_batch(documents, sync_to_s3=sync_s3)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Ingestion Results")
    print(f"{'='*60}")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")

    # List successful ingestions
    if results['successful']:
        print("\n  Ingested documents:")
        for item in results['successful']:
            print(f"    - {item['document_id']}: {Path(item['file']).name}")

    # List failures
    if results['failed']:
        print("\n  Failed documents:")
        for item in results['failed']:
            print(f"    - {Path(item['file']).name}: {item['error']}")

    # Run pipeline if requested
    if process and results['successful']:
        print(f"\n{'='*60}")
        print(f"  Running Pipeline")
        print(f"{'='*60}")

        from scripts.run_pipeline import run_pipeline

        for item in results['successful']:
            try:
                run_pipeline(document_id=item['document_id'])
            except Exception as e:
                print(f"  Pipeline error for {item['document_id']}: {e}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Document Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_documents.py document.pdf
  python scripts/ingest_documents.py ./documents/ --recursive
  python scripts/ingest_documents.py ./docs/ --type pdf --process
  python scripts/ingest_documents.py image.png --sync-s3
        """
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to document file or directory"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directories recursively"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["pdf", "image"],
        help="Filter by document type"
    )
    parser.add_argument(
        "--process", "-p",
        action="store_true",
        help="Run full pipeline after ingestion"
    )
    parser.add_argument(
        "--sync-s3",
        action="store_true",
        help="Sync documents to S3 (if configured)"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list documents that would be ingested"
    )

    args = parser.parse_args()

    # Find documents
    documents = find_documents(
        args.path,
        recursive=args.recursive,
        file_type=args.type
    )

    if args.list_only:
        print(f"\nFound {len(documents)} document(s):")
        for doc in documents:
            print(f"  - {doc}")
        return

    # Ingest documents
    ingest_documents(
        documents,
        process=args.process,
        sync_s3=args.sync_s3
    )


if __name__ == "__main__":
    main()
