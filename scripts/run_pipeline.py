#!/usr/bin/env python3
"""
Pipeline Runner Script

This script runs the full document processing pipeline:
1. Process all pending documents (text extraction)
2. Generate embeddings for processed documents
3. Build/update the search index

Use this script to process new documents or rebuild the pipeline
after configuration changes.

Usage:
    # Process all pending documents
    python scripts/run_pipeline.py

    # Process a specific document
    python scripts/run_pipeline.py --document-id doc_abc123

    # Force rebuild everything
    python scripts/run_pipeline.py --rebuild

    # Skip certain stages
    python scripts/run_pipeline.py --skip-preprocessing
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from ingestion import DocumentIngester
from preprocessing import DocumentPreprocessor
from embeddings import EmbeddingGenerator
from search import SemanticSearch


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_stats(label: str, count: int, total: int = None):
    """Print statistics."""
    if total is not None:
        print(f"  {label}: {count}/{total}")
    else:
        print(f"  {label}: {count}")


def run_pipeline(
    document_id: str = None,
    skip_preprocessing: bool = False,
    skip_embeddings: bool = False,
    skip_indexing: bool = False,
    rebuild: bool = False
):
    """
    Run the document processing pipeline.

    Args:
        document_id: Optional specific document to process
        skip_preprocessing: Skip the preprocessing stage
        skip_embeddings: Skip embedding generation
        skip_indexing: Skip search index building
        rebuild: Force rebuild all stages
    """
    start_time = datetime.now()

    print_header("Document Analyzer Pipeline")
    print(f"  Started: {start_time.isoformat()}")
    print(f"  Mode: {'Single document' if document_id else 'Batch processing'}")
    print(f"  Rebuild: {rebuild}")

    # Initialize components
    ingester = DocumentIngester()
    preprocessor = DocumentPreprocessor()
    embedding_gen = EmbeddingGenerator()
    search_engine = SemanticSearch()

    results = {
        "preprocessing": {"success": 0, "failed": 0, "skipped": 0},
        "embeddings": {"success": 0, "failed": 0, "skipped": 0},
        "indexing": {"size": 0}
    }

    # ==========================================================================
    # Stage 1: Preprocessing
    # ==========================================================================
    if not skip_preprocessing:
        print_header("Stage 1: Preprocessing")

        if document_id:
            # Process single document
            if not rebuild and preprocessor.is_processed(document_id):
                print(f"  Document already processed: {document_id}")
                results["preprocessing"]["skipped"] = 1
            else:
                try:
                    processed = preprocessor.process_document(document_id)
                    print(f"  Processed: {document_id}")
                    print(f"  Words extracted: {processed.word_count}")
                    results["preprocessing"]["success"] = 1
                except Exception as e:
                    print(f"  Error: {e}")
                    results["preprocessing"]["failed"] = 1
        else:
            # Process all pending
            if rebuild:
                # Reprocess all documents
                documents = ingester.list_documents()
                for doc in documents:
                    try:
                        processed = preprocessor.process_document(doc.document_id)
                        print(f"  Processed: {doc.original_filename}")
                        results["preprocessing"]["success"] += 1
                    except Exception as e:
                        print(f"  Error processing {doc.document_id}: {e}")
                        results["preprocessing"]["failed"] += 1
            else:
                # Process only unprocessed documents
                batch_results = preprocessor.process_all_pending()
                results["preprocessing"]["success"] = len(batch_results["successful"])
                results["preprocessing"]["failed"] = len(batch_results["failed"])

        print(f"\n  Preprocessing complete:")
        print_stats("Success", results["preprocessing"]["success"])
        print_stats("Failed", results["preprocessing"]["failed"])
        print_stats("Skipped", results["preprocessing"]["skipped"])
    else:
        print_header("Stage 1: Preprocessing (SKIPPED)")

    # ==========================================================================
    # Stage 2: Embedding Generation
    # ==========================================================================
    if not skip_embeddings:
        print_header("Stage 2: Embedding Generation")

        if document_id:
            # Generate for single document
            if not rebuild and embedding_gen.has_embedding(document_id):
                print(f"  Embedding already exists: {document_id}")
                results["embeddings"]["skipped"] = 1
            else:
                try:
                    embedding = embedding_gen.generate_document_embedding(document_id)
                    print(f"  Generated embedding: {document_id}")
                    print(f"  Text dimension: {embedding.text_dimension}")
                    results["embeddings"]["success"] = 1
                except Exception as e:
                    print(f"  Error: {e}")
                    results["embeddings"]["failed"] = 1
        else:
            # Generate for all
            batch_results = embedding_gen.generate_all_embeddings(
                skip_existing=not rebuild
            )
            results["embeddings"]["success"] = len(batch_results["successful"])
            results["embeddings"]["failed"] = len(batch_results["failed"])

        print(f"\n  Embedding generation complete:")
        print_stats("Success", results["embeddings"]["success"])
        print_stats("Failed", results["embeddings"]["failed"])
        print_stats("Skipped", results["embeddings"]["skipped"])
    else:
        print_header("Stage 2: Embedding Generation (SKIPPED)")

    # ==========================================================================
    # Stage 3: Search Index Building
    # ==========================================================================
    if not skip_indexing:
        print_header("Stage 3: Search Index Building")

        try:
            index_size = search_engine.build_index(force_rebuild=rebuild)
            results["indexing"]["size"] = index_size
            print(f"  Index built successfully")
            print(f"  Documents indexed: {index_size}")
        except Exception as e:
            print(f"  Error building index: {e}")
    else:
        print_header("Stage 3: Search Index Building (SKIPPED)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("Pipeline Complete")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Documents processed: {results['preprocessing']['success']}")
    print(f"  Embeddings generated: {results['embeddings']['success']}")
    print(f"  Search index size: {results['indexing']['size']}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the document processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # Process all pending
  python scripts/run_pipeline.py --document-id X   # Process specific document
  python scripts/run_pipeline.py --rebuild         # Rebuild everything
  python scripts/run_pipeline.py --skip-embeddings # Skip embedding stage
        """
    )

    parser.add_argument(
        "--document-id", "-d",
        help="Process a specific document by ID"
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="Force rebuild all stages (reprocess everything)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing stage"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding generation stage"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip the search index building stage"
    )

    args = parser.parse_args()

    run_pipeline(
        document_id=args.document_id,
        skip_preprocessing=args.skip_preprocessing,
        skip_embeddings=args.skip_embeddings,
        skip_indexing=args.skip_indexing,
        rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()
