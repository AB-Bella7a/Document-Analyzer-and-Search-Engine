#!/usr/bin/env python3
"""
Document Analyzer & Search Engine - Main Entry Point

This is the main application entry point that can run the system
in different modes:
- API server mode (default)
- CLI mode for pipeline operations
- Interactive search mode

Usage:
    # Start the API server (default)
    python main.py

    # Run the processing pipeline
    python main.py pipeline

    # Run interactive search
    python main.py search

    # Show help
    python main.py --help
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_api_server():
    """Start the FastAPI server."""
    import uvicorn
    from config import settings

    print("=" * 60)
    print("  Document Analyzer & Search Engine")
    print("  Starting API Server...")
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )


def run_pipeline(args):
    """Run the document processing pipeline."""
    from scripts.run_pipeline import run_pipeline as pipeline_runner

    pipeline_runner(
        document_id=args.document_id,
        rebuild=args.rebuild,
        skip_preprocessing=args.skip_preprocessing,
        skip_embeddings=args.skip_embeddings,
        skip_indexing=args.skip_indexing
    )


def run_search(args):
    """Run interactive search or single query."""
    from scripts.search_cli import interactive_mode, run_search as search_query
    from search import SemanticSearch

    search_engine = SemanticSearch()

    if args.build_index:
        print("Building search index...")
        search_engine.build_index(force_rebuild=True)

    if args.query:
        search_query(args.query, search_engine, top_k=args.top_k)
    else:
        interactive_mode(search_engine, top_k=args.top_k)


def run_ingest(args):
    """Ingest documents."""
    from scripts.ingest_documents import find_documents, ingest_documents

    documents = find_documents(
        args.path,
        recursive=args.recursive,
        file_type=args.type
    )

    ingest_documents(
        documents,
        process=args.process,
        sync_s3=args.sync_s3
    )


def show_status():
    """Show system status and statistics."""
    from config import settings
    from ingestion import DocumentIngester
    from preprocessing import DocumentPreprocessor
    from embeddings import EmbeddingGenerator
    from search import SemanticSearch

    print("\n" + "=" * 60)
    print("  Document Analyzer & Search Engine - Status")
    print("=" * 60)

    # Configuration
    print("\n  Configuration:")
    print(f"    Version: {settings.APP_VERSION}")
    print(f"    Debug Mode: {settings.DEBUG}")
    print(f"    Data Directory: {settings.DATA_DIR}")
    print(f"    S3 Enabled: {settings.is_s3_configured()}")

    # Document counts
    try:
        ingester = DocumentIngester()
        preprocessor = DocumentPreprocessor()
        embedding_gen = EmbeddingGenerator()
        search_engine = SemanticSearch()

        total_docs = len(ingester.list_documents())
        processed_docs = len(preprocessor.list_processed())
        embedded_docs = len(embedding_gen.list_embeddings())

        index_stats = search_engine.get_index_stats()
        index_size = index_stats.get("total_documents", 0)

        print("\n  Document Pipeline:")
        print(f"    Ingested Documents: {total_docs}")
        print(f"    Processed Documents: {processed_docs}")
        print(f"    Embedded Documents: {embedded_docs}")
        print(f"    Search Index Size: {index_size}")

        # Pipeline health
        print("\n  Pipeline Health:")
        if total_docs > 0:
            preprocess_pct = (processed_docs / total_docs) * 100
            embed_pct = (embedded_docs / total_docs) * 100
            print(f"    Preprocessing: {preprocess_pct:.1f}% complete")
            print(f"    Embeddings: {embed_pct:.1f}% complete")
        else:
            print("    No documents in system")

    except Exception as e:
        print(f"\n  Error getting statistics: {e}")

    # Models
    print("\n  AI Models:")
    print(f"    Text Embedding: {settings.TEXT_EMBEDDING_MODEL}")
    print(f"    Image Embedding: {settings.IMAGE_EMBEDDING_MODEL}")

    print("\n" + "=" * 60)


def main():
    """Main entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="Document Analyzer & Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  api       Start the REST API server (default)
  pipeline  Run the document processing pipeline
  search    Search documents (interactive or single query)
  ingest    Ingest documents from files/directories
  status    Show system status and statistics

Examples:
  python main.py                           # Start API server
  python main.py pipeline                  # Process all pending documents
  python main.py search "machine learning" # Search documents
  python main.py ingest ./docs/            # Ingest documents
  python main.py status                    # Show system status
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # API command (default)
    api_parser = subparsers.add_parser("api", help="Start the REST API server")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run processing pipeline")
    pipeline_parser.add_argument("--document-id", "-d", help="Process specific document")
    pipeline_parser.add_argument("--rebuild", "-r", action="store_true", help="Force rebuild")
    pipeline_parser.add_argument("--skip-preprocessing", action="store_true")
    pipeline_parser.add_argument("--skip-embeddings", action="store_true")
    pipeline_parser.add_argument("--skip-indexing", action="store_true")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", nargs="?", help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5)
    search_parser.add_argument("--build-index", "-b", action="store_true")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", type=Path, help="File or directory path")
    ingest_parser.add_argument("--recursive", "-r", action="store_true")
    ingest_parser.add_argument("--type", "-t", choices=["pdf", "image"])
    ingest_parser.add_argument("--process", "-p", action="store_true")
    ingest_parser.add_argument("--sync-s3", action="store_true")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    # Route to appropriate command
    if args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "ingest":
        run_ingest(args)
    elif args.command == "status":
        show_status()
    else:
        # Default: run API server
        run_api_server()


if __name__ == "__main__":
    main()
