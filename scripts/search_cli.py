#!/usr/bin/env python3
"""
Search CLI Script

This script provides a command-line interface for semantic search.
It supports:
- Interactive search mode
- Single query mode
- Finding similar documents
- Various output formats

Usage:
    # Interactive search mode
    python scripts/search_cli.py

    # Single query
    python scripts/search_cli.py "machine learning algorithms"

    # Find similar documents
    python scripts/search_cli.py --similar-to doc_abc123

    # Customize results
    python scripts/search_cli.py "query" --top-k 10 --min-score 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from search import SemanticSearch, SearchResult
from preprocessing import DocumentPreprocessor


def format_result(
    result: SearchResult,
    show_preview: bool = True,
    preview_length: int = 200
) -> str:
    """
    Format a search result for display.

    Args:
        result: SearchResult object
        show_preview: Whether to show text preview
        preview_length: Maximum preview length

    Returns:
        Formatted string
    """
    lines = [
        f"\n  [{result.rank}] {result.original_filename}",
        f"      Document ID: {result.document_id}",
        f"      Score: {result.similarity_score:.4f}",
        f"      Type: {result.document_type} | Words: {result.word_count}"
    ]

    if show_preview and result.text_preview:
        preview = result.text_preview[:preview_length]
        if len(result.text_preview) > preview_length:
            preview += "..."
        # Indent preview
        preview_lines = preview.split('\n')
        lines.append(f"      Preview: {preview_lines[0]}")
        for line in preview_lines[1:3]:  # Show up to 3 lines
            lines.append(f"               {line}")

    return "\n".join(lines)


def run_search(
    query: str,
    search_engine: SemanticSearch,
    top_k: int = 5,
    min_score: float = 0.0,
    show_preview: bool = True
) -> None:
    """
    Run a search query and display results.

    Args:
        query: Search query
        search_engine: SemanticSearch instance
        top_k: Number of results
        min_score: Minimum score threshold
        show_preview: Show text previews
    """
    print(f"\n  Searching: \"{query}\"")
    print("  " + "-" * 50)

    results = search_engine.search(query, top_k=top_k, min_score=min_score)

    if not results:
        print("\n  No results found.")
        return

    print(f"\n  Found {len(results)} result(s):")

    for result in results:
        print(format_result(result, show_preview=show_preview))


def run_similar(
    document_id: str,
    search_engine: SemanticSearch,
    top_k: int = 5,
    show_preview: bool = True
) -> None:
    """
    Find documents similar to a given document.

    Args:
        document_id: Document ID to find similar to
        search_engine: SemanticSearch instance
        top_k: Number of results
        show_preview: Show text previews
    """
    print(f"\n  Finding documents similar to: {document_id}")
    print("  " + "-" * 50)

    try:
        results = search_engine.search_by_document(document_id, top_k=top_k)
    except ValueError as e:
        print(f"\n  Error: {e}")
        return

    if not results:
        print("\n  No similar documents found.")
        return

    print(f"\n  Found {len(results)} similar document(s):")

    for result in results:
        print(format_result(result, show_preview=show_preview))


def interactive_mode(search_engine: SemanticSearch, top_k: int = 5) -> None:
    """
    Run interactive search mode.

    Args:
        search_engine: SemanticSearch instance
        top_k: Default number of results
    """
    print("\n" + "=" * 60)
    print("  Document Analyzer - Interactive Search")
    print("=" * 60)
    print("\n  Commands:")
    print("    <query>          - Search for documents")
    print("    /similar <id>    - Find similar documents")
    print("    /stats           - Show index statistics")
    print("    /help            - Show this help")
    print("    /quit or /exit   - Exit interactive mode")
    print("\n  Options (add to query):")
    print("    -k <num>         - Set number of results")
    print("    -m <score>       - Set minimum score (0-1)")
    print()

    while True:
        try:
            user_input = input("  search> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split()
            command = parts[0].lower()

            if command in ["/quit", "/exit", "/q"]:
                print("\n  Goodbye!")
                break

            elif command == "/help":
                print("\n  Commands:")
                print("    /similar <id>  - Find similar documents")
                print("    /stats         - Show index statistics")
                print("    /quit          - Exit")

            elif command == "/stats":
                stats = search_engine.get_index_stats()
                print(f"\n  Index Statistics:")
                print(f"    Status: {stats.get('status')}")
                print(f"    Documents: {stats.get('total_documents')}")
                print(f"    Index Type: {stats.get('index_type')}")

            elif command == "/similar":
                if len(parts) < 2:
                    print("  Usage: /similar <document_id>")
                else:
                    run_similar(parts[1], search_engine, top_k=top_k)

            else:
                print(f"  Unknown command: {command}")

        else:
            # Parse query with options
            query = user_input
            k = top_k
            min_score = 0.0

            # Extract options
            if " -k " in query:
                parts = query.split(" -k ")
                query = parts[0]
                try:
                    k = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

            if " -m " in query:
                parts = query.split(" -m ")
                query = parts[0]
                try:
                    min_score = float(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

            run_search(query, search_engine, top_k=k, min_score=min_score)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Search documents using natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/search_cli.py                          # Interactive mode
  python scripts/search_cli.py "machine learning"       # Single query
  python scripts/search_cli.py --similar-to doc_abc123  # Find similar
  python scripts/search_cli.py "query" -k 10 -m 0.5     # With options
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit for interactive mode)"
    )
    parser.add_argument(
        "--similar-to", "-s",
        help="Find documents similar to this document ID"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--min-score", "-m",
        type=float,
        default=0.0,
        help="Minimum similarity score (0-1, default: 0)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Don't show text previews"
    )
    parser.add_argument(
        "--build-index", "-b",
        action="store_true",
        help="Build/rebuild search index before searching"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics and exit"
    )

    args = parser.parse_args()

    # Initialize search engine
    print("\n  Initializing search engine...")
    search_engine = SemanticSearch()

    # Build index if requested
    if args.build_index:
        print("  Building search index...")
        count = search_engine.build_index(force_rebuild=True)
        print(f"  Index built with {count} documents")

    # Show stats if requested
    if args.stats:
        stats = search_engine.get_index_stats()
        print("\n  Index Statistics:")
        for key, value in stats.items():
            if key != "document_ids":  # Don't print full list
                print(f"    {key}: {value}")
        return

    # Run appropriate mode
    if args.similar_to:
        run_similar(
            args.similar_to,
            search_engine,
            top_k=args.top_k,
            show_preview=not args.no_preview
        )
    elif args.query:
        run_search(
            args.query,
            search_engine,
            top_k=args.top_k,
            min_score=args.min_score,
            show_preview=not args.no_preview
        )
    else:
        interactive_mode(search_engine, top_k=args.top_k)


if __name__ == "__main__":
    main()
