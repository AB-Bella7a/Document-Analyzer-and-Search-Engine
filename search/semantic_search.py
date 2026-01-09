"""
Semantic Search Engine using FAISS

This module implements semantic search functionality:
1. Build FAISS index from document embeddings
2. Convert queries to embeddings
3. Perform efficient similarity search
4. Return ranked results with metadata

FAISS (Facebook AI Similarity Search) is optimized for efficient
similarity search and clustering of dense vectors.

Example:
    >>> from search import SemanticSearch
    >>>
    >>> # Initialize and build index
    >>> search_engine = SemanticSearch()
    >>> search_engine.build_index()
    >>>
    >>> # Perform search
    >>> results = search_engine.search("machine learning algorithms", top_k=5)
    >>> for result in results:
    ...     print(f"{result.document_id}: {result.similarity_score:.3f}")
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import sys

# Import configuration and dependencies
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from embeddings import EmbeddingGenerator, DocumentEmbedding
from preprocessing import DocumentPreprocessor


@dataclass
class SearchResult:
    """
    A single search result.

    Attributes:
        document_id: Unique identifier of the matched document
        similarity_score: Cosine similarity score (0-1, higher is better)
        rank: Position in results (1-based)
        original_filename: Original filename of the document
        document_type: Type of document ('pdf' or 'image')
        word_count: Number of words in the document
        text_preview: Preview of document text (first N characters)
        metadata: Additional document metadata
    """
    document_id: str
    similarity_score: float
    rank: int
    original_filename: str = ""
    document_type: str = ""
    word_count: int = 0
    text_preview: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FAISSIndex:
    """
    FAISS index wrapper for vector similarity search.

    This class manages the FAISS index, handling:
    - Index creation and configuration
    - Adding vectors
    - Searching
    - Saving/loading to disk

    Supports two index types:
    - 'flat': Exact search (brute force, most accurate)
    - 'ivf': Approximate search (faster for large datasets)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 100
    ):
        """
        Initialize FAISS index.

        Args:
            dimension: Dimension of vectors to index
            index_type: Type of index ('flat' or 'ivf')
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.is_trained = False

        self._create_index()

    def _create_index(self):
        """Create the FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for search. "
                "Install with: pip install faiss-cpu"
            )

        if self.index_type == "flat":
            # Flat index: exact search using inner product (for normalized vectors)
            # This is equivalent to cosine similarity for normalized vectors
            self.index = faiss.IndexFlatIP(self.dimension)
            self.is_trained = True
            print(f"[Search] Created Flat index (dimension: {self.dimension})")

        elif self.index_type == "ivf":
            # IVF index: approximate search using clustering
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.is_trained = False
            print(f"[Search] Created IVF index (dimension: {self.dimension}, "
                  f"nlist: {self.nlist})")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def train(self, vectors: np.ndarray):
        """
        Train the index (required for IVF index).

        Args:
            vectors: Training vectors of shape (n, dimension)
        """
        if self.index_type == "flat":
            # Flat index doesn't need training
            return

        if vectors.shape[0] < self.nlist:
            print(f"[Search] Warning: Training with {vectors.shape[0]} vectors, "
                  f"but nlist={self.nlist}. Consider reducing nlist.")

        print(f"[Search] Training IVF index with {vectors.shape[0]} vectors...")
        self.index.train(vectors)
        self.is_trained = True
        print("[Search] Index training complete")

    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.

        Args:
            vectors: Vectors to add of shape (n, dimension)
        """
        if not self.is_trained and self.index_type == "ivf":
            raise RuntimeError("Index must be trained before adding vectors")

        # Ensure vectors are normalized (for cosine similarity)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = vectors / norms

        self.index.add(normalized.astype(np.float32))
        print(f"[Search] Added {vectors.shape[0]} vectors to index")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> tuple:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices)
            - distances: Similarity scores of shape (top_k,)
            - indices: Indices of matched vectors of shape (top_k,)
        """
        # Reshape query if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Perform search
        distances, indices = self.index.search(
            query_vector.astype(np.float32),
            top_k
        )

        return distances[0], indices[0]

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def save(self, path: Path):
        """Save index to disk."""
        import faiss
        faiss.write_index(self.index, str(path))
        print(f"[Search] Index saved to: {path}")

    def load(self, path: Path):
        """Load index from disk."""
        import faiss
        self.index = faiss.read_index(str(path))
        self.is_trained = True
        print(f"[Search] Index loaded from: {path} ({self.index.ntotal} vectors)")


class SemanticSearch:
    """
    Semantic Search Engine

    This class provides the main interface for semantic search:
    - Build search index from document embeddings
    - Search with natural language queries
    - Return ranked results with metadata

    Example:
        >>> search = SemanticSearch()
        >>>
        >>> # Build index from embeddings
        >>> search.build_index()
        >>>
        >>> # Search for documents
        >>> results = search.search("machine learning", top_k=5)
        >>> for r in results:
        ...     print(f"{r.rank}. {r.original_filename} ({r.similarity_score:.3f})")
    """

    def __init__(
        self,
        embeddings_dir: Optional[Path] = None,
        index_type: Optional[str] = None
    ):
        """
        Initialize the semantic search engine.

        Args:
            embeddings_dir: Directory containing document embeddings
            index_type: FAISS index type ('flat' or 'ivf')
        """
        self.embeddings_dir = embeddings_dir or settings.EMBEDDINGS_DIR
        self.index_type = index_type or settings.FAISS_INDEX_TYPE

        # Index and document mapping
        self.index: Optional[FAISSIndex] = None
        self.document_ids: List[str] = []  # Maps index position to document ID
        self.document_metadata: Dict[str, Dict] = {}  # Document ID to metadata

        # Initialize embedding generator for query encoding
        self.embedding_generator = EmbeddingGenerator(
            embeddings_dir=self.embeddings_dir
        )

        # Initialize preprocessor for document text access
        self.preprocessor = DocumentPreprocessor()

        # Index file paths
        self.index_path = self.embeddings_dir / "faiss_index.bin"
        self.metadata_path = self.embeddings_dir / "index_metadata.json"

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build the FAISS index from document embeddings.

        Args:
            force_rebuild: If True, rebuild even if index exists

        Returns:
            Number of documents indexed
        """
        # Check if index already exists
        if not force_rebuild and self.index_path.exists():
            print("[Search] Loading existing index...")
            self.load_index()
            return len(self.document_ids)

        print("[Search] Building new index...")

        # Load all embeddings
        embeddings_list = []
        self.document_ids = []
        self.document_metadata = {}

        for embedding in self.embedding_generator.list_embeddings():
            if embedding.text_embedding is not None:
                embeddings_list.append(embedding.text_embedding)
                self.document_ids.append(embedding.document_id)

                # Store metadata
                self.document_metadata[embedding.document_id] = {
                    "embedding_model": embedding.embedding_model,
                    "created_timestamp": embedding.created_timestamp,
                    **embedding.metadata
                }

        if not embeddings_list:
            print("[Search] No embeddings found to index")
            return 0

        # Stack embeddings into matrix
        embeddings_matrix = np.vstack(embeddings_list)
        dimension = embeddings_matrix.shape[1]

        print(f"[Search] Indexing {len(embeddings_list)} documents "
              f"(dimension: {dimension})")

        # Create and populate index
        self.index = FAISSIndex(
            dimension=dimension,
            index_type=self.index_type,
            nlist=min(settings.FAISS_NLIST, len(embeddings_list))
        )

        # Train if needed (for IVF index)
        if self.index_type == "ivf":
            self.index.train(embeddings_matrix)

        # Add vectors
        self.index.add(embeddings_matrix)

        # Save index and metadata
        self.save_index()

        print(f"[Search] Index built with {self.index.size} vectors")
        return self.index.size

    def save_index(self):
        """Save index and metadata to disk."""
        if self.index is None:
            return

        # Save FAISS index
        self.index.save(self.index_path)

        # Save metadata
        metadata = {
            "document_ids": self.document_ids,
            "document_metadata": self.document_metadata,
            "index_type": self.index_type,
            "created_timestamp": datetime.utcnow().isoformat()
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Search] Index metadata saved to: {self.metadata_path}")

    def load_index(self):
        """Load index and metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Index files not found. Run build_index() first.")

        # Load metadata
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.document_ids = metadata["document_ids"]
        self.document_metadata = metadata["document_metadata"]
        self.index_type = metadata.get("index_type", "flat")

        # Load FAISS index
        dimension = settings.TEXT_EMBEDDING_DIMENSION
        self.index = FAISSIndex(dimension=dimension, index_type=self.index_type)
        self.index.load(self.index_path)

        print(f"[Search] Index loaded: {self.index.size} vectors")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default from config)
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult objects, ranked by relevance

        Example:
            >>> results = search.search("neural networks", top_k=3)
            >>> for r in results:
            ...     print(f"{r.rank}. {r.original_filename}: {r.similarity_score:.3f}")
        """
        if self.index is None or self.index.size == 0:
            print("[Search] Index is empty. Building index...")
            self.build_index()

            if self.index is None or self.index.size == 0:
                return []

        # Set default top_k
        top_k = top_k or settings.DEFAULT_TOP_K
        top_k = min(top_k, settings.MAX_TOP_K, self.index.size)

        print(f"[Search] Searching for: '{query}' (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            # Skip invalid indices or low scores
            if idx < 0 or idx >= len(self.document_ids):
                continue
            if score < min_score:
                continue

            doc_id = self.document_ids[idx]
            metadata = self.document_metadata.get(doc_id, {})

            # Get text preview from processed document
            text_preview = ""
            processed = self.preprocessor.get_processed(doc_id)
            if processed:
                text_preview = processed.cleaned_text[:200] + "..." \
                    if len(processed.cleaned_text) > 200 else processed.cleaned_text

            result = SearchResult(
                document_id=doc_id,
                similarity_score=float(score),
                rank=rank,
                original_filename=metadata.get("original_filename", ""),
                document_type=metadata.get("document_type", ""),
                word_count=metadata.get("word_count", 0),
                text_preview=text_preview,
                metadata=metadata
            )
            results.append(result)

        print(f"[Search] Found {len(results)} results")
        return results

    def search_by_document(
        self,
        document_id: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.

        Args:
            document_id: ID of the reference document
            top_k: Number of results to return

        Returns:
            List of SearchResult objects (excluding the reference document)
        """
        # Load document embedding
        doc_embedding = self.embedding_generator.get_embedding(document_id)
        if doc_embedding is None or doc_embedding.text_embedding is None:
            raise ValueError(f"No embedding found for document: {document_id}")

        top_k = (top_k or settings.DEFAULT_TOP_K) + 1  # +1 to exclude self

        # Search using document embedding
        scores, indices = self.index.search(doc_embedding.text_embedding, top_k)

        # Build results (excluding self)
        results = []
        rank = 1
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.document_ids):
                continue

            doc_id = self.document_ids[idx]
            if doc_id == document_id:  # Skip self
                continue

            metadata = self.document_metadata.get(doc_id, {})

            result = SearchResult(
                document_id=doc_id,
                similarity_score=float(score),
                rank=rank,
                original_filename=metadata.get("original_filename", ""),
                document_type=metadata.get("document_type", ""),
                word_count=metadata.get("word_count", 0),
                metadata=metadata
            )
            results.append(result)
            rank += 1

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "not_built", "total_documents": 0}

        return {
            "status": "ready",
            "total_documents": self.index.size,
            "index_type": self.index_type,
            "document_ids": self.document_ids,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path)
        }

    def delete_from_index(self, document_id: str) -> bool:
        """
        Remove a document from the index.

        Note: FAISS flat indices don't support deletion, so this
        marks the document as deleted and requires index rebuild.

        Args:
            document_id: ID of document to remove

        Returns:
            True if document was found and marked for deletion
        """
        if document_id not in self.document_ids:
            return False

        # Mark for deletion (actual removal requires rebuild)
        idx = self.document_ids.index(document_id)
        self.document_ids[idx] = "__DELETED__"
        if document_id in self.document_metadata:
            del self.document_metadata[document_id]

        print(f"[Search] Document {document_id} marked for deletion. "
              "Run build_index(force_rebuild=True) to apply changes.")
        return True


# =============================================================================
# Standalone execution for testing
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Engine")
    parser.add_argument("--build", "-b", action="store_true", help="Build search index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--search", "-s", help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--similar-to", help="Find documents similar to given ID")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    args = parser.parse_args()

    search_engine = SemanticSearch()

    if args.build or args.rebuild:
        print("\n=== Building Search Index ===")
        count = search_engine.build_index(force_rebuild=args.rebuild)
        print(f"Indexed {count} documents")

    elif args.search:
        print(f"\n=== Searching: '{args.search}' ===")
        results = search_engine.search(args.search, top_k=args.top_k)

        if not results:
            print("No results found.")
        else:
            for result in results:
                print(f"\n{result.rank}. {result.original_filename}")
                print(f"   Score: {result.similarity_score:.4f}")
                print(f"   Type: {result.document_type}, Words: {result.word_count}")
                if result.text_preview:
                    print(f"   Preview: {result.text_preview[:100]}...")

    elif args.similar_to:
        print(f"\n=== Finding Similar Documents ===")
        results = search_engine.search_by_document(args.similar_to, top_k=args.top_k)

        if not results:
            print("No similar documents found.")
        else:
            for result in results:
                print(f"{result.rank}. {result.original_filename}: {result.similarity_score:.4f}")

    elif args.stats:
        print("\n=== Index Statistics ===")
        stats = search_engine.get_index_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        print("Usage:")
        print("  python semantic_search.py --build           # Build index")
        print("  python semantic_search.py --rebuild         # Force rebuild")
        print("  python semantic_search.py --search 'query'  # Search")
        print("  python semantic_search.py --similar-to <id> # Find similar docs")
        print("  python semantic_search.py --stats           # Show statistics")
