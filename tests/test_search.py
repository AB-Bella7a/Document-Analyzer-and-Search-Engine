"""
Tests for Semantic Search Module

Tests cover:
- FAISS index operations
- Semantic search functionality
- Search result ranking
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from search import SemanticSearch, SearchResult
from search.semantic_search import FAISSIndex


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            document_id="doc_test123",
            similarity_score=0.85,
            rank=1,
            original_filename="test.pdf",
            document_type="pdf",
            word_count=100,
            text_preview="This is a test..."
        )

        assert result.document_id == "doc_test123"
        assert result.similarity_score == 0.85
        assert result.rank == 1

    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        result = SearchResult(
            document_id="doc_test123",
            similarity_score=0.85,
            rank=1
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["document_id"] == "doc_test123"
        assert data["similarity_score"] == 0.85


class TestFAISSIndex:
    """Tests for FAISSIndex wrapper class."""

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        np.random.seed(42)
        return np.random.randn(10, 384).astype(np.float32)

    def test_flat_index_creation(self):
        """Test creating a flat FAISS index."""
        index = FAISSIndex(dimension=384, index_type="flat")

        assert index.dimension == 384
        assert index.index_type == "flat"
        assert index.is_trained is True  # Flat index doesn't need training
        assert index.size == 0

    def test_add_vectors(self, sample_vectors):
        """Test adding vectors to index."""
        index = FAISSIndex(dimension=384, index_type="flat")
        index.add(sample_vectors)

        assert index.size == 10

    def test_search_vectors(self, sample_vectors):
        """Test searching for similar vectors."""
        index = FAISSIndex(dimension=384, index_type="flat")
        index.add(sample_vectors)

        # Search with the first vector (should find itself as most similar)
        query = sample_vectors[0]
        distances, indices = index.search(query, top_k=3)

        assert len(distances) == 3
        assert len(indices) == 3
        assert indices[0] == 0  # First result should be itself

    def test_save_and_load(self, sample_vectors, temp_dir):
        """Test saving and loading index."""
        index = FAISSIndex(dimension=384, index_type="flat")
        index.add(sample_vectors)

        # Save index
        save_path = temp_dir / "test_index.bin"
        index.save(save_path)

        # Load into new index
        new_index = FAISSIndex(dimension=384, index_type="flat")
        new_index.load(save_path)

        assert new_index.size == 10


class TestSemanticSearch:
    """Tests for SemanticSearch class."""

    def test_search_engine_initialization(self, test_data_dir, mock_settings):
        """Test search engine initializes correctly."""
        search = SemanticSearch(
            embeddings_dir=test_data_dir / "embeddings"
        )

        assert search.embeddings_dir.exists()
        assert search.document_ids == []

    def test_get_index_stats_empty(self, test_data_dir, mock_settings):
        """Test getting stats for empty index."""
        search = SemanticSearch(
            embeddings_dir=test_data_dir / "embeddings"
        )

        stats = search.get_index_stats()

        assert stats["status"] == "not_built"
        assert stats["total_documents"] == 0

    def test_build_index_empty(self, test_data_dir, mock_settings):
        """Test building index with no embeddings."""
        search = SemanticSearch(
            embeddings_dir=test_data_dir / "embeddings"
        )

        count = search.build_index()

        assert count == 0

    def test_search_empty_index(self, test_data_dir, mock_settings):
        """Test searching with empty index."""
        search = SemanticSearch(
            embeddings_dir=test_data_dir / "embeddings"
        )

        results = search.search("test query")

        assert results == []

    def test_save_and_load_index(self, test_data_dir, mock_settings):
        """Test saving and loading search index metadata."""
        search = SemanticSearch(
            embeddings_dir=test_data_dir / "embeddings"
        )

        # Build empty index
        search.build_index()

        # Check files were created
        assert search.metadata_path.exists()


class TestSearchIntegration:
    """Integration tests for search with mock embeddings."""

    @pytest.fixture
    def mock_embeddings(self, test_data_dir):
        """Create mock embedding files."""
        import json

        embeddings_dir = test_data_dir / "embeddings"

        # Create mock embeddings for 3 documents
        for i in range(3):
            doc_id = f"doc_test{i}"

            # Create embedding array
            np.random.seed(i)
            text_embedding = np.random.randn(384).astype(np.float32)
            text_embedding = text_embedding / np.linalg.norm(text_embedding)

            # Save embedding
            np.savez_compressed(
                embeddings_dir / f"{doc_id}.npz",
                text_embedding=text_embedding
            )

            # Save metadata
            meta = {
                "document_id": doc_id,
                "embedding_model": "test-model",
                "text_dimension": 384,
                "image_dimension": 0,
                "created_timestamp": "2024-01-01T00:00:00",
                "metadata": {
                    "original_filename": f"test{i}.pdf",
                    "document_type": "pdf",
                    "word_count": 100 + i * 50
                }
            }
            with open(embeddings_dir / f"{doc_id}_meta.json", "w") as f:
                json.dump(meta, f)

        return embeddings_dir

    def test_build_index_with_embeddings(self, test_data_dir, mock_embeddings, mock_settings):
        """Test building index with actual embeddings."""
        search = SemanticSearch(embeddings_dir=mock_embeddings)

        count = search.build_index()

        assert count == 3
        assert len(search.document_ids) == 3

    def test_search_returns_ranked_results(self, test_data_dir, mock_embeddings, mock_settings):
        """Test that search returns properly ranked results."""
        search = SemanticSearch(embeddings_dir=mock_embeddings)
        search.build_index()

        # Create a mock query embedding (similar to doc_test0)
        np.random.seed(0)
        query_embedding = np.random.randn(384).astype(np.float32)

        # Mock the embedding generator
        from unittest.mock import patch, MagicMock

        mock_gen = MagicMock()
        mock_gen.embed_query.return_value = query_embedding

        with patch.object(search, 'embedding_generator', mock_gen):
            results = search.search("test query", top_k=3)

        assert len(results) > 0
        # Results should be ranked (scores in descending order)
        if len(results) > 1:
            assert results[0].similarity_score >= results[1].similarity_score
