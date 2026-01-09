"""
Tests for REST API Endpoints

Tests cover:
- Health check endpoint
- Document upload and management
- Search functionality
- Pipeline operations
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip tests if FastAPI not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_settings):
    """Create a test client for the API."""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and status endpoints."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_root_redirects_to_docs(self, client):
        """Test root URL redirects to documentation."""
        response = client.get("/", follow_redirects=False)

        assert response.status_code == 307
        assert "/docs" in response.headers.get("location", "")

    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/api")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_list_documents_empty(self, client, mock_settings):
        """Test listing documents when none exist."""
        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["documents"] == []

    def test_upload_document_invalid_type(self, client, mock_settings):
        """Test uploading unsupported file type."""
        files = {"file": ("test.xyz", b"content", "application/octet-stream")}
        response = client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_pdf_document(self, client, mock_settings, sample_pdf):
        """Test uploading a PDF document."""
        with open(sample_pdf, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            response = client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document_id" in data
        assert data["metadata"]["document_type"] == "pdf"

    def test_get_document_not_found(self, client, mock_settings):
        """Test getting non-existent document."""
        response = client.get("/api/v1/documents/nonexistent_doc")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_document_not_found(self, client, mock_settings):
        """Test deleting non-existent document."""
        response = client.delete("/api/v1/documents/nonexistent_doc")

        assert response.status_code == 404


class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_post(self, client, mock_settings):
        """Test POST search endpoint."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test query", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "total_results" in data
        assert "results" in data

    def test_search_get(self, client, mock_settings):
        """Test GET search endpoint."""
        response = client.get("/api/v1/search?q=test+query&top_k=5")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"

    def test_search_empty_query(self, client, mock_settings):
        """Test search with empty query fails."""
        response = client.post(
            "/api/v1/search",
            json={"query": "", "top_k": 5}
        )

        assert response.status_code == 422  # Validation error

    def test_search_invalid_top_k(self, client, mock_settings):
        """Test search with invalid top_k."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "top_k": 1000}  # Exceeds max
        )

        assert response.status_code == 422


class TestPipelineEndpoints:
    """Tests for pipeline operation endpoints."""

    def test_process_document_not_found(self, client, mock_settings):
        """Test processing non-existent document."""
        response = client.post(
            "/api/v1/pipeline/process",
            json={"document_id": "nonexistent_doc"}
        )

        assert response.status_code == 404

    def test_generate_embeddings_not_found(self, client, mock_settings):
        """Test generating embeddings for non-existent document."""
        response = client.post(
            "/api/v1/pipeline/embed",
            json={"document_id": "nonexistent_doc"}
        )

        assert response.status_code == 404

    def test_build_index(self, client, mock_settings):
        """Test building search index."""
        response = client.post("/api/v1/pipeline/build-index")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "index_size" in data


class TestAPISchemas:
    """Tests for API request/response schemas."""

    def test_search_request_validation(self, client, mock_settings):
        """Test search request validation."""
        # Valid request
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "top_k": 5, "min_score": 0.5}
        )
        assert response.status_code == 200

        # Invalid min_score
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "min_score": 1.5}
        )
        assert response.status_code == 422

    def test_pipeline_request_validation(self, client, mock_settings):
        """Test pipeline request validation."""
        response = client.post(
            "/api/v1/pipeline/run",
            json={"process_all": True, "rebuild_index": False}
        )

        assert response.status_code == 200
