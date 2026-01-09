"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
Fixtures create temporary directories and mock data for testing
without affecting the real data directories.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test data.

    This fixture creates a fresh temporary directory for each test,
    ensuring test isolation.
    """
    temp_path = Path(tempfile.mkdtemp(prefix="doc_analyzer_test_"))
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def test_data_dir(temp_dir: Path) -> Path:
    """
    Create a test data directory structure.

    Creates the same directory structure as the real data directory:
    - raw/
    - processed/
    - embeddings/
    - uploads/
    """
    (temp_dir / "raw" / "pdfs").mkdir(parents=True)
    (temp_dir / "raw" / "images").mkdir(parents=True)
    (temp_dir / "raw" / "metadata").mkdir(parents=True)
    (temp_dir / "processed").mkdir()
    (temp_dir / "embeddings").mkdir()
    (temp_dir / "uploads").mkdir()
    (temp_dir / "storage").mkdir()

    return temp_dir


@pytest.fixture(scope="function")
def sample_pdf(temp_dir: Path) -> Path:
    """
    Create a simple test PDF file.

    Note: This creates a minimal valid PDF for testing.
    For real PDF extraction tests, you may need actual PDFs.
    """
    pdf_path = temp_dir / "sample.pdf"

    # Minimal PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""

    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    return pdf_path


@pytest.fixture(scope="function")
def sample_image(temp_dir: Path) -> Path:
    """
    Create a simple test image file.

    Creates a small PNG image for testing.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    image_path = temp_dir / "sample.png"

    # Create a simple 100x100 image with some text area
    img = Image.new("RGB", (100, 100), color="white")
    img.save(image_path)

    return image_path


@pytest.fixture(scope="function")
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    text_path = temp_dir / "sample.txt"
    text_path.write_text("This is a sample text file for testing.")
    return text_path


@pytest.fixture(scope="function")
def mock_settings(temp_dir: Path, monkeypatch):
    """
    Mock the settings to use temporary directories.

    This prevents tests from modifying real data directories.
    """
    from config import settings

    # Override directory settings
    monkeypatch.setattr(settings, "DATA_DIR", temp_dir)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", temp_dir / "raw")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", temp_dir / "processed")
    monkeypatch.setattr(settings, "EMBEDDINGS_DIR", temp_dir / "embeddings")
    monkeypatch.setattr(settings, "UPLOADS_DIR", temp_dir / "uploads")

    # Disable S3 for tests
    monkeypatch.setattr(settings, "USE_S3_STORAGE", False)

    return settings


@pytest.fixture(scope="function")
def sample_documents(test_data_dir: Path) -> dict:
    """
    Create a set of sample documents for testing.

    Returns a dictionary with paths to different document types.
    """
    docs = {}

    # Create sample PDF (minimal valid PDF)
    pdf_path = test_data_dir / "uploads" / "test_document.pdf"
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
170
%%EOF"""
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    docs["pdf"] = pdf_path

    # Create sample image
    try:
        from PIL import Image
        image_path = test_data_dir / "uploads" / "test_image.png"
        img = Image.new("RGB", (200, 200), color="white")
        img.save(image_path)
        docs["image"] = image_path
    except ImportError:
        pass

    return docs
