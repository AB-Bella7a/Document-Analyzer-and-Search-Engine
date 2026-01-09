"""
Document Preprocessing Module

This module handles the extraction and cleaning of content from documents.
It provides functionality to:
- Extract text from PDFs using pdfplumber
- Perform OCR on images using pytesseract
- Clean and normalize extracted text
- Save processed output as structured JSON

Usage:
    from preprocessing import DocumentPreprocessor

    preprocessor = DocumentPreprocessor()
    result = preprocessor.process_document("doc_abc123")
"""

from .preprocessor import DocumentPreprocessor, ProcessedDocument

__all__ = ["DocumentPreprocessor", "ProcessedDocument"]
