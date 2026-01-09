"""
Embedding Generation Module

This module implements embedding generation for the Document Analyzer:
1. Text embeddings using sentence-transformers (all-MiniLM-L6-v2)
2. Image embeddings using CLIP (openai/clip-vit-base-patch32)
3. Batch processing for efficiency
4. Persistent storage of embeddings

The module uses pre-trained models without training from scratch,
but is structured to allow fine-tuning additions later.

Example:
    >>> from embeddings import EmbeddingGenerator
    >>>
    >>> generator = EmbeddingGenerator()
    >>>
    >>> # Generate text embedding
    >>> embedding = generator.embed_text("This is a test document.")
    >>>
    >>> # Generate embeddings for a processed document
    >>> doc_embedding = generator.generate_document_embedding("doc_abc123")
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import sys

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from preprocessing import DocumentPreprocessor, ProcessedDocument


@dataclass
class DocumentEmbedding:
    """
    Stores embedding data for a document.

    Attributes:
        document_id: Unique document identifier
        text_embedding: Vector embedding of document text
        image_embedding: Vector embedding of document image (if applicable)
        embedding_model: Name of the model used for text embedding
        image_model: Name of the model used for image embedding
        text_dimension: Dimension of text embedding vector
        image_dimension: Dimension of image embedding vector
        created_timestamp: When embeddings were generated
        metadata: Additional metadata (word count, source, etc.)
    """
    document_id: str
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    embedding_model: str = ""
    image_model: str = ""
    text_dimension: int = 0
    image_dimension: int = 0
    created_timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_timestamp:
            self.created_timestamp = datetime.utcnow().isoformat()
        if self.text_embedding is not None:
            self.text_dimension = len(self.text_embedding)
        if self.image_embedding is not None:
            self.image_dimension = len(self.image_embedding)

    def save(self, directory: Path) -> Path:
        """
        Save embedding to disk.

        Embeddings are saved as .npz files (numpy compressed format)
        with metadata in a companion .json file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy file
        npz_path = directory / f"{self.document_id}.npz"
        save_dict = {}
        if self.text_embedding is not None:
            save_dict["text_embedding"] = self.text_embedding
        if self.image_embedding is not None:
            save_dict["image_embedding"] = self.image_embedding

        np.savez_compressed(npz_path, **save_dict)

        # Save metadata as JSON
        meta_path = directory / f"{self.document_id}_meta.json"
        meta_dict = {
            "document_id": self.document_id,
            "embedding_model": self.embedding_model,
            "image_model": self.image_model,
            "text_dimension": self.text_dimension,
            "image_dimension": self.image_dimension,
            "created_timestamp": self.created_timestamp,
            "metadata": self.metadata
        }
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

        return npz_path

    @classmethod
    def load(cls, directory: Path, document_id: str) -> Optional["DocumentEmbedding"]:
        """
        Load embedding from disk.

        Args:
            directory: Directory containing embedding files
            document_id: ID of document to load

        Returns:
            DocumentEmbedding if found, None otherwise
        """
        directory = Path(directory)
        npz_path = directory / f"{document_id}.npz"
        meta_path = directory / f"{document_id}_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            return None

        # Load embeddings
        data = np.load(npz_path)
        text_emb = data.get("text_embedding")
        image_emb = data.get("image_embedding")

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        return cls(
            document_id=document_id,
            text_embedding=text_emb,
            image_embedding=image_emb,
            embedding_model=meta.get("embedding_model", ""),
            image_model=meta.get("image_model", ""),
            text_dimension=meta.get("text_dimension", 0),
            image_dimension=meta.get("image_dimension", 0),
            created_timestamp=meta.get("created_timestamp", ""),
            metadata=meta.get("metadata", {})
        )


class TextEmbedder:
    """
    Text embedding generator using sentence-transformers.

    This class wraps sentence-transformers models to generate
    dense vector representations of text.

    The default model (all-MiniLM-L6-v2) provides a good balance
    of quality and speed, with 384-dimensional embeddings.

    Example:
        >>> embedder = TextEmbedder()
        >>> embedding = embedder.embed("Hello world")
        >>> print(embedding.shape)
        (384,)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the text embedder.

        Args:
            model_name: Name of sentence-transformers model to use
                       (default from config: all-MiniLM-L6-v2)
        """
        self.model_name = model_name or settings.TEXT_EMBEDDING_MODEL
        self.model = None
        self._dimension = settings.TEXT_EMBEDDING_DIMENSION

    def _load_model(self):
        """Load the sentence-transformers model (lazy loading)."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Embeddings] Loading text model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self._dimension = self.model.get_sentence_embedding_dimension()
                print(f"[Embeddings] Text model loaded (dimension: {self._dimension})")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        self._load_model()

        # Handle empty text
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)

        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default from config)
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (num_texts, dimension)
        """
        self._load_model()

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        # Handle empty input
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )

        return embeddings.astype(np.float32)


class ImageEmbedder:
    """
    Image embedding generator using CLIP.

    CLIP (Contrastive Language-Image Pre-training) creates embeddings
    that align images and text in the same vector space, enabling
    cross-modal search.

    Example:
        >>> embedder = ImageEmbedder()
        >>> embedding = embedder.embed("path/to/image.png")
        >>> print(embedding.shape)
        (512,)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the image embedder.

        Args:
            model_name: Name of CLIP model to use
                       (default from config: openai/clip-vit-base-patch32)
        """
        self.model_name = model_name or settings.IMAGE_EMBEDDING_MODEL
        self.model = None
        self.processor = None
        self._dimension = settings.IMAGE_EMBEDDING_DIMENSION

    def _load_model(self):
        """Load the CLIP model (lazy loading)."""
        if self.model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch

                print(f"[Embeddings] Loading image model: {self.model_name}")
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)

                # Get embedding dimension from model config
                self._dimension = self.model.config.projection_dim
                print(f"[Embeddings] Image model loaded (dimension: {self._dimension})")

                # Set to evaluation mode
                self.model.eval()

            except ImportError:
                raise ImportError(
                    "transformers and torch are required. "
                    "Install with: pip install transformers torch"
                )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def embed(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Numpy array of shape (dimension,)
        """
        self._load_model()

        try:
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        # Load and process image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize
        embedding = image_features.numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)

    def embed_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images.

        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing

        Returns:
            Numpy array of shape (num_images, dimension)
        """
        self._load_model()

        try:
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError("Pillow and torch are required")

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        if not image_paths:
            return np.zeros((0, self._dimension), dtype=np.float32)

        embeddings = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                image = Image.open(path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)

            # Process batch
            inputs = self.processor(images=images, return_tensors="pt")

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize and add to results
            batch_embeddings = image_features.numpy()
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, axis=1, keepdims=True
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP text embedding.

        This creates a text embedding in the same space as image embeddings,
        enabling text-to-image search.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        self._load_model()

        import torch

        # Process text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize
        embedding = text_features.numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)


class EmbeddingGenerator:
    """
    Main embedding generation pipeline.

    This class orchestrates the generation of embeddings for documents,
    combining text and image embedding capabilities.

    Example:
        >>> generator = EmbeddingGenerator()
        >>>
        >>> # Generate embedding for a processed document
        >>> doc_emb = generator.generate_document_embedding("doc_abc123")
        >>>
        >>> # Generate embeddings for all documents
        >>> results = generator.generate_all_embeddings()
    """

    def __init__(
        self,
        embeddings_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None
    ):
        """
        Initialize the embedding generator.

        Args:
            embeddings_dir: Directory to store embeddings (default from config)
            processed_dir: Directory containing processed documents
        """
        self.embeddings_dir = embeddings_dir or settings.EMBEDDINGS_DIR
        self.processed_dir = processed_dir or settings.PROCESSED_DATA_DIR

        # Ensure directories exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedders (lazy loaded)
        self._text_embedder = None
        self._image_embedder = None

        # Initialize preprocessor for document access
        self.preprocessor = DocumentPreprocessor(processed_dir=self.processed_dir)

    @property
    def text_embedder(self) -> TextEmbedder:
        """Get or create text embedder."""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder()
        return self._text_embedder

    @property
    def image_embedder(self) -> ImageEmbedder:
        """Get or create image embedder."""
        if self._image_embedder is None:
            self._image_embedder = ImageEmbedder()
        return self._image_embedder

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Numpy array embedding
        """
        return self.text_embedder.embed(text)

    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate embedding for image.

        Args:
            image_path: Path to image file

        Returns:
            Numpy array embedding
        """
        return self.image_embedder.embed(image_path)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Uses text embedder for text-based semantic search.

        Args:
            query: Search query text

        Returns:
            Numpy array embedding
        """
        return self.embed_text(query)

    def generate_document_embedding(
        self,
        document_id: str,
        include_image: bool = True
    ) -> DocumentEmbedding:
        """
        Generate embeddings for a processed document.

        Args:
            document_id: ID of the processed document
            include_image: Whether to include image embedding for image documents

        Returns:
            DocumentEmbedding with text and optionally image embeddings

        Raises:
            ValueError: If document not found or not processed
        """
        print(f"[Embeddings] Generating embeddings for: {document_id}")

        # Load processed document
        processed = self.preprocessor.get_processed(document_id)
        if not processed:
            raise ValueError(f"Processed document not found: {document_id}")

        # Generate text embedding from cleaned text
        text_embedding = None
        if processed.cleaned_text:
            print("[Embeddings] Generating text embedding...")
            text_embedding = self.text_embedder.embed(processed.cleaned_text)

        # Generate image embedding if applicable
        image_embedding = None
        if include_image and processed.document_type == "image":
            # Get original image path from ingestion metadata
            from ingestion import DocumentIngester
            ingester = DocumentIngester()
            metadata = ingester.get_metadata(document_id)
            if metadata and Path(metadata.storage_location).exists():
                print("[Embeddings] Generating image embedding...")
                image_embedding = self.image_embedder.embed(metadata.storage_location)

        # Create document embedding
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            embedding_model=self.text_embedder.model_name,
            image_model=self.image_embedder.model_name if image_embedding is not None else "",
            metadata={
                "original_filename": processed.original_filename,
                "document_type": processed.document_type,
                "word_count": processed.word_count
            }
        )

        # Save embedding
        doc_embedding.save(self.embeddings_dir)
        print(f"[Embeddings] Saved embedding for: {document_id}")

        return doc_embedding

    def get_embedding(self, document_id: str) -> Optional[DocumentEmbedding]:
        """
        Load a document embedding by ID.

        Args:
            document_id: ID of the document

        Returns:
            DocumentEmbedding if found, None otherwise
        """
        return DocumentEmbedding.load(self.embeddings_dir, document_id)

    def has_embedding(self, document_id: str) -> bool:
        """Check if a document has embeddings generated."""
        return (self.embeddings_dir / f"{document_id}.npz").exists()

    def generate_all_embeddings(
        self,
        include_image: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all processed documents.

        Args:
            include_image: Whether to include image embeddings
            skip_existing: Skip documents that already have embeddings

        Returns:
            Dictionary with 'successful' and 'failed' lists
        """
        results = {
            "successful": [],
            "failed": []
        }

        # Get all processed documents
        processed_docs = self.preprocessor.list_processed()
        print(f"[Embeddings] Found {len(processed_docs)} processed documents")

        for doc in processed_docs:
            # Skip if already has embedding
            if skip_existing and self.has_embedding(doc.document_id):
                print(f"[Embeddings] Skipping (already embedded): {doc.document_id}")
                continue

            try:
                doc_emb = self.generate_document_embedding(
                    doc.document_id,
                    include_image=include_image
                )
                results["successful"].append({
                    "document_id": doc.document_id,
                    "text_dimension": doc_emb.text_dimension,
                    "image_dimension": doc_emb.image_dimension
                })
            except Exception as e:
                results["failed"].append({
                    "document_id": doc.document_id,
                    "error": str(e)
                })
                print(f"[Embeddings] Error: {e}")

        print(f"[Embeddings] Complete: {len(results['successful'])} success, "
              f"{len(results['failed'])} failed")

        return results

    def list_embeddings(self) -> List[DocumentEmbedding]:
        """
        List all generated embeddings.

        Returns:
            List of DocumentEmbedding objects
        """
        embeddings = []

        for npz_file in self.embeddings_dir.glob("*.npz"):
            doc_id = npz_file.stem
            embedding = self.get_embedding(doc_id)
            if embedding:
                embeddings.append(embedding)

        # Sort by creation timestamp
        embeddings.sort(key=lambda x: x.created_timestamp, reverse=True)

        return embeddings

    def get_all_text_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all text embeddings into memory.

        Returns:
            Dictionary mapping document_id to text embedding
        """
        embeddings = {}

        for embedding in self.list_embeddings():
            if embedding.text_embedding is not None:
                embeddings[embedding.document_id] = embedding.text_embedding

        return embeddings


# =============================================================================
# Standalone execution for testing
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Generation Pipeline")
    parser.add_argument("--document-id", "-d", help="Generate embedding for specific document")
    parser.add_argument("--text", "-t", help="Generate embedding for text string")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all embeddings")
    parser.add_argument("--list", "-l", action="store_true", help="List all embeddings")
    args = parser.parse_args()

    generator = EmbeddingGenerator()

    if args.list:
        print("\n=== Document Embeddings ===")
        embeddings = generator.list_embeddings()
        if not embeddings:
            print("No embeddings found.")
        for emb in embeddings:
            print(f"  {emb.document_id}:")
            print(f"    Text dim: {emb.text_dimension}, Image dim: {emb.image_dimension}")

    elif args.document_id:
        print(f"\n=== Generating Embedding: {args.document_id} ===")
        doc_emb = generator.generate_document_embedding(args.document_id)
        print(f"Text embedding dimension: {doc_emb.text_dimension}")
        print(f"Image embedding dimension: {doc_emb.image_dimension}")

    elif args.text:
        print(f"\n=== Generating Text Embedding ===")
        embedding = generator.embed_text(args.text)
        print(f"Text: {args.text[:50]}...")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 5 values: {embedding[:5]}")

    elif args.all:
        print("\n=== Generating All Embeddings ===")
        results = generator.generate_all_embeddings()
        print(f"\nResults: {len(results['successful'])} success, "
              f"{len(results['failed'])} failed")

    else:
        print("Usage:")
        print("  python embedding_generator.py --document-id <id>")
        print("  python embedding_generator.py --text 'some text'")
        print("  python embedding_generator.py --all")
        print("  python embedding_generator.py --list")
