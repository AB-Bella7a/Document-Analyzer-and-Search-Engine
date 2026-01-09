"""
Storage Manager Module

This module provides a unified interface for file storage operations,
supporting both local filesystem and AWS S3 cloud storage.

Key Features:
- Abstract storage interface for consistency
- Local storage for development and testing
- S3 storage for production deployment
- Automatic fallback to local when S3 is not configured

Architecture:
- StorageBackend: Abstract base class defining the interface
- LocalStorage: Implementation for local filesystem
- S3Storage: Implementation for AWS S3
- StorageManager: Factory that provides the appropriate backend

Example:
    >>> from storage import StorageManager
    >>>
    >>> # Get storage instance (automatically chooses backend)
    >>> storage = StorageManager()
    >>>
    >>> # Upload a file
    >>> storage.upload_file("/local/path/doc.pdf", "documents/doc.pdf")
    >>>
    >>> # Download a file
    >>> storage.download_file("documents/doc.pdf", "/local/path/doc.pdf")
    >>>
    >>> # List files
    >>> files = storage.list_files("documents/")
"""

import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO, Union
import json
import sys

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    This defines the interface that all storage implementations
    must follow, ensuring consistent behavior across different
    storage systems.
    """

    @abstractmethod
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload a file to storage.

        Args:
            local_path: Path to the local file
            remote_path: Destination path in storage
            metadata: Optional metadata to store with the file

        Returns:
            URI/path to the uploaded file
        """
        pass

    @abstractmethod
    def upload_bytes(
        self,
        data: bytes,
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload bytes directly to storage.

        Args:
            data: Raw bytes to upload
            remote_path: Destination path in storage
            metadata: Optional metadata to store with the file

        Returns:
            URI/path to the uploaded file
        """
        pass

    @abstractmethod
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> Path:
        """
        Download a file from storage.

        Args:
            remote_path: Path in storage
            local_path: Destination local path

        Returns:
            Path to the downloaded file
        """
        pass

    @abstractmethod
    def download_bytes(self, remote_path: str) -> bytes:
        """
        Download file contents as bytes.

        Args:
            remote_path: Path in storage

        Returns:
            File contents as bytes
        """
        pass

    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """
        Delete a file from storage.

        Args:
            remote_path: Path to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            remote_path: Path to check

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List files in storage.

        Args:
            prefix: Path prefix to filter by
            recursive: Whether to list recursively

        Returns:
            List of file info dictionaries
        """
        pass

    @abstractmethod
    def get_file_info(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file.

        Args:
            remote_path: Path to the file

        Returns:
            File info dictionary or None if not found
        """
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage implementation.

    This backend stores files on the local filesystem,
    useful for development and testing without cloud dependencies.

    Files are organized under a base directory with the same
    structure as they would have in S3.

    Example:
        >>> storage = LocalStorage("/data/storage")
        >>> storage.upload_file("doc.pdf", "documents/doc.pdf")
        # File stored at: /data/storage/documents/doc.pdf
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize local storage.

        Args:
            base_dir: Base directory for storage (default: DATA_DIR/storage)
        """
        self.base_dir = Path(base_dir) if base_dir else settings.DATA_DIR / "storage"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Storage] Local storage initialized at: {self.base_dir}")

    def _get_full_path(self, remote_path: str) -> Path:
        """Get the full local path for a remote path."""
        # Normalize path separators and remove leading slashes
        normalized = remote_path.replace("\\", "/").lstrip("/")
        return self.base_dir / normalized

    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a file to local storage."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        dest_path = self._get_full_path(remote_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(local_path, dest_path)

        # Save metadata if provided
        if metadata:
            meta_path = dest_path.with_suffix(dest_path.suffix + ".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"[Storage] Uploaded: {remote_path}")
        return f"file://{dest_path}"

    def upload_bytes(
        self,
        data: bytes,
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload bytes to local storage."""
        dest_path = self._get_full_path(remote_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bytes
        with open(dest_path, "wb") as f:
            f.write(data)

        # Save metadata if provided
        if metadata:
            meta_path = dest_path.with_suffix(dest_path.suffix + ".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"[Storage] Uploaded bytes: {remote_path}")
        return f"file://{dest_path}"

    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> Path:
        """Download a file from local storage."""
        source_path = self._get_full_path(remote_path)
        local_path = Path(local_path)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found in storage: {remote_path}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, local_path)

        print(f"[Storage] Downloaded: {remote_path} -> {local_path}")
        return local_path

    def download_bytes(self, remote_path: str) -> bytes:
        """Download file contents as bytes."""
        source_path = self._get_full_path(remote_path)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found in storage: {remote_path}")

        with open(source_path, "rb") as f:
            return f.read()

    def delete_file(self, remote_path: str) -> bool:
        """Delete a file from local storage."""
        file_path = self._get_full_path(remote_path)

        if not file_path.exists():
            return False

        file_path.unlink()

        # Also delete metadata file if it exists
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        if meta_path.exists():
            meta_path.unlink()

        print(f"[Storage] Deleted: {remote_path}")
        return True

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists in local storage."""
        return self._get_full_path(remote_path).exists()

    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """List files in local storage."""
        search_dir = self._get_full_path(prefix)
        files = []

        if not search_dir.exists():
            return files

        # Use glob pattern based on recursive setting
        pattern = "**/*" if recursive else "*"

        for file_path in search_dir.glob(pattern):
            # Skip directories and metadata files
            if file_path.is_dir() or file_path.suffix == ".json":
                continue

            rel_path = file_path.relative_to(self.base_dir)
            stat = file_path.stat()

            files.append({
                "path": str(rel_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "uri": f"file://{file_path}"
            })

        return files

    def get_file_info(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a file."""
        file_path = self._get_full_path(remote_path)

        if not file_path.exists():
            return None

        stat = file_path.stat()
        info = {
            "path": remote_path,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "uri": f"file://{file_path}"
        }

        # Load metadata if it exists
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                info["metadata"] = json.load(f)

        return info


class S3Storage(StorageBackend):
    """
    AWS S3 storage implementation.

    This backend stores files in an S3 bucket, suitable for
    production deployments requiring scalable cloud storage.

    Requires AWS credentials to be configured either via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - .env file
    - IAM roles (when running on AWS)

    Example:
        >>> storage = S3Storage("my-bucket")
        >>> storage.upload_file("doc.pdf", "documents/doc.pdf")
        # File stored at: s3://my-bucket/documents/doc.pdf
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None
    ):
        """
        Initialize S3 storage.

        Args:
            bucket_name: S3 bucket name (default from config)
            region: AWS region (default from config)
        """
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.region = region or settings.AWS_REGION

        if not self.bucket_name:
            raise ValueError("S3 bucket name is required")

        # Initialize boto3 client
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            print(f"[Storage] S3 storage initialized: s3://{self.bucket_name}")
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

    def _normalize_key(self, path: str) -> str:
        """Normalize a path to a valid S3 key."""
        return path.replace("\\", "/").lstrip("/")

    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a file to S3."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3_key = self._normalize_key(remote_path)

        # Upload with optional metadata
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata

        self.s3_client.upload_file(
            str(local_path),
            self.bucket_name,
            s3_key,
            ExtraArgs=extra_args if extra_args else None
        )

        uri = f"s3://{self.bucket_name}/{s3_key}"
        print(f"[Storage] Uploaded to S3: {uri}")
        return uri

    def upload_bytes(
        self,
        data: bytes,
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload bytes to S3."""
        s3_key = self._normalize_key(remote_path)

        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=data,
            **extra_args
        )

        uri = f"s3://{self.bucket_name}/{s3_key}"
        print(f"[Storage] Uploaded bytes to S3: {uri}")
        return uri

    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> Path:
        """Download a file from S3."""
        s3_key = self._normalize_key(remote_path)
        local_path = Path(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        self.s3_client.download_file(
            self.bucket_name,
            s3_key,
            str(local_path)
        )

        print(f"[Storage] Downloaded from S3: {s3_key} -> {local_path}")
        return local_path

    def download_bytes(self, remote_path: str) -> bytes:
        """Download file contents as bytes from S3."""
        s3_key = self._normalize_key(remote_path)

        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=s3_key
        )

        return response["Body"].read()

    def delete_file(self, remote_path: str) -> bool:
        """Delete a file from S3."""
        s3_key = self._normalize_key(remote_path)

        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            print(f"[Storage] Deleted from S3: {s3_key}")
            return True
        except Exception as e:
            print(f"[Storage] Delete failed: {e}")
            return False

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists in S3."""
        s3_key = self._normalize_key(remote_path)

        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """List files in S3."""
        s3_prefix = self._normalize_key(prefix)
        files = []

        paginator = self.s3_client.get_paginator("list_objects_v2")

        params = {
            "Bucket": self.bucket_name,
            "Prefix": s3_prefix
        }

        if not recursive:
            params["Delimiter"] = "/"

        for page in paginator.paginate(**params):
            for obj in page.get("Contents", []):
                files.append({
                    "path": obj["Key"],
                    "size": obj["Size"],
                    "modified": obj["LastModified"].isoformat(),
                    "uri": f"s3://{self.bucket_name}/{obj['Key']}"
                })

        return files

    def get_file_info(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a file in S3."""
        s3_key = self._normalize_key(remote_path)

        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            return {
                "path": s3_key,
                "size": response["ContentLength"],
                "modified": response["LastModified"].isoformat(),
                "uri": f"s3://{self.bucket_name}/{s3_key}",
                "metadata": response.get("Metadata", {})
            }
        except self.s3_client.exceptions.ClientError:
            return None

    def generate_presigned_url(
        self,
        remote_path: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            remote_path: Path to the file
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL string
        """
        s3_key = self._normalize_key(remote_path)

        url = self.s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": s3_key
            },
            ExpiresIn=expiration
        )

        return url


class StorageManager:
    """
    Storage Manager Factory

    This class provides a unified interface to storage operations,
    automatically selecting the appropriate backend based on
    configuration.

    When S3 is configured, it uses S3Storage.
    Otherwise, it falls back to LocalStorage.

    The manager can also sync between local and cloud storage.

    Example:
        >>> storage = StorageManager()
        >>>
        >>> # Upload works with either backend
        >>> storage.upload_file("doc.pdf", "documents/doc.pdf")
        >>>
        >>> # Check which backend is active
        >>> print(storage.backend_type)
        'local'  # or 's3'
    """

    def __init__(self, force_local: bool = False):
        """
        Initialize storage manager.

        Args:
            force_local: If True, use local storage even if S3 is configured
        """
        self.force_local = force_local
        self._backend: Optional[StorageBackend] = None
        self._local_storage: Optional[LocalStorage] = None

    @property
    def backend(self) -> StorageBackend:
        """Get the storage backend (lazy initialization)."""
        if self._backend is None:
            if not self.force_local and settings.is_s3_configured():
                try:
                    self._backend = S3Storage()
                except Exception as e:
                    print(f"[Storage] S3 initialization failed: {e}")
                    print("[Storage] Falling back to local storage")
                    self._backend = LocalStorage()
            else:
                self._backend = LocalStorage()
        return self._backend

    @property
    def local_storage(self) -> LocalStorage:
        """Get local storage instance (for syncing)."""
        if self._local_storage is None:
            self._local_storage = LocalStorage()
        return self._local_storage

    @property
    def backend_type(self) -> str:
        """Get the type of backend in use."""
        if isinstance(self.backend, S3Storage):
            return "s3"
        return "local"

    # Delegate all operations to the backend
    def upload_file(self, *args, **kwargs) -> str:
        """Upload a file to storage."""
        return self.backend.upload_file(*args, **kwargs)

    def upload_bytes(self, *args, **kwargs) -> str:
        """Upload bytes to storage."""
        return self.backend.upload_bytes(*args, **kwargs)

    def download_file(self, *args, **kwargs) -> Path:
        """Download a file from storage."""
        return self.backend.download_file(*args, **kwargs)

    def download_bytes(self, *args, **kwargs) -> bytes:
        """Download file contents as bytes."""
        return self.backend.download_bytes(*args, **kwargs)

    def delete_file(self, *args, **kwargs) -> bool:
        """Delete a file from storage."""
        return self.backend.delete_file(*args, **kwargs)

    def file_exists(self, *args, **kwargs) -> bool:
        """Check if a file exists."""
        return self.backend.file_exists(*args, **kwargs)

    def list_files(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """List files in storage."""
        return self.backend.list_files(*args, **kwargs)

    def get_file_info(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get file information."""
        return self.backend.get_file_info(*args, **kwargs)

    def sync_to_cloud(self, prefix: str = "") -> Dict[str, Any]:
        """
        Sync local files to cloud storage.

        Args:
            prefix: Path prefix to sync

        Returns:
            Sync results dictionary
        """
        if not isinstance(self.backend, S3Storage):
            return {"status": "skipped", "reason": "S3 not configured"}

        results = {"uploaded": [], "failed": []}

        # Get local files
        local_files = self.local_storage.list_files(prefix)

        for file_info in local_files:
            try:
                local_path = self.local_storage.base_dir / file_info["path"]
                self.backend.upload_file(local_path, file_info["path"])
                results["uploaded"].append(file_info["path"])
            except Exception as e:
                results["failed"].append({
                    "path": file_info["path"],
                    "error": str(e)
                })

        print(f"[Storage] Sync complete: {len(results['uploaded'])} uploaded, "
              f"{len(results['failed'])} failed")

        return results

    def sync_from_cloud(self, prefix: str = "") -> Dict[str, Any]:
        """
        Sync files from cloud to local storage.

        Args:
            prefix: Path prefix to sync

        Returns:
            Sync results dictionary
        """
        if not isinstance(self.backend, S3Storage):
            return {"status": "skipped", "reason": "S3 not configured"}

        results = {"downloaded": [], "failed": []}

        # Get cloud files
        cloud_files = self.backend.list_files(prefix)

        for file_info in cloud_files:
            try:
                local_path = self.local_storage.base_dir / file_info["path"]
                self.backend.download_file(file_info["path"], local_path)
                results["downloaded"].append(file_info["path"])
            except Exception as e:
                results["failed"].append({
                    "path": file_info["path"],
                    "error": str(e)
                })

        print(f"[Storage] Sync complete: {len(results['downloaded'])} downloaded, "
              f"{len(results['failed'])} failed")

        return results


# =============================================================================
# Standalone execution for testing
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Storage Manager")
    parser.add_argument("--list", "-l", nargs="?", const="", help="List files (optional prefix)")
    parser.add_argument("--upload", "-u", nargs=2, metavar=("LOCAL", "REMOTE"), help="Upload file")
    parser.add_argument("--download", "-d", nargs=2, metavar=("REMOTE", "LOCAL"), help="Download file")
    parser.add_argument("--delete", help="Delete a file")
    parser.add_argument("--info", help="Get file info")
    parser.add_argument("--sync-to-cloud", action="store_true", help="Sync local to S3")
    parser.add_argument("--sync-from-cloud", action="store_true", help="Sync S3 to local")
    args = parser.parse_args()

    storage = StorageManager()
    print(f"Storage backend: {storage.backend_type}")

    if args.list is not None:
        print(f"\n=== Files (prefix: '{args.list}') ===")
        files = storage.list_files(args.list)
        for f in files:
            print(f"  {f['path']} ({f['size']} bytes)")

    elif args.upload:
        local, remote = args.upload
        uri = storage.upload_file(local, remote)
        print(f"Uploaded to: {uri}")

    elif args.download:
        remote, local = args.download
        path = storage.download_file(remote, local)
        print(f"Downloaded to: {path}")

    elif args.delete:
        if storage.delete_file(args.delete):
            print(f"Deleted: {args.delete}")
        else:
            print(f"Not found: {args.delete}")

    elif args.info:
        info = storage.get_file_info(args.info)
        if info:
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(f"Not found: {args.info}")

    elif args.sync_to_cloud:
        results = storage.sync_to_cloud()
        print(f"Results: {results}")

    elif args.sync_from_cloud:
        results = storage.sync_from_cloud()
        print(f"Results: {results}")

    else:
        print("\nUsage:")
        print("  python storage_manager.py --list [prefix]")
        print("  python storage_manager.py --upload <local> <remote>")
        print("  python storage_manager.py --download <remote> <local>")
        print("  python storage_manager.py --delete <path>")
        print("  python storage_manager.py --info <path>")
        print("  python storage_manager.py --sync-to-cloud")
        print("  python storage_manager.py --sync-from-cloud")
