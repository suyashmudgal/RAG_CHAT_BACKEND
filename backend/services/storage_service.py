"""Supabase Storage client and file management service.

Handles user-isolated document storage in private Supabase Storage buckets.
Ensures deterministic user-scoped paths:
    users/{pg_user_id}/documents/{document_id}/{safe_filename}
"""

from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and unsafe characters.

    - Strips leading/trailing whitespace
    - Extracts basename only (removes any directory separators)
    - Replaces path traversal sequences (../, ..\\)
    - Replaces special characters with underscores, allowing only [a-zA-Z0-9_.-]
    - Removes leading dots to avoid hidden files
    """
    if not filename:
        return "document"

    # Extract basename only
    base = Path(filename).name
    # Strip any remaining slash/backslash sequences
    base = re.sub(r"[\/\\]+", "", base)
    # Remove path traversal patterns
    base = re.sub(r"\.\.+", ".", base)
    # Replace unsafe characters
    cleaned = re.sub(r"[^\w\.\-]", "_", base)
    # Avoid leading dots
    cleaned = cleaned.lstrip(".")
    return cleaned or "document"


def build_storage_path(user_id: int, document_id: str, filename: str) -> str:
    """Build a deterministic, user-scoped storage path for Supabase Storage.

    Format:
        users/{user_id}/documents/{document_id}/{safe_filename}
    """
    safe_name = sanitize_filename(filename)
    return f"users/{user_id}/documents/{document_id}/{safe_name}"


class SupabaseStorageService:
    """Service for interacting with Supabase Storage via REST API."""

    def __init__(
        self,
        supabase_url: str | None = None,
        service_role_key: str | None = None,
        bucket_name: str | None = None,
    ) -> None:
        self.supabase_url = (
            supabase_url or settings.resolved_supabase_url
        ).rstrip("/")
        self.service_role_key = (
            service_role_key or settings.supabase_service_role_key
        ).strip()
        self.bucket_name = (
            bucket_name or settings.supabase_storage_bucket or "documents"
        ).strip()

        # In-memory mock store used when service role key is not configured (e.g. offline testing)
        self._mock_store: dict[str, bytes] = {}

    @property
    def is_configured(self) -> bool:
        """Return True if live Supabase Storage credentials are configured."""
        return bool(self.supabase_url and self.service_role_key)

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    # ── Bucket Management ───────────────────────────────────────────────────

    async def ensure_bucket(self, bucket_name: str | None = None) -> bool:
        """Check if bucket exists; if not, create it as a private bucket."""
        target_bucket = bucket_name or self.bucket_name
        if not self.is_configured:
            logger.info("Storage running in mock mode: bucket '%s' ready", target_bucket)
            return True

        url = f"{self.supabase_url}/storage/v1/bucket/{target_bucket}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=self._headers())
                if res.status_code == 200:
                    logger.debug("Bucket '%s' exists in Supabase Storage", target_bucket)
                    return True

                # Bucket does not exist — create it privately
                create_url = f"{self.supabase_url}/storage/v1/bucket"
                payload = {
                    "id": target_bucket,
                    "name": target_bucket,
                    "public": False,  # MUST be private
                }
                create_res = await client.post(
                    create_url,
                    json=payload,
                    headers=self._headers("application/json"),
                )
                if create_res.status_code in (200, 201):
                    logger.info("Created private bucket '%s' in Supabase Storage", target_bucket)
                    return True
                elif create_res.status_code == 409 or "already exists" in create_res.text.lower():
                    return True
                else:
                    logger.warning(
                        "Failed to create bucket '%s': %s (status %d)",
                        target_bucket,
                        create_res.text,
                        create_res.status_code,
                    )
                    return False
        except Exception as exc:
            logger.error("Error ensuring Supabase bucket '%s': %s", target_bucket, exc)
            return False

    # ── Upload ──────────────────────────────────────────────────────────────

    async def upload_file(
        self,
        storage_path: str,
        content: bytes,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload raw bytes to Supabase Storage at the specified storage_path."""
        if not storage_path:
            raise ValueError("Storage path cannot be empty")

        if not content_type:
            content_type, _ = mimetypes.guess_type(storage_path)
            content_type = content_type or "application/octet-stream"

        if not self.is_configured:
            # Mock mode storage
            key = f"{self.bucket_name}/{storage_path}"
            self._mock_store[key] = content
            logger.info("Mock upload: stored %d bytes at '%s'", len(content), key)
            return {
                "bucket": self.bucket_name,
                "storage_path": storage_path,
                "size": len(content),
                "content_type": content_type,
            }

        url = f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{storage_path}"
        headers = self._headers(content_type)
        headers["x-upsert"] = "true"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, content=content, headers=headers)
                if res.status_code not in (200, 201):
                    logger.error(
                        "Failed to upload '%s' to Supabase Storage: %s (status %d)",
                        storage_path,
                        res.text,
                        res.status_code,
                    )
                    raise RuntimeError(f"Storage upload failed: {res.text}")

                data = res.json() if res.text.strip().startswith("{") else {}
                return {
                    "bucket": self.bucket_name,
                    "storage_path": storage_path,
                    "size": len(content),
                    "content_type": content_type,
                    "data": data,
                }
        except Exception as exc:
            logger.error("Supabase Storage upload error for '%s': %s", storage_path, exc)
            raise

    # ── Download ────────────────────────────────────────────────────────────

    async def download_file(self, storage_path: str) -> bytes:
        """Download raw bytes from Supabase Storage for the specified path."""
        if not storage_path:
            raise ValueError("Storage path cannot be empty")

        if not self.is_configured:
            key = f"{self.bucket_name}/{storage_path}"
            if key in self._mock_store:
                return self._mock_store[key]
            # Also check without bucket prefix
            if storage_path in self._mock_store:
                return self._mock_store[storage_path]
            raise FileNotFoundError(f"File '{storage_path}' not found in mock storage")

        # Use authenticated endpoint for private bucket
        url = f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket_name}/{storage_path}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(url, headers=self._headers())
                if res.status_code == 404:
                    raise FileNotFoundError(f"File '{storage_path}' not found in Supabase Storage")
                if res.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download '{storage_path}' from storage (status {res.status_code})"
                    )
                return res.content
        except Exception as exc:
            logger.error("Supabase Storage download error for '%s': %s", storage_path, exc)
            raise

    # ── Delete ──────────────────────────────────────────────────────────────

    async def delete_file(self, storage_path: str) -> bool:
        """Delete an object from Supabase Storage."""
        if not storage_path:
            return False

        if not self.is_configured:
            key = f"{self.bucket_name}/{storage_path}"
            self._mock_store.pop(key, None)
            self._mock_store.pop(storage_path, None)
            logger.info("Mock delete: removed '%s'", storage_path)
            return True

        url = f"{self.supabase_url}/storage/v1/object/{self.bucket_name}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Supabase Storage delete takes a JSON array of prefixes/paths
                res = await client.request(
                    "DELETE",
                    url,
                    json={"prefixes": [storage_path]},
                    headers=self._headers("application/json"),
                )
                if res.status_code in (200, 204):
                    logger.info("Deleted '%s' from Supabase Storage", storage_path)
                    return True
                else:
                    logger.warning(
                        "Delete '%s' returned status %d: %s",
                        storage_path,
                        res.status_code,
                        res.text,
                    )
                    return False
        except Exception as exc:
            logger.error("Supabase Storage delete error for '%s': %s", storage_path, exc)
            return False

    # ── Signed URL ──────────────────────────────────────────────────────────

    async def create_signed_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generate a short-lived signed URL for authenticated user download/view."""
        if not storage_path:
            raise ValueError("Storage path cannot be empty")

        if not self.is_configured:
            return f"mock://storage/{self.bucket_name}/{storage_path}?expires={expires_in}"

        url = f"{self.supabase_url}/storage/v1/object/sign/{self.bucket_name}/{storage_path}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    json={"expiresIn": expires_in},
                    headers=self._headers("application/json"),
                )
                if res.status_code != 200:
                    raise RuntimeError(f"Failed to generate signed URL: {res.text}")
                data = res.json()
                signed_path = data.get("signedURL") or data.get("signedUrl")
                if signed_path and signed_path.startswith("http"):
                    return signed_path
                return f"{self.supabase_url}/storage/v1{signed_path}"
        except Exception as exc:
            logger.error("Supabase Storage signed URL error for '%s': %s", storage_path, exc)
            raise

    # ── Object Exists Check ─────────────────────────────────────────────────

    async def file_exists(self, storage_path: str) -> bool:
        """Check if an object exists in storage."""
        if not self.is_configured:
            key = f"{self.bucket_name}/{storage_path}"
            return key in self._mock_store or storage_path in self._mock_store

        try:
            # Try to get metadata via authenticated GET or list
            url = f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket_name}/{storage_path}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.head(url, headers=self._headers())
                if res.status_code in (200, 204):
                    return True
                # Fallback to GET with range 0-0
                res = await client.get(
                    url,
                    headers={**self._headers(), "Range": "bytes=0-0"},
                )
                return res.status_code in (200, 206)
        except Exception:
            return False
