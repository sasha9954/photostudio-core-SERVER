from __future__ import annotations

from pathlib import Path
import os

from app.core.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BACKEND_DIR / "static"
ASSETS_DIR = STATIC_DIR / "assets"
VIDEOS_DIR = STATIC_DIR / "videos"


def ensure_static_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def asset_url(filename: str) -> str:
    base = (settings.PUBLIC_BASE_URL or "http://127.0.0.1:8000").rstrip("/")
    return f"{base}/static/assets/{filename}"


def resolve_asset_filename_with_image_fallback(filename: str) -> str | None:
    """Return existing filename for old data with extension drift.

    First tries exact filename, then basename with common image extensions.
    """
    if not filename:
        return None

    exact = ASSETS_DIR / filename
    if exact.is_file():
        return filename

    base, ext = os.path.splitext(filename)
    if not base:
        return None

    if ext.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        candidates = [".jpg", ".jpeg", ".png", ".webp"]
    else:
        candidates = [ext, ".jpg", ".jpeg", ".png", ".webp"]

    for candidate_ext in candidates:
        candidate_name = f"{base}{candidate_ext}"
        if (ASSETS_DIR / candidate_name).is_file():
            return candidate_name

    return None
