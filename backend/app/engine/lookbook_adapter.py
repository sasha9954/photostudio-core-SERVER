from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, List, Tuple

from app.core.config import settings
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url



def _ensure_assets_dir() -> None:
    ensure_static_dirs()


def _guess_ext(mime: str) -> str:
    m = (mime or "").lower()
    if m == "image/png":
        return ".png"
    if m in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if m == "image/webp":
        return ".webp"
    return ".png"


def save_b64_image_as_asset(mime: str, b64: str) -> str:
    """Save base64 image (no data: prefix) into backend static/assets and return absolute URL."""
    raw = base64.b64decode(b64)
    _ensure_assets_dir()
    h = hashlib.sha256(raw).hexdigest()[:16]
    ext = _guess_ext(mime)
    fn = f"{h}{ext}"
    path = os.path.join(ASSETS_DIR, fn)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    return asset_url(fn)


def build_legacy_scene(model_url: str, location_url: str) -> Dict[str, Any]:
    return {
        "model": {"source": "url", "imgUrl": model_url},
        "location": {"source": "url", "imgUrl": location_url},
    }


def cards_to_legacy_shots(cards: List[Dict[str, Any]], fmt: str) -> List[Dict[str, Any]]:
    """Convert lookbook session cards to legacy engine shots.

    Slots mapping (v1):
      1-4 -> ITEM
      5-7 -> DETAIL
      8   -> LOGO
    """
    shots: List[Dict[str, Any]] = []
    for c in cards or []:
        ref_url = c.get("refUrl")
        if not ref_url:
            continue

        slot = int(c.get("slot") or 0)
        if slot == 8 or c.get("type") == "logo":
            shot_type = "LOGO"
        elif slot in (5, 6, 7):
            shot_type = "DETAIL"
        else:
            shot_type = "ITEM"

        shots.append(
            {
                "id": f"slot{slot}" if slot else (c.get("id") or "shot"),
                "refImage": {"source": "url", "imgUrl": ref_url},
                "shotType": shot_type,
                "cameraAngle": c.get("camera") or "front",
                "poseStyle": c.get("pose") or "classic",
                "format": fmt or "1:1",
            }
        )
    return shots


def run_legacy_lookbook_photoshoot(mode: str, model_url: str, location_url: str, cards: List[Dict[str, Any]], fmt: str, debug: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Run legacy engine and return (ok, payload). payload on ok: {urls: [...], debug?: ...}
    payload on error: {code,message,hint,shotId?,debug?}
    """
    # Ensure env vars (legacy engine reads from os.getenv)
    if settings.GEMINI_API_KEY and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    if settings.GEMINI_IMAGE_MODEL and not os.getenv("GEMINI_IMAGE_MODEL"):
        os.environ["GEMINI_IMAGE_MODEL"] = settings.GEMINI_IMAGE_MODEL
    if settings.GEMINI_VISION_MODEL and not os.getenv("GEMINI_VISION_MODEL"):
        os.environ["GEMINI_VISION_MODEL"] = settings.GEMINI_VISION_MODEL

    # Lazy import so backend can start even if engine deps are missing
    from app.engine.legacy_engine.engine_init import load_engine_config
    from app.engine.legacy_engine.lookbook_engine import photoshoot

    cfg = load_engine_config()
    prompts_dir = os.path.join(os.path.dirname(__file__), "legacy_engine", "prompts")

    scene = build_legacy_scene(model_url, location_url)
    shots = cards_to_legacy_shots(cards, fmt)
    if not shots:
        return False, {
            "code": "LOOKBOOK_NO_SHOTS",
            "message": "Нет карточек для фотосессии",
            "hint": "Загрузи хотя бы 1 фото вещи (слот 1-4) или детали/логотип (слот 5-8).",
        }

    resp = photoshoot(cfg, prompts_dir, mode, scene, shots, debug=debug)
    if not resp.get("ok"):
        return False, {
            "code": resp.get("code") or "ENGINE_ERROR",
            "message": resp.get("message") or "Ошибка движка",
            "hint": resp.get("hint") or "Попробуй другой реф/ракурс.",
            "shotId": resp.get("shotId"),
            "debug": resp.get("debug"),
        }

    urls: List[str] = []
    for r in resp.get("results") or []:
        img = r.get("image") or ""
        # data:mime;base64,...
        if img.startswith("data:") and ";base64," in img:
            header, b64 = img.split(",", 1)
            mime = header.split(";", 1)[0].replace("data:", "") or "image/png"
            urls.append(save_b64_image_as_asset(mime, b64))

    if not urls:
        return False, {
            "code": "ENGINE_NO_IMAGE",
            "message": "Движок не вернул изображение",
            "hint": "Попробуй другой реф/ракурс или проверь GEMINI_IMAGE_MODEL.",
        }

    out: Dict[str, Any] = {"urls": urls}
    if debug:
        out["debug"] = {
            "engine": "legacy",
            "model": cfg.image_model,
            "vision": cfg.vision_model,
        }
    return True, out
