from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import re
import base64
import hashlib
import io
import zipfile
import mimetypes
import urllib.request
import urllib.parse
import tempfile
import subprocess
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url, resolve_asset_filename_with_image_fallback

router = APIRouter()

def _ensure_assets_dir():
    try:
        ensure_static_dirs()
    except Exception:
        pass

def _guess_ext_from_content_type(ct: str | None) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    if not ct:
        return ""
    # manual fixes for common audio types (mimetypes varies by OS)
    if ct in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if ct in ("audio/wav", "audio/x-wav"):
        return ".wav"
    if ct in ("audio/ogg",):
        return ".ogg"
    if ct in ("audio/mp4", "audio/m4a", "audio/x-m4a"):
        return ".m4a"
    ext = mimetypes.guess_extension(ct) or ""
    if ext == ".jpe":
        ext = ".jpg"
    return ext

def _probe_audio_duration_sec(raw: bytes, ext: str) -> float | None:
    """Return audio duration in seconds using ffprobe. Returns None if unavailable."""
    try:
        # write temp file
        suffix = ext or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                tmp_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            s = (r.stdout or "").strip()
            dur = float(s) if s else None
            if dur and dur > 0:
                return dur
            return None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        return None


@router.post("/assets/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Generic upload endpoint for small user-provided files (audio/images).

    Frontend uses it for storyboard AUDIO node.
    Returns absolute /static/assets URL.
    """
    _ensure_assets_dir()

    if file is None:
        raise HTTPException(status_code=400, detail="no_file")

    ct = (file.content_type or "").split(";")[0].strip().lower()
    try:
        raw = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="read_failed")

    if not raw:
        raise HTTPException(status_code=400, detail="empty")

    # soft limits to avoid abuse in local dev
    if len(raw) > 60 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="too_large")

    ext = _guess_ext_from_content_type(ct)
    if not ext:
        # fallback from original filename
        try:
            ext = os.path.splitext(file.filename or "")[1].lower()
        except Exception:
            ext = ""

    # allowlist
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg", ".m4a"}
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"unsupported_ext:{ext or 'none'}")

    audio_ext = {".mp3", ".wav", ".ogg", ".m4a"}
    if ct.startswith("audio/") or ext in audio_ext:
        ext = ".mp3"

    hid = _hash_bytes(raw)
    fn = f"{hid}{ext}"
    out_path = os.path.join(ASSETS_DIR, fn)

    # idempotent write
    if not os.path.exists(out_path):
        try:
            with open(out_path, "wb") as f:
                f.write(raw)
        except Exception:
            raise HTTPException(status_code=500, detail="write_failed")

    print("saved asset =", str(out_path), os.path.exists(out_path))

    
    duration_sec = None
    if ct.startswith("audio/"):
        try:
            duration_sec = _probe_audio_duration_sec(raw, ".mp3")
        except Exception:
            duration_sec = None

    url = asset_url(fn)
    return {
        "url": url,
        "mime": ct,
        "bytes": len(raw),
        "durationSec": duration_sec,
        "name": file.filename or fn,
    }

def _hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()[:16]

def _safe_filename_from_url(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        name = os.path.basename(p.path)
        return name
    except Exception:
        return ""


class FromDataUrlIn(BaseModel):
    dataUrl: str

class FromUrlIn(BaseModel):
    url: str

class ZipIn(BaseModel):
    urls: list[str]

def _ensure_dir():
    ensure_static_dirs()

def _guess_ext(mime: str) -> str:
    m = (mime or "").lower()
    if m.startswith("audio/"):
        return ".mp3"
    if m == "image/png":
        return ".png"
    if m in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if m == "image/webp":
        return ".webp"
    return ".png"

@router.post("/assets/fromDataUrl")
def from_data_url(req: Request, body: FromDataUrlIn):
    s = body.dataUrl or ""
    if not s.startswith("data:"):
        raise HTTPException(status_code=400, detail="dataUrl must start with data:")
    m = re.match(r"^data:([^;]+);base64,(.+)$", s, re.IGNORECASE | re.DOTALL)
    if not m:
        raise HTTPException(status_code=400, detail="Invalid dataUrl")
    mime = m.group(1).strip()
    b64 = m.group(2).strip()
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    _ensure_dir()
    h = hashlib.sha256(raw).hexdigest()[:16]
    ext = _guess_ext(mime)
    fn = f"{h}{ext}"
    path = os.path.join(ASSETS_DIR, fn)

    # write once (idempotent)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)

    # return absolute URL (frontend needs correct host)
    url = asset_url(fn)
    return {"url": url, "mime": mime, "bytes": len(raw)}


@router.post("/assets/fromUrl")
def from_url(payload: FromUrlIn, request: Request):
    url = (payload.url or "").strip()
    if not url or not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=400, detail="bad_url")

    _ensure_assets_dir()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhotoStudio/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
    except Exception as e:
        raise HTTPException(status_code=400, detail="fetch_failed")

    if not data or len(data) < 32:
        raise HTTPException(status_code=400, detail="empty")

    ext = _guess_ext_from_content_type(ct)
    if not ext:
        # fallback by path
        name = _safe_filename_from_url(url).lower()
        ext = os.path.splitext(name)[1]
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        # default to png if unknown
        ext = ".png"
    if ext == ".jpeg":
        ext = ".jpg"

    hid = _hash_bytes(data)
    fn = f"{hid}{ext}"
    out_path = os.path.join(ASSETS_DIR, fn)
    try:
        with open(out_path, "wb") as f:
            f.write(data)
    except Exception:
        raise HTTPException(status_code=500, detail="write_failed")

    return {"url": asset_url(fn)}


@router.post("/zip")
def zip_assets(payload: ZipIn, request: Request):
    _ensure_assets_dir()
    urls = payload.urls or []
    urls = [u for u in urls if isinstance(u, str) and u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="no_urls")

    # allow only our /static/assets files
    files = []
    for u in urls[:50]:
        u = u.strip()
        try:
            p = urllib.parse.urlparse(u)
            path = p.path or ""
        except Exception:
            path = u
        if "/static/assets/" not in path:
            continue
        fn = path.split("/static/assets/")[-1]
        fn = os.path.basename(fn)
        if not re.match(r"^[0-9a-f]{8,32}\.(png|jpg|jpeg|webp)$", fn, flags=re.I):
            continue
        resolved_fn = resolve_asset_filename_with_image_fallback(fn) or fn
        fp = os.path.join(ASSETS_DIR, resolved_fn)
        if os.path.exists(fp):
            files.append((resolved_fn, fp))

    if not files:
        raise HTTPException(status_code=400, detail="no_local_assets")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i, (fn, fp) in enumerate(files, start=1):
            arc = f"{i:02d}_{fn}"
            z.write(fp, arcname=arc)
    buf.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="prints.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
