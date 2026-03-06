import os
import io
import re
import math
import base64
import hashlib
import urllib.request
import yaml
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageStat, ImageEnhance

from app.core.config import settings
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url
import uuid

def _sha256_bytes(b: bytes) -> str:
    try:
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return 'NA'


# --- Prints: external print type profiles (YAML) ---
_PRINT_PROFILES_CACHE = None


# --- Garment rules (optional) ---
_GARMENT_RULES_CACHE = None

def _resolve_garment_rules_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / "prints" / "garment_rules.yaml"

def _load_garment_rules() -> dict:
    """Load garment_rules.yaml (safe defaults if missing)."""
    global _GARMENT_RULES_CACHE
    if _GARMENT_RULES_CACHE is not None:
        return _GARMENT_RULES_CACHE
    path = _resolve_garment_rules_path()
    if not path.exists():
        _GARMENT_RULES_CACHE = {"defaults": {"enabled": False}}
        return _GARMENT_RULES_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _GARMENT_RULES_CACHE = data
    except Exception:
        _GARMENT_RULES_CACHE = {"defaults": {"enabled": False}}
    return _GARMENT_RULES_CACHE

def _apply_center_seam_gap(mask_l: Image.Image, overlay_rgba: Image.Image, *, gap_px: int, bbox, img_w: int, img_h: int):
    """Cut a tiny vertical stripe at the image center inside bbox.
    This prevents prints crossing a shirt placket / zipper line.
    """
    if gap_px <= 0:
        return mask_l, overlay_rgba
    try:
        x0, y0, x1, y1 = bbox
        x0 = int(max(0, min(img_w - 1, x0)))
        x1 = int(max(0, min(img_w, x1)))
        y0 = int(max(0, min(img_h - 1, y0)))
        y1 = int(max(0, min(img_h, y1)))
        if x1 <= x0 or y1 <= y0:
            return mask_l, overlay_rgba

        cx = img_w // 2
        # stripe width = gap_px total
        half = max(0, gap_px // 2)
        sx0 = max(x0, cx - half)
        sx1 = min(x1, cx + (gap_px - half))
        if sx1 <= sx0:
            return mask_l, overlay_rgba

        cut_h = max(1, y1 - y0)
        cut = Image.new("L", (sx1 - sx0, cut_h), 0)
        mask_l.paste(cut, (sx0, y0))

        if overlay_rgba.mode != "RGBA":
            overlay_rgba = overlay_rgba.convert("RGBA")
        r, g, b, a = overlay_rgba.split()
        a.paste(Image.new("L", (sx1 - sx0, cut_h), 0), (sx0, y0))
        overlay_rgba = Image.merge("RGBA", (r, g, b, a))
        return mask_l, overlay_rgba
    except Exception:
        return mask_l, overlay_rgba

def _resolve_print_types_path() -> Path:
    """Try a few reasonable locations for print_modes.yaml (preferred) or legacy print_types.yaml."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "prints" / "print_modes.yaml",   # preferred
        here / "prints" / "print_types.yaml",   # legacy
        here / "print_modes.yaml",              # fallback (same dir)
        here / "print_types.yaml",              # legacy fallback (same dir)
        here.parent / "engine" / "prints" / "print_modes.yaml",  # extra fallback
        here.parent / "engine" / "prints" / "print_types.yaml",  # legacy extra fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    # default to first candidate (for error message)
    return candidates[0]

def _load_print_profiles() -> dict:
    global _PRINT_PROFILES_CACHE
    if _PRINT_PROFILES_CACHE is not None:
        return _PRINT_PROFILES_CACHE
    path = _resolve_print_types_path()
    if not path.exists():
        raise RuntimeError(f"print_modes.yaml/print_types.yaml not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "DEFAULT" not in data:
        raise RuntimeError("print_modes.yaml (or print_types.yaml) must contain DEFAULT profile")
    _PRINT_PROFILES_CACHE = data
    print(f"[prints] loaded print_types.yaml: {path}")
    return data

def _get_print_profile(mode: str) -> dict:
    profiles = _load_print_profiles()
    if not mode:
        return profiles["DEFAULT"]
    key = str(mode).strip().upper()
    return profiles.get(key, profiles["DEFAULT"])





# --- PRINT PROMPT TEMPLATES (TXT) ---
_PRINT_PROMPT_CACHE = {}

def _resolve_print_prompt_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    # Prompts live next to Lookbook prompts: backend/app/engine/prompts/prints/*.txt
    base = here / 'prompts' / 'prints'
    if name and str(name).lower().endswith('.txt'):
        fname = str(name)
    else:
        fname = f"{name}.txt" if name else 'default.txt'
    return base / fname

def _load_print_prompt(name: str) -> str:
    key = (name or 'default').strip().lower()
    # In dev we want prompt edits to apply without restarting.
    # Cache stores: {key: {"txt": str, "mtime": float, "path": str}}
    nocache = os.getenv('PRINTS_PROMPTS_NOCACHE', '').strip() in ('1', 'true', 'True', 'yes', 'YES')
    path = _resolve_print_prompt_path(name or 'default')
    if not path.exists():
        path = _resolve_print_prompt_path('default.txt')
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0

    if (not nocache) and key in _PRINT_PROMPT_CACHE:
        cached = _PRINT_PROMPT_CACHE.get(key) or {}
        if cached.get('path') == str(path) and cached.get('mtime') == mtime and cached.get('txt'):
            return cached['txt']
    txt = path.read_text(encoding='utf-8')
    if not nocache:
        _PRINT_PROMPT_CACHE[key] = {"txt": txt, "mtime": mtime, "path": str(path)}
    return txt

def _render_print_prompt(template_name: str, frame_lock_rules: str, mode_lock_rules: str, profile_text: str) -> str:
    tpl = _load_print_prompt(template_name or 'default')
    base = (
        tpl.replace('{{FRAME_LOCK_RULES}}', frame_lock_rules or '')
           .replace('{{MODE_LOCK_RULES}}', mode_lock_rules or '')
           .replace('{{PROFILE_TEXT}}', profile_text or '')
    )
    # V2.5: Surface-aware integration.
    # The engine supplies an additional geometry hint image (edges/relief) derived from the base.
    # This block forces the model to treat the print as ink/film bonded to fabric, following folds/seams.

    surface_rules = """

SURFACE INTEGRATION RULES:
The garment surface geometry provided in the reference images represents folds, seams and volume.
The print MUST follow:
- fabric curvature
- lighting gradients
- seam deformation
- puff volume transitions

The design is ink/film bonded to fabric (DTF/print), NOT a flat sticker overlay.
Do not redraw the garment or change stitching/baffles; only integrate the print.

SURFACE PRIORITY (ABSOLUTE):
Inside the print area you MUST deform the artwork to match the garment surface.
If there is any conflict between preserving the artwork and matching fabric geometry,
MATCHING FABRIC GEOMETRY WINS.
The print must bend, stretch, compress, and break across puff seams and folds.
Do NOT keep the design flat.
Lighting adaptation alone is NOT sufficient — GEOMETRY deformation is REQUIRED.

SEAM / ZIPPER / PLACKET RULES:
- If a zipper or button placket crosses the print area, the print MUST be naturally split by it (two panels).
- The print must show micro discontinuity across seams/stitches and baffle transitions (panelization).
- Do NOT paint over zipper teeth, stitching lines, or placket edges; let them interrupt the print.


EDGE PRIORITY RULES (HARD BOUNDARIES):
The edge reference image marks HARD garment boundaries and physical surface breaks (placket edges, zipper borders, pocket edges, panel borders).
- The artwork MUST NOT continue smoothly across these hard edges.
- If the print area crosses a hard edge, the design MUST split, terminate, or show a sharp discontinuity at that boundary.
- Prefer breaking the artwork over "saving" its original shape: EDGE BOUNDARIES WIN.
"""
    return base + surface_rules
# --- /PRINT PROMPT TEMPLATES ---

def _ensure_dir():
    ensure_static_dirs()


def _is_local_static_url(url: str) -> Optional[str]:
    """
    If url is like {PUBLIC_BASE_URL}/static/assets/<fn>, return absolute file path.
    """
    if not url:
        return None
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    m = re.match(rf"^{re.escape(base)}/static/assets/([A-Za-z0-9_\-]+\.(png|jpg|jpeg|webp))$", url, re.IGNORECASE)
    if not m:
        return None
    fn = m.group(1)
    return os.path.join(ASSETS_DIR, fn)


def _load_image_from_url(url: str) -> Image.Image:
    """
    Loads image from local static (fast path) or via HTTP(S).
    """
    # Frontend history/state can contain relative URLs ("/static/..." or "static/...").
    # Normalize them to an absolute backend URL.
    if url.startswith("/"):
        url = settings.PUBLIC_BASE_URL.rstrip("/") + url
    elif url.startswith("static/"):
        url = settings.PUBLIC_BASE_URL.rstrip("/") + "/" + url

    local_path = _is_local_static_url(url)
    if local_path and os.path.exists(local_path):
        img = Image.open(local_path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA")

    # http(s)
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
    except Exception as e:
        raise ValueError(f"Не удалось загрузить изображение: {url}. {e}")

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except Exception:
        raise ValueError("Файл не является изображением")
    return img.convert("RGBA")


def _ensure_min_long_side(img: Image.Image, min_long: int = 2048) -> Tuple[Image.Image, float, float]:
    """Ensure image long side >= min_long. Returns (img2, sx, sy) scale factors applied to width/height."""
    w, h = img.size
    long_side = max(w, h)
    if long_side >= min_long or long_side <= 0:
        return img, 1.0, 1.0
    scale = float(min_long) / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img2 = img.resize((nw, nh), Image.Resampling.LANCZOS)
    sx = nw / float(w)
    sy = nh / float(h)
    return img2, sx, sy

def _quad_to_pixels(quad, W: int, H: int) -> List[Tuple[float, float]]:
    """Accepts either:
    - 4 points [[x,y],...]*4 (normalized 0..1 or pixel)
    - 8 numbers [x1,y1,x2,y2,x3,y3,x4,y4] (normalized or pixel)
    - bbox [x,y,w,h] or {x,y,w,h} (normalized or pixel)
    and returns 4 points in pixel space.
    """
    if quad is None:
        raise ValueError("Missing or invalid quad (expected 8 numbers)")

    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    # dict bbox
    if isinstance(quad, dict):
        if all(k in quad for k in ("x", "y", "w", "h")) and all(is_num(quad[k]) for k in ("x","y","w","h")):
            x, y, w, h = float(quad["x"]), float(quad["y"]), float(quad["w"]), float(quad["h"])
            # normalize?
            if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                x, y, w, h = x*W, y*H, w*W, h*H
            return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        raise ValueError("Missing or invalid quad (expected 8 numbers)")

    # list/tuple
    if isinstance(quad, (list, tuple)):
        # bbox as 4 numbers
        if len(quad) == 4 and all(is_num(v) for v in quad):
            x, y, w, h = map(float, quad)
            if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                x, y, w, h = x*W, y*H, w*W, h*H
            return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

        # 8 numbers
        if len(quad) == 8 and all(is_num(v) for v in quad):
            pts = [(float(quad[0]), float(quad[1])),
                   (float(quad[2]), float(quad[3])),
                   (float(quad[4]), float(quad[5])),
                   (float(quad[6]), float(quad[7]))]
            if max(max(abs(x), abs(y)) for x, y in pts) <= 1.5:
                pts = [(x*W, y*H) for x, y in pts]
            return pts

        # 4 points
        if len(quad) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(is_num(v) for v in p) for p in quad):
            pts = [(float(p[0]), float(p[1])) for p in quad]
            if max(max(abs(x), abs(y)) for x, y in pts) <= 1.5:
                pts = [(x*W, y*H) for x, y in pts]
            return pts

    raise ValueError("Missing or invalid quad (expected 8 numbers)")

def _scale_quad(quad: List[Tuple[float, float]], sx: float, sy: float) -> List[Tuple[float, float]]:
    if sx == 1.0 and sy == 1.0:
        return quad
    return [(float(x) * sx, float(y) * sy) for (x, y) in quad]



def _flip_quad_x(quad: List[Tuple[float, float]], W: int) -> List[Tuple[float, float]]:
    """Mirror quad horizontally in IMAGE space (x -> W - x).
    Keeps point winding stable (so perspective coeffs don't invert unexpectedly)."""
    if not quad or len(quad) != 4:
        return quad
    q2 = [(float(W) - float(x), float(y)) for (x, y) in quad]
    try:
        # preserve winding (clockwise/counter-clockwise)
        def _area(pts):
            a = 0.0
            for i in range(len(pts)):
                x1, y1 = float(pts[i][0]), float(pts[i][1])
                x2, y2 = float(pts[(i + 1) % len(pts)][0]), float(pts[(i + 1) % len(pts)][1])
                a += x1 * y2 - x2 * y1
            return a
        if _area(q2) * _area(quad) < 0:
            q2 = [q2[0], q2[3], q2[2], q2[1]]
    except Exception:
        pass
    return q2

def _auto_flip_x_enabled() -> bool:
    """Whether to auto-flip placement horizontally when frontend does NOT specify coord_space/flip_x.

    IMPORTANT:
    - Default is OFF to avoid unexpected geometry changes for normal DTF/logo placement.
    - Enable explicitly by setting PRINTS_AUTO_FLIP_X=1 (or 'true/on/yes').
    """
    v = (os.getenv('PRINTS_AUTO_FLIP_X', '') or '').strip().lower()
    if v in ('1','true','yes','on'):
        return True
    return False

def _force_flip_x_enabled() -> bool:
    """Force-flip X for quad coordinates.

    Why: in dev we sometimes run with a mirrored preview/canvas coordinate system.
    If PRINTS_FORCE_FLIP_X is not set, we default to True for local/dev to keep parity stable.
    In production (PS_ENV not dev), default is False unless explicitly enabled.
    """
    raw = os.getenv('PRINTS_FORCE_FLIP_X', None)
    if raw is None or str(raw).strip() == '':
        ps_env = (os.getenv('PS_ENV') or os.getenv('ENV') or '').strip().lower()
        if ps_env in ('dev', 'development', 'local'):
            return True
        return False
    v = str(raw).strip().lower()
    return v in ('1', 'true', 'yes', 'on')




def get_image_size(url: str) -> Tuple[int, int]:
    """Return (w,h) for a given image URL (local /static/assets fast-path or http(s))."""
    img = _load_image_from_url(url)
    return img.size


def _normalize_mode(mode: str) -> str:
    m = (mode or "").lower().strip()
    # map extended UI modes to base styles
    if m in ("chevron_patch", "rubber_patch", "metal_plate", "metal_badge", "badge", "patch", "label", "foil", "iron_on", "ironon", "tag"):
        return "sticker"
    if m in ("silicone_3d",):
        return "embroidery"
    if m in ("silk_screen", "heat_press", "sublimation", "print", "dtf"):
        return "print_dtf"
    if m in ("deboss",):
        return "embroidery"
    return m


def _auto_cutout_background(design: Image.Image) -> Image.Image:
    """
    Best-effort background removal for design assets that are NOT transparent (common for JPG screenshots / photos).
    Goals:
    - Be CONSERVATIVE: never cut into the subject/artwork (avoid eating highlights/white parts inside).
    - Remove only the background that is CONNECTED TO THE BORDER (flood-fill), then feather edges.
    - Works for near-uniform backgrounds of any color (white/black/gray/etc) when border variance is low.
    If background is not uniform enough, we keep the image as-is and let Gemini handle removal per prompt rules.
    """
    if design.mode != "RGBA":
        design = design.convert("RGBA")

    # If it already has transparency, keep it (avoid destroying PNG logos)
    a = design.split()[-1]
    if a.getextrema()[0] < 250:  # has some transparency already
        return design

    w, h = design.size
    if w < 16 or h < 16:
        return design

    px = design.load()

    # --- sample border pixels (for bg color + uniformity) ---
    samples = []
    step = max(1, min(w, h) // 50)
    for x in range(0, w, step):
        samples.append(px[x, 0][:3])
        samples.append(px[x, h - 1][:3])
    for y in range(0, h, step):
        samples.append(px[0, y][:3])
        samples.append(px[w - 1, y][:3])

    if not samples:
        return design

    # Mean + stddev of border to decide if bg is "uniform enough"
    mr = sum(s[0] for s in samples) / len(samples)
    mg = sum(s[1] for s in samples) / len(samples)
    mb = sum(s[2] for s in samples) / len(samples)
    bg = (mr, mg, mb)

    # stddev
    vr = sum((s[0] - mr) ** 2 for s in samples) / len(samples)
    vg = sum((s[1] - mg) ** 2 for s in samples) / len(samples)
    vb = sum((s[2] - mb) ** 2 for s in samples) / len(samples)
    std = ((vr + vg + vb) / 3.0) ** 0.5

    # If border is not uniform, DO NOT attempt (prevents damaging complex designs/photos)
    # You can tune via env, but default is conservative.
    try:
        std_thr = float(os.getenv("PRINTS_CUTOUT_BORDER_STD_THR", "22"))
    except Exception:
        std_thr = 22.0
    if std > std_thr:
        return design

    # thresholds (distance in RGB space)
    # luma-based tuning, but allow any uniform bg color
    bg_luma = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    if bg_luma < 60:
        thr0, thr1 = 26.0, 80.0   # dark bg
    elif bg_luma > 195:
        thr0, thr1 = 16.0, 62.0   # bright bg
    else:
        thr0, thr1 = 18.0, 70.0   # mid bg

    # Build distance field (optionally with numpy for speed)
    try:
        import numpy as _np
        arr = _np.array(design, dtype=_np.float32)  # RGBA
        rgb = arr[..., :3]
        bgv = _np.array([bg[0], bg[1], bg[2]], dtype=_np.float32).reshape((1,1,3))
        dist = _np.sqrt(_np.sum((rgb - bgv) ** 2, axis=2))
        cand = (dist <= thr1)  # background candidates (loose)
    except Exception:
        dist = None
        cand = [[False]*w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                pr, pg, pb = px[x, y][:3]
                dr = pr - bg[0]
                dg = pg - bg[1]
                db = pb - bg[2]
                d = (dr*dr + dg*dg + db*db) ** 0.5
                cand[y][x] = (d <= thr1)

    # Flood-fill ONLY from borders on candidate pixels
    bg_conn = None
    try:
        import numpy as _np
        bg_conn = _np.zeros((h, w), dtype=_np.uint8)
        q = deque()

        def push(x, y):
            if x < 0 or y < 0 or x >= w or y >= h:
                return
            if bg_conn[y, x]:
                return
            if not cand[y, x]:
                return
            bg_conn[y, x] = 1
            q.append((x, y))

        for x in range(w):
            push(x, 0); push(x, h-1)
        for y in range(h):
            push(0, y); push(w-1, y)

        while q:
            x, y = q.popleft()
            push(x+1, y); push(x-1, y); push(x, y+1); push(x, y-1)
    except Exception:
        bg_conn = [[0]*w for _ in range(h)]
        q = deque()

        def push2(x, y):
            if x < 0 or y < 0 or x >= w or y >= h:
                return
            if bg_conn[y][x]:
                return
            if not cand[y][x]:
                return
            bg_conn[y][x] = 1
            q.append((x, y))

        for x in range(w):
            push2(x, 0); push2(x, h-1)
        for y in range(h):
            push2(0, y); push2(w-1, y)

        while q:
            x, y = q.popleft()
            push2(x+1, y); push2(x-1, y); push2(x, y+1); push2(x, y-1)

    # Create alpha: only for border-connected background pixels we fade out.
    newA = Image.new("L", (w, h), 255)
    ap = newA.load()

    if dist is not None:
        # numpy path
        import numpy as _np
        # ramp alpha from dist
        a_ramp = _np.clip((dist - thr0) / max(1e-6, (thr1 - thr0)), 0.0, 1.0) * 255.0
        a_ramp = a_ramp.astype(_np.uint8)
        if isinstance(bg_conn, _np.ndarray):
            # bg pixels get ramp alpha, others stay 255
            out = _np.full((h, w), 255, dtype=_np.uint8)
            out[bg_conn.astype(bool)] = a_ramp[bg_conn.astype(bool)]
            newA = Image.fromarray(out, mode="L")
        else:
            # fallback - shouldn't happen
            pass
    else:
        # pure python path
        for y in range(h):
            for x in range(w):
                if bg_conn[y][x]:
                    pr, pg, pb = px[x, y][:3]
                    dr = pr - bg[0]
                    dg = pg - bg[1]
                    db = pb - bg[2]
                    d = (dr*dr + dg*dg + db*db) ** 0.5
                    if d <= thr0:
                        aa = 0
                    elif d >= thr1:
                        aa = 255
                    else:
                        aa = int(255.0 * (d - thr0) / (thr1 - thr0))
                    ap[x, y] = aa
                else:
                    ap[x, y] = 255

    # Post-process alpha:
    # - close tiny pinholes in background region
    # - soften edge (keep soft outline, avoid cutting subject)
    try:
        newA = newA.filter(ImageFilter.MedianFilter(3))
        newA = newA.filter(ImageFilter.GaussianBlur(0.8))
    except Exception:
        pass
    # Hard-zero only very small alpha to kill haze
    newA = newA.point(lambda v: 0 if v < 10 else v)

    img = design.copy()

    # De-matte / de-halo edges: remove background fringing from semi-transparent pixels
    try:
        bgR, bgG, bgB = bg
        import numpy as _np
        _arr = _np.array(img).astype(_np.float32)  # RGBA
        _a = ( _np.array(newA).astype(_np.float32) / 255.0 )[..., None]
        _rgb = _arr[..., :3]
        _mask = (_a > 0.0) & (_a < 1.0)
        if _mask.any():
            _bg = _np.array([bgR, bgG, bgB], dtype=_np.float32).reshape((1,1,3))
            _rgb2 = (_rgb - _bg * (1.0 - _a)) / _np.clip(_a, 1e-3, 1.0)
            _rgb = _np.clip(_rgb2, 0.0, 255.0)
            _arr[..., :3] = _rgb
            img = Image.fromarray(_arr.astype(_np.uint8), mode="RGBA")
    except Exception:
        pass

    img.putalpha(newA)
    return img


def _alpha_coverage(img: Image.Image, thr: int = 12) -> float:
    """Fraction of pixels with alpha > thr."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    a = img.split()[-1]
    # downsample for speed
    a_small = a.resize((max(8, a.width // 4), max(8, a.height // 4)), Image.NEAREST)
    ap = a_small.getdata()
    total = len(ap)
    if total <= 0:
        return 1.0
    on = sum(1 for v in ap if v > thr)
    return on / float(total)

def _border_alpha_mean(img: Image.Image) -> float:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    a = img.split()[-1]
    w, h = a.size
    if w < 3 or h < 3:
        return 255.0
    px = a.load()
    samples = []
    step = max(1, min(w, h) // 40)
    for x in range(0, w, step):
        samples.append(px[x, 0])
        samples.append(px[x, h - 1])
    for y in range(0, h, step):
        samples.append(px[0, y])
        samples.append(px[w - 1, y])
    return float(sum(samples) / max(1, len(samples)))

def _auto_cutout_background_anycolor(design: Image.Image) -> Image.Image:
    """Background removal for non-transparent designs with *any* mostly-solid background color.

    Uses border sampling to estimate background RGB, then removes pixels close to that color.
    This is a deterministic fallback when the 'near black/white' heuristic is not enough.
    """
    if design.mode != "RGBA":
        design = design.convert("RGBA")

    a = design.split()[-1]
    if a.getextrema()[0] < 250:
        return design

    w, h = design.size
    if w < 10 or h < 10:
        return design

    px = design.load()
    samples = []
    step = max(1, min(w, h) // 50)
    for x in range(0, w, step):
        samples.append(px[x, 0][:3])
        samples.append(px[x, h - 1][:3])
    for y in range(0, h, step):
        samples.append(px[0, y][:3])
        samples.append(px[w - 1, y][:3])

    # If border colors are too diverse, likely not a solid bg -> bail
    rs = [s[0] for s in samples]; gs = [s[1] for s in samples]; bs = [s[2] for s in samples]
    def _std(vs):
        if len(vs) < 2:
            return 0.0
        m = sum(vs)/len(vs)
        return (sum((x-m)*(x-m) for x in vs)/len(vs))**0.5
    if max(_std(rs), _std(gs), _std(bs)) > 22.0:
        return design

    bg = (sum(rs)/len(rs), sum(gs)/len(gs), sum(bs)/len(bs))
    # adaptive thresholds based on border noise
    noise = max(_std(rs), _std(gs), _std(bs))
    thr0 = 14.0 + noise * 0.6
    thr1 = 52.0 + noise * 1.1

    img = design.copy()
    rp = img.load()
    newA = Image.new("L", (w, h), 255)
    ap = newA.load()

    for y in range(h):
        for x in range(w):
            pr, pg, pb, _pa = rp[x, y]
            dr = pr - bg[0]
            dg = pg - bg[1]
            db = pb - bg[2]
            dist = (dr*dr + dg*dg + db*db) ** 0.5
            if dist <= thr0:
                aa = 0
            elif dist >= thr1:
                aa = 255
            else:
                aa = int(255.0 * (dist - thr0) / (thr1 - thr0))
            ap[x, y] = aa

    newA = newA.filter(ImageFilter.MedianFilter(3))
    newA = newA.filter(ImageFilter.MaxFilter(3))
    newA = newA.filter(ImageFilter.GaussianBlur(0.7))
    newA = newA.point(lambda v: 0 if v < 12 else v)

    # De-matte / de-halo edges: remove background color fringing from semi-transparent pixels
    try:
        bgR, bgG, bgB = bg  # (floats)
        # work in numpy for speed; fall back silently if unavailable
        import numpy as _np
        _arr = _np.array(img).astype(_np.float32)  # RGBA
        _a = _arr[..., 3:4] / 255.0
        _rgb = _arr[..., :3]
        # only adjust where 0<alpha<1
        _mask = (_a > 0.0) & (_a < 1.0)
        if _mask.any():
            _bg = _np.array([bgR, bgG, bgB], dtype=_np.float32).reshape((1,1,3))
            _rgb2 = (_rgb - _bg * (1.0 - _a)) / _np.clip(_a, 1e-3, 1.0)
            _rgb = _np.clip(_rgb2, 0.0, 255.0)
            _arr[..., :3] = _rgb
            img = Image.fromarray(_arr.astype(_np.uint8), mode='RGBA')
    except Exception:
        pass
    img.putalpha(newA)
    return img

def _cutout_guard(design: Image.Image) -> Image.Image:
    """Run cutout and verify it actually removed the background.

    NOTE: Many user-provided "PNGs" are actually screenshots/JPGs with a white matte.
    We try cutout first, then an any-color cutout, and finally a deterministic
    white-matte suppression to avoid residual rectangular backgrounds.
    """
    # NOTE (2026-03): local cutout is DISABLED by default because it often destroys details (e.g., people/complex art).
    # We now rely on Gemini prompt rules + MASK zone to ignore any design background.
    if os.getenv("PRINTS_LOCAL_CUTOUT_ENABLE", "0") == "0":
        return design.convert("RGBA") if getattr(design, 'mode', None) != "RGBA" else design


    d1 = _auto_cutout_background(design)
    cov = _alpha_coverage(d1)
    bord = _border_alpha_mean(d1)

    # If almost everything is opaque AND border is opaque, background likely still present.
    if cov > 0.90 and bord > 240.0:
        d2 = _auto_cutout_background_anycolor(design)
        # keep the better one (lower opaque coverage)
        if _alpha_coverage(d2) < cov - 0.03:
            d1 = d2
            cov = _alpha_coverage(d1)
            bord = _border_alpha_mean(d1)

    # Final guard: if border is still opaque, try suppressing white matte in RGB
    # (common for logos/art pasted on white background).
    try:
        if _border_alpha_mean(d1) > 200.0:
            d1 = _suppress_white_matte(d1)
    except Exception:
        pass


    # If border is still opaque (but not necessarily white), try a generic matte suppression.
    try:
        if _border_alpha_mean(d1) > 200.0:
            d1 = _suppress_border_matte_anycolor(d1)
    except Exception:
        pass

    return d1



def _suppress_white_matte(img: "Image.Image", near: int = 245, softness: int = 40, border_px: int = 8) -> "Image.Image":
    """Reduce alpha where RGB is near-white (common white matte / screenshot backgrounds).

    - Only activates aggressively if border RGB is near-white.
    - Keeps true white details inside the design by requiring pixels to be *very* close to white.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img).astype(np.float32)
    rgb = arr[..., :3]
    a = arr[..., 3]

    h, w = a.shape

    # Check border whiteness to decide strength
    b = border_px
    b = max(1, min(b, min(h, w)//10))
    border = np.concatenate([
        rgb[:b, :, :].reshape(-1, 3),
        rgb[-b:, :, :].reshape(-1, 3),
        rgb[:, :b, :].reshape(-1, 3),
        rgb[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    border_min = np.min(border, axis=1)  # min(R,G,B)
    border_white_ratio = float(np.mean(border_min >= near))

    # If border is not mostly white, do a very mild suppression only (avoid damaging art)
    strength = 1.0 if border_white_ratio >= 0.60 else 0.35

    m = np.min(rgb, axis=2)  # 0..255
    # t in [0..1] where 1 means "definitely background white"
    t = np.clip((m - float(near)) / max(1.0, float(softness)), 0.0, 1.0) * strength

    # Only suppress where alpha is already high (i.e. background kept opaque)
    gate = (a >= 40.0).astype(np.float32)
    new_a = a * (1.0 - t * gate)

    arr[..., 3] = np.clip(new_a, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def _suppress_border_matte_anycolor(img: "Image.Image", *, tol: int = 24, softness: int = 32, border_px: int = 8) -> "Image.Image":
    """Remove a flat matte/background color estimated from the image border.

    Works for screenshots/JPGs pasted on any solid background (white/gray/colored).
    We estimate the dominant border RGB and fade alpha for pixels close to it.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img).astype(np.float32)
    rgb = arr[..., :3]
    a = arr[..., 3]

    h, w = a.shape
    b = max(1, min(border_px, min(h, w)//10))

    border = np.concatenate([
        rgb[:b, :, :].reshape(-1, 3),
        rgb[-b:, :, :].reshape(-1, 3),
        rgb[:, :b, :].reshape(-1, 3),
        rgb[:, -b:, :].reshape(-1, 3),
    ], axis=0)

    mean = border.mean(axis=0)
    std = border.std(axis=0).mean()

    # If border color isn't consistent, do nothing.
    if std > 18.0:
        return img

    diff = np.max(np.abs(rgb - mean[None, None, :]), axis=2)  # Chebyshev distance
    # Build matte mask with soft falloff
    t0 = float(max(1, tol))
    t1 = t0 + float(max(1, softness))
    m = np.clip((diff - t0) / (t1 - t0), 0.0, 1.0)  # 0 near matte, 1 away

    # Fade alpha near matte
    a2 = a * m

    out = np.dstack([rgb, a2]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _fabric_deform_and_shade(rgba: "Image.Image", base: "Image.Image", bbox: tuple, mode_norm: str) -> "Image.Image":
    """Fabric-aware micro deformation + shading.

    Goal: preserve crisp typography while still "sitting" on fabric (wrinkles/light).
    We do:
      - micro warp (very small) driven by base luminance gradients
      - multiply-style shading to inherit fabric lighting
    """
    try:
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(base.width, int(x1)); y1 = min(base.height, int(y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return rgba

        roi_base = base.crop((x0, y0, x1, y1)).convert("RGB")
        roi = rgba.crop((x0, y0, x1, y1)).convert("RGBA")

        ba = np.array(roi_base).astype(np.float32)
        da = np.array(roi).astype(np.float32)
        d_rgb = da[..., :3]
        d_a = da[..., 3] / 255.0

        # strength knobs (env), with safe defaults
        warp_on = os.getenv("PRINTS_FABRIC_WARP", "1").strip() not in ("0", "false", "False")
        shade_on = os.getenv("PRINTS_FABRIC_SHADE", "1").strip() not in ("0", "false", "False")

        # For strict typography modes, keep warp very small
        strict = mode_norm in ("dtf", "embroidery", "vyshyvanka_panel", "embroidery_panel")
        max_warp = float(os.getenv("PRINTS_FABRIC_WARP_PX_STRICT" if strict else "PRINTS_FABRIC_WARP_PX", "1.2" if strict else "3.0"))
        edge_protect = float(os.getenv("PRINTS_FABRIC_EDGE_PROTECT", "10.0" if strict else "6.0"))
        shade_gain = float(os.getenv("PRINTS_FABRIC_SHADE_GAIN", "0.22" if strict else "0.30"))

        h, w = d_a.shape

        # Build wrinkle/gradient maps from base ROI
        gray = cv2.cvtColor(ba.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # High-pass-ish component
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        hi = np.clip(gray - blur, -1.0, 1.0)

        if warp_on and max_warp > 0.0:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            # normalize
            mag = np.sqrt(gx * gx + gy * gy) + 1e-6
            gx /= mag; gy /= mag
            # warp field from hi-frequency + gradient direction
            field = cv2.GaussianBlur(hi, (0, 0), 1.2)
            dx = gx * field * max_warp
            dy = gy * field * max_warp

            # edge-protect: reduce warp near alpha edges
            alpha_u8 = np.clip(d_a * 255.0, 0, 255).astype(np.uint8)
            _, binm = cv2.threshold(alpha_u8, 16, 255, cv2.THRESH_BINARY)
            dist = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
            dist = np.clip(dist / max(1e-3, edge_protect), 0.0, 1.0)
            dx *= dist
            dy *= dist

            # remap
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            map_x = (xs + dx).astype(np.float32)
            map_y = (ys + dy).astype(np.float32)

            # cv2.remap expects BGRA if we pass 4ch
            d_bgra = da[..., [2, 1, 0, 3]].astype(np.uint8)
            rem = cv2.remap(d_bgra, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            da = rem[..., [2, 1, 0, 3]].astype(np.float32)
            d_rgb = da[..., :3]
            d_a = da[..., 3] / 255.0

        if shade_on and shade_gain > 0.0:
            # shading multiplier from base luminance (wrinkles & lighting)
            # range ~ [1-shade_gain .. 1+shade_gain]
            sh = np.clip(1.0 + hi * shade_gain, 1.0 - shade_gain, 1.0 + shade_gain)
            d_rgb = d_rgb * sh[..., None]

        out = np.dstack([np.clip(d_rgb, 0, 255), np.clip(d_a * 255.0, 0, 255)]).astype(np.uint8)
        roi2 = Image.fromarray(out, "RGBA")

        # paste back
        out_full = rgba.copy()
        out_full.paste(roi2, (x0, y0), roi2)
        return out_full
    except Exception:
        return rgba


def _dehalo_alpha_hard(img: "Image.Image", thr: int = 210) -> "Image.Image":
    """Hard-remove semi-transparent halo pixels.
    Many logos have a faint matte (white/gray) in semi-transparent edge pixels which becomes visible on fabric.
    We fix this deterministically by zeroing pixels with alpha below a threshold.
    """
    try:
        from PIL import Image
        import numpy as _np
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = _np.array(img, dtype=_np.uint8)
        a = arr[..., 3]
        # zero low-alpha pixels (kills haze)
        mask = a < thr
        if mask.any():
            arr[mask] = 0
        return Image.fromarray(arr, mode="RGBA")
    except Exception:
        return img



def _arc_warp_design(design_rgba: Image.Image, warp: dict) -> Image.Image:
    """Cheap 2-pass arc warp to roughly match UI mid-handle bends.
    warp keys: top/right/bottom/left in [-1..1]. Positive = pull outward.
    This is NOT a true cloth simulation; it's a light preview-aligned deformation.
    """
    try:
        w = warp or {}
        top = float(w.get("top", 0.0) or 0.0)
        right = float(w.get("right", 0.0) or 0.0)
        bottom = float(w.get("bottom", 0.0) or 0.0)
        left = float(w.get("left", 0.0) or 0.0)
    except Exception:
        return design_rgba

    if max(abs(top), abs(right), abs(bottom), abs(left)) < 1e-3:
        return design_rgba

    img = design_rgba.convert("RGBA")
    W, H = img.size
    if W < 4 or H < 4:
        return img

    # combine opposite sides into axis curves (same as UI preview)
    curve_v = max(-1.0, min(1.0, (bottom - top)))
    curve_h = max(-1.0, min(1.0, (right - left)))

    # PASS 1: vertical (shift columns)
    amp_v = curve_v * (H * 0.28)
    step_x = 2
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for x in range(0, W, step_x):
        x2 = min(W, x + step_x)
        t = (x + (x2 - x) * 0.5 - W / 2.0) / (W / 2.0)
        wv = 1.0 - t * t  # 0..1..0
        y_shift = int(round(-amp_v * wv))
        col = img.crop((x, 0, x2, H))
        tmp.alpha_composite(col, (x, y_shift))

    # PASS 2: horizontal (shift rows)
    amp_h = curve_h * (W * 0.22)
    step_y = 2
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for y in range(0, H, step_y):
        y2 = min(H, y + step_y)
        t = (y + (y2 - y) * 0.5 - H / 2.0) / (H / 2.0)
        wv = 1.0 - t * t
        x_shift = int(round(-amp_h * wv))
        row = tmp.crop((0, y, W, y2))
        out.alpha_composite(row, (x_shift, y))

    return out


def _foreground_mask_by_bg(base_rgba: Image.Image, quad: List[Tuple[float, float]]) -> Image.Image:
    """Best-effort occlusion/visibility mask without ML.

    We estimate background color from pixels just outside the quad bounding box and keep pixels
    that differ from background. This clips the design to the visible garment/body when the quad
    extends into a flat studio background.

    Returns an L mask (0..255).
    """
    base = base_rgba.convert("RGB")
    w, h = base.size
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0 = max(0, int(min(xs)) - 10)
    y0 = max(0, int(min(ys)) - 10)
    x1 = min(w - 1, int(max(xs)) + 10)
    y1 = min(h - 1, int(max(ys)) + 10)

    crop = base.crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = crop.size
    cp = crop.load()

    # Sample a thin frame near crop border to estimate background
    samples = []
    step = max(1, min(cw, ch) // 40)
    for x in range(0, cw, step):
        samples.append(cp[x, 0])
        samples.append(cp[x, ch - 1])
    for y in range(0, ch, step):
        samples.append(cp[0, y])
        samples.append(cp[cw - 1, y])

    r = sum(s[0] for s in samples) / max(1, len(samples))
    g = sum(s[1] for s in samples) / max(1, len(samples))
    b = sum(s[2] for s in samples) / max(1, len(samples))
    bg = (r, g, b)

    mask = Image.new("L", (cw, ch), 0)
    mp = mask.load()

    # Threshold tuned for plain studio backgrounds (your common case)
    thr = 28.0
    for y in range(ch):
        for x in range(cw):
            pr, pg, pb = cp[x, y]
            dr = pr - bg[0]
            dg = pg - bg[1]
            db = pb - bg[2]
            dist = (dr*dr + dg*dg + db*db) ** 0.5
            mp[x, y] = 255 if dist >= thr else 0

    # smooth and slightly grow to avoid cutting garment edges
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))

    full = Image.new("L", (w, h), 0)
    full.paste(mask, (x0, y0))
    return full

def _maybe_supersample_design(design: Image.Image, quad: List[Tuple[float, float]]) -> Image.Image:
    """If target quad is very small, upscale design before warp to preserve text/detail."""
    try:
        # approximate target size in pixels
        (x0,y0),(x1,y1),(x2,y2),(x3,y3) = quad
        tw = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
        th = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        tmin = max(1.0, min(tw, th))
        # If very small, upscale (capped)
        if tmin < 140:
            scale = min(6, max(2, int(math.ceil(140 / tmin))))
            nw = int(design.width * scale)
            nh = int(design.height * scale)
            return design.resize((nw, nh), resample=Image.LANCZOS)
    except Exception:
        pass
    return design

def _quad_size(quad: List[Tuple[float, float]]) -> Tuple[float, float]:
    (x0,y0),(x1,y1),(x2,y2),(x3,y3) = quad
    w = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
    h = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    return max(1.0, w), max(1.0, h)

def _pad_design_to_aspect(design: Image.Image, target_aspect: float) -> Image.Image:
    """Pad (letterbox) design with transparency to match target aspect without distorting the logo."""
    if design.mode != "RGBA":
        design = design.convert("RGBA")
    w, h = design.size
    if w < 2 or h < 2:
        return design
    src_aspect = w / float(h)
    if target_aspect <= 0:
        return design
    # If already close enough, keep
    if abs(src_aspect - target_aspect) / max(0.0001, target_aspect) < 0.06:
        return design

    if src_aspect > target_aspect:
        # too wide -> pad height
        new_h = int(round(w / target_aspect))
        pad_top = (new_h - h) // 2
        out = Image.new("RGBA", (w, new_h), (0,0,0,0))
        out.paste(design, (0, pad_top))
        return out
    else:
        # too tall -> pad width
        new_w = int(round(h * target_aspect))
        pad_left = (new_w - w) // 2
        out = Image.new("RGBA", (new_w, h), (0,0,0,0))
        out.paste(design, (pad_left, 0))
        return out

def _save_png(img: Image.Image) -> str:
    """
    Save img as PNG in assets with content hash, return absolute URL.
    """
    _ensure_dir()
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    h = hashlib.sha256(raw).hexdigest()[:16]
    fn = f"{h}.png"
    path = os.path.join(ASSETS_DIR, fn)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    return asset_url(fn)


def _make_thumb(img: Image.Image, max_side: int = 360) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img.copy()
    scale = max_side / float(s)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def _coeffs_perspective(src, dst):
    # Prefer pure-Python solver (no numpy dependency)
    return _coeffs_perspective_no_np(src, dst)


def _coeffs_perspective_no_np(src, dst):
    # Minimal 8x8 solver without numpy (Gauss). Rarely used.
    A = []
    B = []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)

    # Gauss elimination
    n = 8
    M = [A[i] + [B[i]] for i in range(n)]
    for col in range(n):
        # pivot
        pivot = col
        for r in range(col, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("Bad transform")
        # normalize
        div = M[col][col]
        for c in range(col, n+1):
            M[col][c] /= div
        # eliminate
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            for c in range(col, n+1):
                M[r][c] -= factor * M[col][c]
    return tuple(M[i][n] for i in range(n))


def _perspective_coeffs(src, dst):
    return _coeffs_perspective_no_np(src, dst)


def _transform_rgba_perspective_premultiplied(img_rgba: Image.Image, coeffs, out_size: tuple[int,int], resample=Image.Resampling.BICUBIC) -> Image.Image:
    """Perspective-transform RGBA without black/colored fringe: premultiply RGB by alpha, transform RGB and alpha separately, then unpremultiply."""
    img_rgba = img_rgba.convert("RGBA")
    w,h = out_size
    r,g,b,a = img_rgba.split()
    # premultiply
    import numpy as _np
    rgb = Image.merge("RGB",(r,g,b))
    rgb_np = _np.array(rgb).astype("float32")
    a_np = _np.array(a).astype("float32")/255.0
    rgb_np *= a_np[...,None]
    rgb_premul = Image.fromarray(_np.clip(rgb_np,0,255).astype("uint8"), mode="RGB")
    # transform
    rgb_w = rgb_premul.transform((w,h), Image.Transform.PERSPECTIVE, coeffs, resample=resample, fillcolor=(0,0,0))
    a_w = a.transform((w,h), Image.Transform.PERSPECTIVE, coeffs, resample=resample, fillcolor=0)
    # unpremultiply
    rgbw_np = _np.array(rgb_w).astype("float32")
    aw_np = _np.array(a_w).astype("float32")/255.0
    # avoid divide by zero
    denom = _np.maximum(aw_np[...,None], 1e-6)
    rgb_un = rgbw_np/denom
    rgb_un = _np.clip(rgb_un,0,255).astype("uint8")
    out_rgb = Image.fromarray(rgb_un, mode="RGB")
    return Image.merge("RGBA", (*out_rgb.split(), a_w))

def _apply_style(warped: Image.Image, mode: str) -> Image.Image:
    """
    warped: RGBA logo already perspective-warped to base size.
    mode: print_dtf | embroidery | sticker
    Returns styled RGBA overlay.
    """
    mode = _normalize_mode(mode)

    # Split alpha for effects
    r, g, b, a = warped.split()
    alpha = a

    if mode == "print_dtf":
        # soft edge + slight shadow, keep colors
        soft = alpha.filter(ImageFilter.GaussianBlur(0.8))
        shadow = soft.filter(ImageFilter.GaussianBlur(2.2))
        shadow = ImageOps.colorize(shadow, black="#000000", white="#000000").convert("RGBA")
        shadow.putalpha(shadow.split()[-1].point(lambda x: int(x * 0.20)))
        out = Image.alpha_composite(shadow, warped)
        return out

    if mode == "sticker":
        # add white border and subtle drop shadow
        border = alpha.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.0))
        border_rgba = Image.new("RGBA", warped.size, (255, 255, 255, 0))
        border_rgba.putalpha(border.point(lambda x: 255 if x > 0 else 0))
        border_rgba = border_rgba.filter(ImageFilter.GaussianBlur(0.8))

        shadow = border.filter(ImageFilter.GaussianBlur(3.0))
        shadow_rgba = Image.new("RGBA", warped.size, (0, 0, 0, 0))
        shadow_rgba.putalpha(shadow.point(lambda x: int(x * 0.22)))

        tmp = Image.alpha_composite(shadow_rgba, border_rgba)
        tmp = Image.alpha_composite(tmp, warped)
        return tmp

    if mode == "embroidery":
        # simulate thread: desaturate a bit, add emboss and highlights
        base = warped.copy()
        # reduce saturation slightly by blending with gray
        gray = ImageOps.grayscale(base).convert("RGBA")
        base = Image.blend(base, gray, 0.18)

        # texture via emboss on alpha edge
        edge = alpha.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.0))
        highlight = ImageOps.colorize(edge, black="#000000", white="#ffffff").convert("RGBA")
        highlight.putalpha(edge.point(lambda x: int(x * 0.35)))

        # subtle bevel
        bevel = alpha.filter(ImageFilter.EMBOSS)
        bevel_rgba = ImageOps.colorize(bevel, black="#000000", white="#ffffff").convert("RGBA")
        bevel_rgba.putalpha(alpha.point(lambda x: int(x * 0.18)))

        tmp = Image.alpha_composite(bevel_rgba, base)
        tmp = Image.alpha_composite(tmp, highlight)
        return tmp

    if mode == "tattoo":
        # Ink-like blend: darken/multiply with slightly softened edges.
        # Keep color but reduce saturation and opacity.
        base = warped.copy()
        gray = ImageOps.grayscale(base).convert("RGBA")
        base = Image.blend(base, gray, 0.35)
        # soften edges a bit
        a2 = base.split()[-1].filter(ImageFilter.GaussianBlur(0.6))
        # reduce opacity
        a2 = a2.point(lambda x: int(x * 0.62))
        base.putalpha(a2)
        return base

    raise ValueError("Неизвестный вид нанесения. Доступно: print_dtf, embroidery, sticker, tattoo (а также: chevron_patch, rubber_patch, silicone_3d, metal_plate, silk_screen, heat_press, deboss, sublimation)")


@dataclass
class ApplyIn:
    base_url: str
    design_url: str
    mode: str
    quad: List[Tuple[float, float]]  # 4 points in base-image pixel coords
    # If True: placement is ABSOLUTE (do not move/recenter/resize). If False: allow slight adjustment inside the quad.
    placement_lock: bool = True
    # If 'full': allow full re-render while preserving scene (validated by engine).
    # If 'mask_only': keep original outside mask.
    render_mode: str = "full"




def _compute_fit_scale_for_warp(warp: dict) -> float:
    """Compute the same fitScale as the frontend preview (PrintsPage.jsx).
    fitScale = clamp(1 - 2*inset, 0.1, 1), where inset = clamp(0.04 + max(needV, needH), 0, 0.35)
    needV = abs(curveV)*0.28, needH = abs(curveH)*0.22
    curveV = bottom - top, curveH = right - left
    """
    # If frontend already computed fitScale/inset, trust it for pixel-perfect preview parity
    try:
        fs = warp.get("fitScale", None)
        if fs is not None:
            fs = float(fs)
            return max(0.1, min(1.0, fs))
        ins = warp.get("inset", None)
        if ins is not None:
            ins = float(ins)
            fs2 = 1.0 - ins * 2.0
            return max(0.1, min(1.0, fs2))
    except Exception:
        pass

    try:
        top = float(warp.get("top", 0) or 0)
        right = float(warp.get("right", 0) or 0)
        bottom = float(warp.get("bottom", 0) or 0)
        left = float(warp.get("left", 0) or 0)
    except Exception:
        return 0.92  # baseInset only

    # clamp inputs
    top = max(-1.0, min(1.0, top))
    right = max(-1.0, min(1.0, right))
    bottom = max(-1.0, min(1.0, bottom))
    left = max(-1.0, min(1.0, left))

    curve_v = max(-1.0, min(1.0, bottom - top))
    curve_h = max(-1.0, min(1.0, right - left))

    need_v = abs(curve_v) * 0.28
    need_h = abs(curve_h) * 0.22

    base_inset = 0.04
    inset = base_inset + max(need_v, need_h)
    inset = max(0.0, min(0.35, inset))

    fit_scale = 1.0 - inset * 2.0
    fit_scale = max(0.1, min(1.0, fit_scale))
    return fit_scale

def _autofit_design_for_warp(design: Image.Image, warp: dict) -> Image.Image:
    """
    Keep the whole warped design visible (no clipping) by adding transparent padding and scaling down.
    Must mirror the frontend preview logic (PrintsPage.jsx canvas preview).

    We treat opposite side handles as independent bends and convert them into axis curves:
      curve_v = bottom - top
      curve_h = right - left

    Arc warp can shift pixels outside the bbox by up to:
      abs(curve_v) * 0.28 * H   (vertical)
      abs(curve_h) * 0.22 * W   (horizontal)

    To keep everything visible, we pre-scale the design into an inner "safe" area
    with a small constant breathing room.
    """
    try:
        top = float(warp.get("top", 0) or 0)
        right = float(warp.get("right", 0) or 0)
        bottom = float(warp.get("bottom", 0) or 0)
        left = float(warp.get("left", 0) or 0)
    except Exception:
        return design

    # clamp inputs
    top = max(-1.0, min(1.0, top))
    right = max(-1.0, min(1.0, right))
    bottom = max(-1.0, min(1.0, bottom))
    left = max(-1.0, min(1.0, left))

    curve_v = max(-1.0, min(1.0, bottom - top))
    curve_h = max(-1.0, min(1.0, right - left))

    need_v = abs(curve_v) * 0.28  # fraction of H
    need_h = abs(curve_h) * 0.22  # fraction of W

    base_inset = 0.04  # constant safe inset (matches frontend)
    inset = base_inset + max(need_v, need_h)
    inset = max(0.0, min(0.35, inset))

    fit_scale = 1.0 - inset * 2.0
    if fit_scale >= 0.999:
        return design
    if fit_scale <= 0.05:
        fit_scale = 0.05

    w, h = design.size
    tw = max(1, int(round(w * fit_scale)))
    th = max(1, int(round(h * fit_scale)))

    scaled = design.resize((tw, th), Image.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = (w - tw) // 2
    oy = (h - th) // 2
    out.alpha_composite(scaled, (ox, oy))
    return out
def apply_design(inp: ApplyIn):
    base = _load_image_from_url(inp.base_url)
    design = _load_image_from_url(inp.design_url).convert("RGBA")

    # Best-effort background cutout for non-alpha logos (prevents square plates)
    try:
        design = _cutout_guard(design)
    except Exception:
        pass

    # Optional arc warp is handled later (after quad scaling) to match frontend preview.

    # --- CANON: upscale base to at least 2K long side (and scale quad accordingly) ---
    orig_w, orig_h = base.size
    if not inp.quad or len(inp.quad) != 8:
        raise ValueError('Missing or invalid quad (expected 8 numbers)')
    quad = _quad_to_pixels(inp.quad, orig_w, orig_h)
    # Optional horizontal flip if frontend coordinate space is mirrored
    flip_x_opt = None
    try:
        if isinstance(getattr(inp, 'options', None), dict):
            flip_x_opt = inp.options.get('flipX')
    except Exception:
        flip_x_opt = None
    if flip_x_opt is True or _force_flip_x_enabled() or (flip_x_opt is None and _auto_flip_x_enabled()):
        quad_px = _flip_quad_x(quad_px, W)
    base, sx, sy = _ensure_min_long_side(base, 2048)
    quad = _scale_quad(quad, sx, sy)

    # --- Preview parity: render design into quad bbox with the same safe inset as frontend ---
    # Frontend preview first draws the (possibly non-square) design into a dw×dh canvas (quad bbox),
    # with a constant base inset and an additional inset derived from bend (warp) amount.
    # Then it applies arc-warp and finally perspective-maps that dw×dh image into the quad.
    # To match preview 1:1, we replicate that exact pipeline here.
    try:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bx0 = float(min(xs)); by0 = float(min(ys))
        bx1 = float(max(xs)); by1 = float(max(ys))
        dw = max(1, int(round(bx1 - bx0)))
        dh = max(1, int(round(by1 - by0)))

        warp = getattr(inp, 'warp', None) or {}
        fit_scale = _compute_fit_scale_for_warp(warp)

        src_w = max(1, int(round(dw * fit_scale)))
        src_h = max(1, int(round(dh * fit_scale)))
        # NOTE: this intentionally stretches the design to match the quad bbox aspect (same as canvas drawImage).
        design_scaled = design.resize((src_w, src_h), Image.LANCZOS)

        design_rect = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        ox = int(round((dw - src_w) / 2.0))
        oy = int(round((dh - src_h) / 2.0))
        ox = max(0, min(dw - src_w, ox))
        oy = max(0, min(dh - src_h, oy))
        design_rect.alpha_composite(design_scaled, (ox, oy))

        # Optional arc warp (from UI mid-handles) on the dw×dh canvas
        if getattr(inp, 'warp', None):
            design_rect = _arc_warp_design(design_rect, warp)

        design = design_rect
    except Exception:
        # If anything goes wrong, fall back to previous behavior.
        pass


    # --- V2 MODE CONTROL: optional horizontal flip for FRONT-VIEW apparel ---
    # Some UIs express placement in 'human left/right' space; image space is mirrored for front-facing people.
    # flip_x can be passed from frontend (preferred). If not provided, we auto-flip for print/embroidery modes (can be disabled via PRINTS_AUTO_FLIP_X=0).
    try:
        mode_norm = _normalize_mode(getattr(inp, 'mode', ''))
        flip_x = getattr(inp, 'flip_x', None)
        coord_space = (getattr(inp, 'coord_space', None) or '').strip().lower()
        want_flip = (flip_x is True) or (coord_space in ('human','person'))
        # Auto-flip is *opt-in* (env) and intended only for 'human-space' UIs.
        if (flip_x is None and not coord_space) and _auto_flip_x_enabled() and mode_norm in ('vyshyvanka_panel','embroidery_panel'):
            want_flip = True
        if want_flip:
            quad = _flip_quad_x(quad, base.width)
    except Exception:
        pass


    # Background cutout for non-transparent logo assets (JPG screenshots)
    design = _cutout_guard(design)

    profile = _get_print_profile(inp.mode)
    profile_text = (
        f"\\n\\nPRINT TYPE PROFILE ({str(inp.mode).strip().upper() if inp.mode else 'DEFAULT'}):\\n"
        f"- description: {profile.get('description')}\\n"
        f"- thickness: {profile.get('thickness')}\\n"
        f"- gloss: {profile.get('gloss')}\\n"
        f"- deformation_follow: {profile.get('deformation_follow')}\\n"
        f"- edge_softness: {profile.get('edge_softness')}\\n"
        f"- edge_clearance_px: {profile.get('edge_clearance_px')}\\n"
        f"- remove_underlying: {profile.get('remove_underlying')}\\n"
    )


    # Preserve small text: supersample when target quad is tiny
    design = _maybe_supersample_design(design, quad)

    # Preserve logo proportions: pad to target quad aspect (avoid squeezing text)
    qw, qh = _quad_size(quad)
    design = _pad_design_to_aspect(design, qw / qh)

    if base.width < 64 or base.height < 64:
        raise ValueError("Слишком маленькое фото изделия")
    if design.width < 8 or design.height < 8:
        raise ValueError("Слишком маленький дизайн/логотип")

    if len(quad) != 4:
        raise ValueError("quad должен содержать 4 точки")

    # source rectangle in design coords
    src = [(0.0, 0.0), (design.width, 0.0), (design.width, design.height), (0.0, design.height)]
    dst = [(float(x), float(y)) for (x, y) in quad]

    coeffs = _perspective_coeffs(src, dst)

    # Warp design into base size
    warped = _transform_rgba_perspective_premultiplied(design, coeffs, (base.width, base.height))

    # Style (print/embroidery) and mild contrast to keep readability
    styled = _apply_style(warped, inp.mode)

    # Optional occlusion: if quad spills into plain background, clip design to foreground (garment/body)
    occ = _foreground_mask_by_bg(base, dst)
    xs = [p[0] for p in dst]
    ys = [p[1] for p in dst]
    bx0 = max(0, int(min(xs))); by0 = max(0, int(min(ys)))
    bx1 = min(base.width - 1, int(max(xs))); by1 = min(base.height - 1, int(max(ys)))
    occ_crop = occ.crop((bx0, by0, bx1 + 1, by1 + 1))

    # foreground ratio (0..1); if too low -> it's mostly background, so don't clip
    try:
        stat = ImageStat.Stat(occ_crop)
        fg_ratio = (stat.mean[0] / 255.0)
    except Exception:
        fg_ratio = 1.0
    use_occ = fg_ratio >= 0.18

    # Fabric-aware micro deformation + shading (keeps typography crisp but integrates with fabric)
    try:
        mode_norm2 = _normalize_mode(getattr(inp, "mode", ""))
    except Exception:
        mode_norm2 = ""
    styled = _fabric_deform_and_shade(styled, base, (bx0, by0, bx1 + 1, by1 + 1), mode_norm2)


    # Composite over base with "overlay that respects fabric": multiply alpha slightly by base luminance
    base_l = ImageOps.grayscale(base).convert("L")
    mod = ImageChops.multiply(
        styled.split()[-1],
        base_l.point(lambda x: int(190 + (x / 255) * 65))  # 190..255 (a bit less darkening than before)
    )
    if use_occ:
        mod = ImageChops.multiply(mod, occ)
    styled.putalpha(mod)

    # Local sharpening to keep small text readable (affects only the placed design)
    try:
        a2 = styled.split()[-1]
        sharp = styled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=170, threshold=3))
        sharp.putalpha(a2)
        styled = sharp
    except Exception:
        pass


    # --- center seam / placket mini-gap (deterministic) ---
    try:
        if str(os.getenv('PRINTS_CENTER_SEAM_GAP_ENABLE','1')).strip().lower() not in ('0','false','no'):
            # Use quad center if provided, else image center.
            cx = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0) if 'pts' in locals() else (base.width // 2)
            gap_px = int(float(os.getenv('PRINTS_CENTER_SEAM_GAP_PX', '0') or '0'))
            if gap_px <= 0:
                gap_px = max(8, int(base.width * float(os.getenv('PRINTS_CENTER_SEAM_GAP_RATIO','0.012'))))
            feather = int(float(os.getenv('PRINTS_CENTER_SEAM_FEATHER_PX','1') or '1'))
            x0 = max(0, cx - gap_px // 2)
            x1 = min(base.width, cx + (gap_px - gap_px // 2))
            if x1 > x0:
                strip = Image.new('L', (base.width, base.height), 0)
                # paint strip only where design exists to avoid cutting empty area
                a = styled.split()[-1]
                # build a soft strip mask
                strip_np = np.zeros((base.height, base.width), dtype=np.uint8)
                strip_np[:, x0:x1] = 255
                strip = Image.fromarray(strip_np, 'L')
                if feather > 0:
                    strip = strip.filter(ImageFilter.GaussianBlur(radius=feather))
                # apply: alpha = alpha * (1 - strip)
                inv = ImageChops.invert(strip)
                a2 = ImageChops.multiply(a, inv)
                styled.putalpha(a2)
    except Exception:
        pass

    
    # --- quilt / seam micro-cuts (deterministic): interrupt print along strong near-horizontal seams (puffer baffles) ---
    try:
        if str(os.getenv('PRINTS_SEAM_MICROCUT_ENABLE','1')).strip().lower() not in ('0','false','no'):
            # Work in full image coords but only where design alpha exists
            a = np.array(styled.split()[-1], dtype=np.uint8)
            if a.max() > 0:
                gray = np.array(ImageOps.grayscale(base), dtype=np.float32) / 255.0
                # emphasize horizontal edges: vertical gradient highlights near-horizontal lines
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                e = np.abs(gy)
                # threshold based on edge energy within printed area
                mask_print = (a > 12)
                vals = e[mask_print]
                if vals.size > 200:
                    thr = float(os.getenv('PRINTS_SEAM_MICROCUT_THR','0')) or float(np.quantile(vals, 0.92))
                    seam = (e >= thr).astype(np.uint8) * 255
                    # connect horizontal segments
                    kx = int(float(os.getenv('PRINTS_SEAM_MICROCUT_KX','17')))
                    ky = int(float(os.getenv('PRINTS_SEAM_MICROCUT_KY','3')))
                    kx = max(5, min(kx, 51)); ky = max(1, min(ky, 9))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                    seam = cv2.morphologyEx(seam, cv2.MORPH_CLOSE, kernel, iterations=1)
                    # keep only within printed area (expand a little)
                    mp = cv2.dilate(mask_print.astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
                    seam = cv2.bitwise_and(seam, mp)
                    # widen cut slightly
                    wpx = int(float(os.getenv('PRINTS_SEAM_MICROCUT_WIDTH_PX','2')))
                    wpx = max(1, min(wpx, 6))
                    seam = cv2.dilate(seam, cv2.getStructuringElement(cv2.MORPH_RECT,(wpx, wpx)), iterations=1)
                    # apply as hard cut (alpha -> 0) but only where alpha exists
                    seam_mask = (seam > 0) & (a > 12)
                    if seam_mask.any():
                        a2 = a.copy()
                        a2[seam_mask] = 0
                        styled.putalpha(Image.fromarray(a2, 'L'))
    except Exception:
        pass

# --- fabric shading (no geometry distortion): imprint follows wrinkles via light/shadow ---
    try:
        if str(os.getenv('PRINTS_FABRIC_SHADE','1')).strip().lower() not in ('0','false','no'):
            a = np.array(styled.split()[-1], dtype=np.float32) / 255.0
            if a.max() > 0.01:
                bl = np.array(ImageOps.grayscale(base), dtype=np.float32) / 255.0
                gain = float(os.getenv('PRINTS_FABRIC_SHADE_GAIN','0.22'))
                # shade factor around 1.0, driven by base luminance deviation
                factor = 1.0 + (bl - bl.mean()) * gain
                factor = np.clip(factor, 0.75, 1.25)
                rgb = np.array(styled.convert('RGB'), dtype=np.float32)
                rgb = rgb * factor[..., None]
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                new_rgb = Image.fromarray(rgb, 'RGB')
                new_rgba = new_rgb.convert('RGBA')
                new_rgba.putalpha(Image.fromarray((a*255).astype(np.uint8), 'L'))
                styled = new_rgba
    except Exception:
        pass

    # --- premultiply to kill white/gray halos on transparent edges (keeps black logos truly black without white outline) ---
    try:
        arr = np.array(styled.convert('RGBA'), dtype=np.float32)
        aa = arr[..., 3:4] / 255.0
        arr[..., :3] = arr[..., :3] * aa
        styled = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA')
    except Exception:
        pass

    out = Image.alpha_composite(base, styled)

    result_url = _save_png(out)
    thumb_url = _save_png(_make_thumb(out, 420))

    return {
        "resultUrl": result_url,
        "thumbUrl": thumb_url,
        "w": out.width,
        "h": out.height,
        "mode": inp.mode,
    }



# -----------------------------
# AI APPLY (marker replacement)
# -----------------------------


def _pil_to_inline(pil_img: Image.Image, mime: str = "image/png") -> dict:
    import base64
    from io import BytesIO
    buf = BytesIO()
    fmt = "PNG" if mime == "image/png" else "JPEG"
    if fmt == "JPEG":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inlineData": {"mimeType": mime, "data": data}}

def _extract_first_image_b64(resp_json: dict) -> str:
    """Return first image as base64 string from a Gemini generateContent response.

    Gemini may use slightly different key casing depending on SDK/REST wrappers
    (inlineData vs inline_data). We support the common variants.
    """
    resp_json = resp_json or {}
    for cand in resp_json.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            # Most common (REST v1beta): {"inlineData":{"mimeType":"image/png","data":"..."}}
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                return inline["data"]
            # Some wrappers may embed bytes directly under "data"
            if part.get("data") and isinstance(part.get("data"), str):
                return part["data"]
            # fileData can point to a URI; we do NOT fetch here (engine expects inline images).
            # But keep a hint in error if only fileData is present.
    raise RuntimeError("Gemini response has no inline image data")

def _b64_to_bytes(b64: str) -> bytes:
    import base64
    return base64.b64decode(b64)

def _draw_marker(base: Image.Image, quad_px: list[tuple[float, float]]) -> Image.Image:
    """Draw ONLY an outline frame (no fill) so the model can see the real fabric/seams under the guide."""
    from PIL import ImageDraw
    img = base.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Outline-only guide (stroke). No fill — keep fabric visible.
    outline = (235, 60, 60)  # red-ish, high contrast
    pts = list(quad_px) + [quad_px[0]]
    try:
        draw.line(pts, fill=outline, width=3)
    except TypeError:
        # Older Pillow may not support width for line in some modes; fallback to 1px polygon outline.
        draw.polygon(quad_px, outline=outline)

    return img

def _make_warp_driver(base_overlay_rgba: Image.Image, alpha_floor: float = 0.68, blur_px: float = 1.2) -> Image.Image:
    """Build a 'warp driver' reference image to help Gemini bend/attach the print uniformly.

    Problem: soft/pastel/low-opacity parts of the design sometimes don't get bent with the fabric (puffer/balloon),
    while saturated parts do. We provide an extra reference where ANY non-zero alpha is raised to a minimum floor.
    This image is NOT the visible print; it is only a deformation/attachment guide.
    """
    ov = base_overlay_rgba.convert("RGBA")
    alpha = ov.split()[3]
    # Make alpha more even across weak areas.
    alpha = ImageOps.autocontrast(alpha)
    if blur_px and blur_px > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(float(blur_px)))

    floor = int(max(0.0, min(1.0, float(alpha_floor))) * 255)
    # Raise any non-zero alpha to at least floor.
    alpha = alpha.point(lambda p: 0 if p <= 0 else (floor if p < floor else p))

    driver = Image.new("RGBA", ov.size, (255, 255, 255, 0))
    driver.putalpha(alpha)
    return driver
def _make_ink_carrier_from_overlay(overlay_rgba: Image.Image, alpha_strength: int = 72, dilate_px: int = 5, blur_px: float = 2.0) -> Image.Image:
    """Build a low-opacity 'ink carrier' reference for sparse/typographic designs.

    Purpose: help Gemini treat thin strokes / separated letters as ONE physical print film so curvature is applied
    continuously across the whole design block (Tommy vs TNF behavior).
    This is NOT the visible print; it's a deformation cue only and must not appear in the final.
    """
    ov = overlay_rgba.convert("RGBA")
    a = ov.split()[-1]
    # Expand a bit so gaps between strokes are still considered part of the same print block.
    if dilate_px and dilate_px > 0:
        a = a.filter(ImageFilter.MaxFilter(int(dilate_px)))
    if blur_px and blur_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(float(blur_px)))
    # Normalize and set a medium alpha so the model "sees" a continuous print mass.
    a = ImageOps.autocontrast(a)
    strength = int(max(0, min(255, int(alpha_strength))))
    a = a.point(lambda p: 0 if p <= 0 else (strength if p < strength else p))
    carrier = Image.new("RGBA", ov.size, (255, 255, 255, 0))
    carrier.putalpha(a)
    return carrier
def _alpha_bbox_and_fill_ratio(overlay_rgba: Image.Image) -> tuple:
    """Return (bbox, fill_ratio) where fill_ratio is alpha_coverage / bbox_area.
    Used to detect sparse/typographic designs (Tommy) vs dense patches (TNF).
    """
    ov = overlay_rgba.convert("RGBA")
    a = ov.split()[-1]
    # bbox of any non-zero alpha
    bbox = a.getbbox()
    if not bbox:
        return None, 0.0
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    # approximate coverage using numpy if available; fall back to histogram
    try:
        import numpy as _np
        arr = _np.array(a.crop(bbox), dtype=_np.uint8)
        coverage = float(_np.count_nonzero(arr)) / float(arr.size)
    except Exception:
        hist = a.crop(bbox).histogram()
        nonzero = sum(hist[1:])
        coverage = float(nonzero) / float(w*h)
    return bbox, float(coverage)

def _make_ink_carrier_bbox(overlay_rgba: Image.Image, alpha_strength: int = 96, margin_pct: float = 0.10, blur_px: float = 6.0) -> Image.Image:
    """Build a soft filled carrier over the overlay bbox (with margin).
    Helps sparse typography behave like one continuous print block.
    """
    ov = overlay_rgba.convert("RGBA")
    a = ov.split()[-1]
    bbox = a.getbbox()
    if not bbox:
        # fallback: empty carrier
        return Image.new("RGBA", ov.size, (255, 255, 255, 0))
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    mx = int(max(1, bw * float(margin_pct)))
    my = int(max(1, bh * float(margin_pct)))
    x0 = max(0, x0 - mx); y0 = max(0, y0 - my)
    x1 = min(ov.size[0], x1 + mx); y1 = min(ov.size[1], y1 + my)
    strength = int(max(0, min(255, int(alpha_strength))))
    mask = Image.new("L", ov.size, 0)
    # filled rectangle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=strength)
    if blur_px and blur_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(blur_px)))
    carrier = Image.new("RGBA", ov.size, (255, 255, 255, 0))
    carrier.putalpha(mask)
    return carrier



def _make_relief_map_from_base(base_rgb: Image.Image, blur_px: float = 8.0, contrast: float = 1.9) -> Image.Image:
    """Create a fabric relief/height reference from the base image.

    Key requirement for our prints: the map must be *garment-driven*, not background-driven.
    Many product photos have a near-white background; naive autocontrast will squash the garment
    range (especially for black jackets), making the displacement useless.

    Strategy:
    - Build a quick garment mask (background ~white).
    - Autocontrast / enhance inside garment bbox only.
    - Blend low-frequency shading + edge magnitude (seams/baffles) so dark garments still have relief.
    - Return RGB grayscale (some models respond better to RGB).
    """
    rgb = base_rgb.convert("RGB")
    g_full = rgb.convert("L")
    arr = np.asarray(g_full, dtype=np.uint8)

    # 1) garment mask: anything significantly darker than near-white background
    thr = 245
    mask = (arr < thr)

    # If mask too small (non-white background / studio), use adaptive percentile threshold.
    if mask.mean() < 0.03:
        p = int(np.percentile(arr, 98))
        thr = max(220, min(250, p - 5))
        mask = (arr < thr)

    # Bbox of garment area
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = int(max(12, 0.02 * max(arr.shape)))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(arr.shape[1] - 1, x1 + pad); y1 = min(arr.shape[0] - 1, y1 + pad)
        crop_box = (x0, y0, x1 + 1, y1 + 1)
    else:
        crop_box = (0, 0, g_full.size[0], g_full.size[1])

    g = g_full.crop(crop_box)

    # 2) low-frequency shading
    if blur_px and blur_px > 0:
        g_shade = g.filter(ImageFilter.GaussianBlur(float(blur_px)))
    else:
        g_shade = g

    g_shade = ImageOps.autocontrast(g_shade)
    if contrast and contrast != 1.0:
        g_shade = ImageEnhance.Contrast(g_shade).enhance(float(contrast))

    # 3) edge magnitude for seams/baffles (important on dark garments)
    g_small = g.resize((max(64, g.size[0] // 2), max(64, g.size[1] // 2)), resample=Image.BILINEAR).convert("L")
    g_arr = np.asarray(g_small, dtype=np.float32) / 255.0

    # Sobel gradients
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    def _conv(a, k):
        # small & simple convolution, valid for our sizes
        h, w = a.shape
        out = np.zeros_like(a)
        ap = np.pad(a, ((1, 1), (1, 1)), mode='edge')
        for yy in range(h):
            for xx in range(w):
                out[yy, xx] = float((ap[yy:yy+3, xx:xx+3] * k).sum())
        return out

    gx = _conv(g_arr, kx)
    gy = _conv(g_arr, ky)
    mag = np.sqrt(gx * gx + gy * gy)

    # normalize edges
    mag = mag - mag.min()
    denom = (mag.max() + 1e-6)
    mag = mag / denom
    # boost mid edges, suppress noise
    mag = np.clip(mag, 0.0, 1.0) ** 0.6

    edges = Image.fromarray((mag * 255.0).astype(np.uint8), mode="L")
    edges = edges.resize(g.size, resample=Image.BILINEAR)
    edges = edges.filter(ImageFilter.GaussianBlur(1.6))
    edges = ImageOps.autocontrast(edges)

    # 4) blend shading + edges (more edges weight for dark garments)
    try:
        mean_l = float(np.asarray(g, dtype=np.float32).mean())
    except Exception:
        mean_l = 140.0
    w_edges = 0.55 if mean_l < 120 else 0.35
    w_shade = 1.0 - w_edges

    shade_arr = np.asarray(g_shade, dtype=np.float32) / 255.0
    edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
    rel = np.clip(w_shade * shade_arr + w_edges * edge_arr, 0.0, 1.0)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-6)
    rel_img = Image.fromarray((rel * 255.0).astype(np.uint8), mode="L")

    # Paste back into full-size map (background neutral mid-gray so it doesn't dominate)
    full_rel = Image.new("L", g_full.size, color=128)
    full_rel.paste(rel_img, crop_box)

    return Image.merge("RGB", (full_rel, full_rel, full_rel))




def _make_garment_anchors_map(base_rgb: Image.Image, mask_l: Image.Image, blur_px: float = 1.0) -> Image.Image:
    """Build a seam/fold 'anchors' map from the BASE image inside the edit mask.

    Purpose:
    - Give Gemini a strong hint where seams, pocket edges, zipper lines and folds are,
      so it can bend the print AND keep seams visible instead of avoiding them.

    Output:
    - RGB image: bright edges on dark background.
    """
    try:
        g = base_rgb.convert("L")
        # Emphasize local contrast (helps on black garments)
        g = ImageOps.autocontrast(g, cutoff=2)
        # Edge map
        e = g.filter(ImageFilter.FIND_EDGES)
        e = e.filter(ImageFilter.GaussianBlur(radius=blur_px))
        e = ImageEnhance.Contrast(e).enhance(2.4)
        e = ImageEnhance.Brightness(e).enhance(1.15)

        # Apply mask: keep anchors only inside mask
        m = mask_l.convert("L")
        if m.size != e.size:
            m = m.resize(e.size, Image.BILINEAR)
        e = Image.composite(e, Image.new("L", e.size, 0), m)

        return Image.merge("RGB", (e, e, e))
    except Exception:
        # Fallback: just return base (won't break the pipeline)
        return base_rgb.convert("RGB")


def _overlay_fill_ratio(overlay_rgba: Image.Image) -> float:
    try:
        a = np.array(overlay_rgba.split()[-1], dtype=np.uint8)
        return float((a > 16).sum()) / float(a.size)
    except Exception:
        return 1.0

def _prewarp_overlay_with_relief(overlay_rgba: Image.Image, relief_gray: Image.Image, strength_px: float = 10.0) -> Image.Image:
    """Pre-warp the overlay using a displacement field derived from a relief/height map.

    Goal: make curvature deterministic (avoid 'flat sticker') while preserving seams/zipper stability.
    - Uses relief gradients (fabric flow) + a gentle radial bulge term (helps convex surfaces like chest/knee).
    - Suppresses warping near strong seam-like vertical edges to avoid 'text breaking' on zippers/plackets.
    - Warps ONLY inside overlay alpha area; outside stays transparent.
    """
    if strength_px <= 0:
        return overlay_rgba
    try:
        ov = overlay_rgba.convert("RGBA")
        W, H = ov.size

        # relief array 0..1
        rel = relief_gray.resize((W, H), resample=Image.BILINEAR).convert("L")
        rel_arr = np.asarray(rel, dtype=np.float32) / 255.0

        ov_arr = np.asarray(ov, dtype=np.float32)
        a = ov_arr[..., 3]
        alpha_mask = (a > 8).astype(np.float32)

        # Edge rigidity: preserve sharp corners/letter strokes (avoid 'kinky' letters after warp)
        # We reduce displacement near strong edges of the overlay alpha (letters/borders).
        try:
            if str(os.getenv("PRINTS_EDGE_RIGIDITY", "1")).strip() not in ("0","false","False","no","NO"):
                agy, agx = np.gradient(alpha_mask)
                amag = np.sqrt(agx * agx + agy * agy)
                k_edge = float(os.getenv("PRINTS_EDGE_RIGIDITY_K", "18.0") or "18.0")
                edge = np.clip(amag * k_edge, 0.0, 1.0)
                edge_mult = float(os.getenv("PRINTS_EDGE_RIGIDITY_MULT", "0.85") or "0.85")
                edge_weight = 1.0 - np.clip(edge * edge_mult, 0.0, 0.95)
            else:
                edge_weight = 1.0
        except Exception:
            edge_weight = 1.0


        # If overlay is basically empty, bail out
        if float(alpha_mask.mean()) < 1e-4:
            return overlay_rgba

        # Smooth relief a bit (it is already blurred, but this stabilizes gradients)
        # (avoid importing cv2; use numpy-friendly separable blur)
        try:
            # small box blur
            k = 3
            pad = k // 2
            rel_p = np.pad(rel_arr, ((pad, pad), (pad, pad)), mode="edge")
            tmp = np.zeros_like(rel_arr)
            for dy in range(k):
                tmp += rel_p[dy:dy+H, pad:pad+W]
            tmp /= float(k)
            rel_p2 = np.pad(tmp, ((pad, pad), (pad, pad)), mode="edge")
            rel_s = np.zeros_like(rel_arr)
            for dx in range(k):
                rel_s += rel_p2[pad:pad+H, dx:dx+W]
            rel_s /= float(k)
            rel_arr = rel_s
        except Exception:
            pass

        # Gradients (fabric flow)
        gy, gx = np.gradient(rel_arr)
        mag = np.sqrt(gx * gx + gy * gy)

        # Seam/zipper suppression mask (strong mostly-vertical edges)
        seam = ((np.abs(gx) > (np.abs(gy) * 1.6 + 1e-6)) & (np.abs(gx) > 0.010) & (mag > 0.014)).astype(np.float32)
        # Spread & soften the seam mask (cheap blur)
        try:
            k2 = 9
            pad2 = k2 // 2
            s_p = np.pad(seam, ((pad2, pad2), (pad2, pad2)), mode="edge")
            s_blur = np.zeros_like(seam)
            for dy in range(k2):
                s_blur += s_p[dy:dy+H, pad2:pad2+W]
            s_blur /= float(k2)
            s_p2 = np.pad(s_blur, ((pad2, pad2), (pad2, pad2)), mode="edge")
            s2 = np.zeros_like(seam)
            for dx in range(k2):
                s2 += s_p2[pad2:pad2+H, dx:dx+W]
            s2 /= float(k2)
            seam = np.clip(s2, 0.0, 1.0)
        except Exception:
            seam = np.clip(seam, 0.0, 1.0)

        # Weight: 1 away from seams, 0 near seams
        seam_weight = 1.0 - np.clip(seam * 1.25, 0.0, 1.0)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        # Optionally create a tiny "no-ink" gap along the strongest vertical seam/zipper
        # to prevent letters from visibly crossing a zipper/placket.
        try:
            if str(os.getenv("PRINTS_SEAM_GAP_ENABLE", "1")).strip() not in ("0","false","False","no","NO"):
                gap_px = int(float(os.getenv("PRINTS_SEAM_GAP_PX", "3") or "3"))
                gap_px = max(0, min(24, gap_px))
                if gap_px > 0:
                    # Column-wise seam strength inside the overlay region
                    col_strength = (seam * alpha_mask).sum(axis=0) / (alpha_mask.sum(axis=0) + 1e-6)
                    seam_x = int(np.argmax(col_strength))
                    # Only apply if the seam is actually strong
                    if float(col_strength[seam_x]) > 0.22:
                        # Reduce warp right at seam and also cut alpha a few px to avoid 'broken text'
                        seam_weight *= (1.0 - np.clip(seam[:, seam_x:seam_x+1] * 0.9, 0.0, 0.9))
                        seam_strip = (np.abs(xs - float(seam_x)) <= float(gap_px)).astype(np.float32)
                        alpha_mask *= (1.0 - seam_strip)

                    # --- Pocket zipper / diagonal seam gap (DTF): also cut a tiny "no-ink" stripe along strong diagonal zippers
                    try:
                        # Diagonal edge mask (rough): both gradients present, decent magnitude
                        diag = ((np.abs(gx) > 0.010) & (np.abs(gy) > 0.010) & (mag > 0.016)).astype(np.float32)
                        diag_w = diag * alpha_mask
                        if float(diag_w.sum()) > 50.0:
                            # Two diagonal orientations: +45 (x+y const) and -45 (x-y const)
                            t1 = (xs + ys).astype(np.int32)
                            t2 = (xs - ys + (H - 1)).astype(np.int32)  # shift to be >=0
                            # Accumulate weights only where overlay exists
                            w = diag_w.astype(np.float32)
                            # bincount expects 1D
                            b1 = np.bincount(t1.ravel(), weights=w.ravel(), minlength=int(W + H))
                            b2 = np.bincount(t2.ravel(), weights=w.ravel(), minlength=int(W + H))
                            c1 = int(np.argmax(b1))
                            c2 = int(np.argmax(b2))
                            # Normalize by how much overlay mass exists on each diagonal bin
                            m1 = np.bincount(t1.ravel(), weights=alpha_mask.ravel(), minlength=int(W + H)) + 1e-6
                            m2 = np.bincount(t2.ravel(), weights=alpha_mask.ravel(), minlength=int(W + H)) + 1e-6
                            s1 = float(b1[c1] / m1[c1])
                            s2 = float(b2[c2] / m2[c2])
                            # Only cut if a strong diagonal seam is detected
                            thr = float(os.getenv("PRINTS_POCKET_SEAM_THR", "0.20") or "0.20")
                            if s1 > thr:
                                # distance to line x+y=c1 is |x+y-c1|/sqrt(2)
                                strip = (np.abs((xs + ys) - float(c1)) <= float(gap_px) * 1.25).astype(np.float32)
                                alpha_mask *= (1.0 - strip)
                            if s2 > thr:
                                # line x-y = c2-(H-1)
                                strip = (np.abs((xs - ys + float(H - 1)) - float(c2)) <= float(gap_px) * 1.25).astype(np.float32)
                                alpha_mask *= (1.0 - strip)
                    except Exception:
                        pass
        except Exception:
            pass

        # Base displacement from gradients (flow)
        flow_strength = float(strength_px) * 0.75
        dx = gx * flow_strength
        dy = gy * flow_strength

        # Gentle radial bulge term (helps convex surfaces)
        # Center of mass of the overlay alpha region
        wsum = float(alpha_mask.sum())
        cx = float((xs * alpha_mask).sum() / max(1.0, wsum))
        cy = float((ys * alpha_mask).sum() / max(1.0, wsum))

        rel_center = rel_arr - float((rel_arr * alpha_mask).sum() / max(1.0, wsum))
        # normalize so extremes don't explode
        rel_center = np.clip(rel_center, -0.35, 0.35)

        bulge_strength = float(strength_px) * 0.55
        nx = (xs - cx) / max(1.0, (W * 0.5))
        ny = (ys - cy) / max(1.0, (H * 0.5))
        dx += nx * rel_center * bulge_strength * (W * 0.08)
        dy += ny * rel_center * bulge_strength * (H * 0.08)

        # Clamp displacement to avoid tearing
        max_disp = float(strength_px) * 1.15
        dx = np.clip(dx, -max_disp, max_disp)
        dy = np.clip(dy, -max_disp, max_disp)

        # Apply seam suppression + alpha gating
        weight = seam_weight * alpha_mask * edge_weight
        dx *= weight
        dy *= weight

        # Coordinates remap
        src_x = np.clip(xs + dx, 0.0, float(W - 1))
        src_y = np.clip(ys + dy, 0.0, float(H - 1))

        # Bilinear sampling
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)

        wa = (x1 - src_x) * (y1 - src_y)
        wb = (src_x - x0) * (y1 - src_y)
        wc = (x1 - src_x) * (src_y - y0)
        wd = (src_x - x0) * (src_y - y0)

        Ia = ov_arr[y0, x0]
        Ib = ov_arr[y0, x1]
        Ic = ov_arr[y1, x0]
        Id = ov_arr[y1, x1]
        out = (Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None])

        # keep outside alpha transparent
        out *= alpha_mask[..., None]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")
    except Exception:
        return overlay_rgba


def _sharpen_print_area(out_rgb: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    """Restore crisp edges in the print area after Gemini edits.
    Applies a mild unsharp mask only inside the overlay alpha.
    """
    try:
        out_rgb = out_rgb.convert("RGB")
        ov = overlay_rgba.convert("RGBA").resize(out_rgb.size, Image.NEAREST)
        a = ov.split()[-1]
        # inner mask to avoid sharpening halos at the outer border
        inner = a.filter(ImageFilter.MinFilter(3)).point(lambda p: 255 if p > 32 else 0)
        if ImageStat.Stat(inner).sum[0] < 2000:
            return out_rgb
        sharp = out_rgb.filter(ImageFilter.UnsharpMask(radius=1.1, percent=165, threshold=3))
        out_rgb = Image.composite(sharp, out_rgb, inner)
        return out_rgb
    except Exception:
        return out_rgb
def _flatness_failure(out_rgb: Image.Image, base_rgb: Image.Image, overlay_rgba: Image.Image) -> bool:
    """Heuristic: detect when the applied print looks too 'flat' (little fabric shading captured).

    We compare luminance variability inside the print area between base and output.
    If output variance is far lower than base variance, it's likely a flat overlay (not conforming to folds).
    """
    try:
        out_rgb = out_rgb.convert("RGB")
        base_rgb = base_rgb.convert("RGB").resize(out_rgb.size, Image.BILINEAR)
        ov = overlay_rgba.convert("RGBA").resize(out_rgb.size, Image.BILINEAR)
        a = ov.split()[-1]
        # Use inner mask to focus on print body, not edges.
        a_inner = a.filter(ImageFilter.MinFilter(5))
        m = np.array(a_inner, dtype=np.uint8) > 24
        if m.sum() < 500:  # too small, skip
            return False
        out_l = np.array(out_rgb.convert("L"), dtype=np.float32)
        base_l = np.array(base_rgb.convert("L"), dtype=np.float32)
        out_std = float(out_l[m].std())
        base_std = float(base_l[m].std())
        # If output has much less shading variation than base, it's suspiciously flat.
        if base_std > 3.0 and out_std < base_std * 0.45:
            return True
    except Exception:
        return False
    return False





def apply_design_ai(inp: ApplyIn) -> dict:
    """
    AI REPLACE ENGINE (Gemini image edit):
    Replace ONLY the projected logo pixels and treat the design as a SOLID FILLED PRINT (never outline) (mask derived from logo alpha) on the base image.
    - base_url: main photo
    - design_url: logo (PNG preferred; if no alpha -> best-effort cutout)
    - quad: 4 points in PIXELS (x,y) describing placement rectangle on base image
    Returns: resultUrl, thumbUrl, w, h, mode (+ ai/model).
    """
    run_id = uuid.uuid4().hex[:8]
    import io
    import base64
    from PIL import Image, ImageFilter
    from app.engine.engine_init import load_engine_config
    from app.engine.gemini_rest import post_generate_content
    from app.engine.media_io import bytes_to_b64

    cfg = load_engine_config()
    if not (cfg.api_key or "").strip():
        raise RuntimeError("GEMINI_API_KEY is empty.")

    # --- resolve mode + profile early (needed by seam/placket logic) ---
    mode = None
    for attr in ('mode','printType','print_type'):
        if hasattr(inp, attr) and getattr(inp, attr):
            mode = getattr(inp, attr)
            break
    if not mode:
        mode = getattr(inp, 'mode', '') or ''
    try:
        profile = _get_print_profile(mode)
    except Exception:
        profile = {'description':'DEFAULT (fallback)','surface':'fabric'}


    base_img = _load_image_from_url(inp.base_url).convert("RGBA")
    design_img = _load_image_from_url(inp.design_url).convert("RGBA")
    # IMPORTANT: Frontend preview can remove background, but Gemini must receive the CUTOUT design too.
    # If the uploaded "logo" is actually a JPG/screenshot with a matte, this removes the rectangle.
    try:
        design_img = _cutout_guard(design_img)
    except Exception:
        # never fail the request because of cutout; fallback to original
        pass

    # --- de-halo / matte cleanup (kills faint background haze around edges) ---
    if os.getenv("PRINTS_DEHALO_ENABLE", "0") != "0":
        try:
            thr = int(os.getenv("PRINTS_DEHALO_ALPHA_THRESHOLD", "210"))
        except Exception:
            thr = 210
        design_img = _dehalo_alpha_hard(design_img, thr)


    
    # Optional arc warp is handled later (after quad scaling) to match frontend preview.

# --- ensure solid alpha (Variant 2: deterministic mask) ---
    # If design has no real transparency (e.g. JPG on black/white),
    # build a binary alpha mask BEFORE Gemini so the model never guesses contours.
    
    # NOTE: we do NOT invent alpha here. We do deterministic backend cutout later via _cutout_guard().


    # --- CANON: upscale base to at least 2K long side (and scale quad accordingly) ---
    orig_w, orig_h = base_img.size
    quad_px = _quad_to_pixels(getattr(inp, "quad", None), orig_w, orig_h)
    # Optional horizontal flip if frontend coordinate space is mirrored
    flip_x_opt = None
    try:
        if isinstance(getattr(inp, 'options', None), dict):
            flip_x_opt = inp.options.get('flipX')
    except Exception:
        flip_x_opt = None
    if flip_x_opt is True or _force_flip_x_enabled() or (flip_x_opt is None and _auto_flip_x_enabled()):
        quad_px = _flip_quad_x(quad_px, W)
    base_img, sx, sy = _ensure_min_long_side(base_img, 2048)
    quad_px = _scale_quad(quad_px, sx, sy)

    # --- Preview parity: render design into quad bbox with the same safe inset as frontend ---
    try:
        xs = [p[0] for p in quad_px]
        ys = [p[1] for p in quad_px]
        bx0 = float(min(xs)); by0 = float(min(ys))
        bx1 = float(max(xs)); by1 = float(max(ys))
        dw = max(1, int(round(bx1 - bx0)))
        dh = max(1, int(round(by1 - by0)))

        warp = getattr(inp, 'warp', None) or {}
        fit_scale = _compute_fit_scale_for_warp(warp)

        src_w = max(1, int(round(dw * fit_scale)))
        src_h = max(1, int(round(dh * fit_scale)))
        design_scaled = design_img.resize((src_w, src_h), Image.LANCZOS)

        design_rect = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        ox = int(round((dw - src_w) / 2.0))
        oy = int(round((dh - src_h) / 2.0))
        design_rect.alpha_composite(design_scaled, (ox, oy))

        if getattr(inp, 'warp', None):
            design_rect = _arc_warp_design(design_rect, warp)

        design_img = design_rect
    except Exception:
        pass


    # --- V2 MODE CONTROL: optional horizontal flip for FRONT-VIEW apparel ---
    # flip_x (bool) or coord_space='human' can be passed from frontend. If absent, auto-flip for print/embroidery modes (disable via PRINTS_AUTO_FLIP_X=0).
    try:
        mode_norm = _normalize_mode(getattr(inp, 'mode', ''))
        flip_x = getattr(inp, 'flip_x', None)
        coord_space = (getattr(inp, 'coord_space', None) or '').strip().lower()
        want_flip = (flip_x is True) or (coord_space in ('human','person'))
        # Auto-flip is *opt-in* (env) and intended only for 'human-space' UIs.
        if (flip_x is None and not coord_space) and _auto_flip_x_enabled() and mode_norm in ('vyshyvanka_panel','embroidery_panel'):
            want_flip = True
        if want_flip:
            quad_px = _flip_quad_x(quad_px, base_img.width)
    except Exception:
        pass

    W, H = base_img.size


    
    # --- precision helpers ---
    def _poly_area(pts):
        a = 0.0
        for i in range(len(pts)):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[(i + 1) % len(pts)][0]), float(pts[(i + 1) % len(pts)][1])
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    def _bbox_of_pts(pts):
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def _maybe_micro_roi(base_rgba: Image.Image, quad: list[tuple[float, float]]):
        # If the edit region is very small, work in a zoomed ROI for better fidelity.
        area = _poly_area(quad)
        if area <= 0:
            return None
        frac = area / float(W * H)
        if frac >= 0.02:  # >=2% => normal path
            return None

        x0, y0, x1, y1 = _bbox_of_pts(quad)
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        pad = max(bw, bh) * 1.8  # generous padding
        rx0 = int(max(0, math.floor((x0 + x1) / 2 - pad)))
        ry0 = int(max(0, math.floor((y0 + y1) / 2 - pad)))
        rx1 = int(min(W, math.ceil((x0 + x1) / 2 + pad)))
        ry1 = int(min(H, math.ceil((y0 + y1) / 2 + pad)))
        if (rx1 - rx0) < 32 or (ry1 - ry0) < 32:
            return None

        roi = base_rgba.crop((rx0, ry0, rx1, ry1)).convert("RGBA")
        rW, rH = roi.size

        # upscale ROI so the small logo becomes "large enough" for the model
        target_min = 768
        scale = max(1.0, float(target_min) / float(min(rW, rH)))
        if scale > 1.0:
            new_size = (int(round(rW * scale)), int(round(rH * scale)))
            roi_up = roi.resize(new_size, Image.Resampling.LANCZOS)
        else:
            roi_up = roi
            scale = 1.0

        # quad in ROI coords (and upscaled coords)
        quad_roi = [((float(x) - rx0) * scale, (float(y) - ry0) * scale) for (x, y) in quad]
        return {"rx0": rx0, "ry0": ry0, "rx1": rx1, "ry1": ry1, "scale": scale, "roi_up": roi_up, "quad_up": quad_roi}

    
    def _detect_zipper_stripe(roi_rgb: Image.Image, quad: list[tuple[float, float]]):
        # Conservative heuristic (no numpy): detect a strong vertical seam/zipper inside the quad bbox.
        g = roi_rgb.convert("L")

        x0f, y0f, x1f, y1f = _bbox_of_pts(quad)
        x0 = int(max(0, math.floor(x0f)))
        y0 = int(max(0, math.floor(y0f)))
        x1 = int(min(roi_rgb.size[0] - 1, math.ceil(x1f)))
        y1 = int(min(roi_rgb.size[1] - 1, math.ceil(y1f)))
        if x1 - x0 < 20 or y1 - y0 < 40:
            return None

        # pad a bit
        pad = int(max(6, (x1 - x0) * 0.06))
        cx0 = max(0, x0 - pad); cx1 = min(roi_rgb.size[0] - 1, x1 + pad)
        cy0 = max(0, y0 - pad); cy1 = min(roi_rgb.size[1] - 1, y1 + pad)

        crop = g.crop((cx0, cy0, cx1 + 1, cy1 + 1))
        # vertical edge magnitude (simple Sobel-x)
        vx = crop.filter(ImageFilter.Kernel((3, 3), [-1,0,1,-2,0,2,-1,0,1], scale=1))
        vp = vx.load()
        cw, ch = vx.size

        # column energies (mean abs edge)
        col = [0.0] * cw
        for yy in range(ch):
            for xx in range(cw):
                col[xx] += float(vp[xx, yy])
        col = [v / float(ch) for v in col]

        if cw < 10:
            return None

        mean = sum(col) / float(cw)
        var = sum((v - mean) * (v - mean) for v in col) / float(cw)
        std = var ** 0.5
        if std < 1e-6:
            return None

        mx = max(col)
        idx = col.index(mx)
        # strong outlier => zipper/seam
        if mx < mean + std * 2.6:
            return None

        # stripe width depends on quad width (slightly wider than before)
        qw = max(1.0, float(x1 - x0))
        stripe_w = max(6, int(round(qw * 0.10)))  # ~10% of quad width
        # return in roi coords
        return {"x": int(cx0 + idx), "w": int(stripe_w), "y0": int(y0), "y1": int(y1)}


    # Work image/quad (optionally zoomed ROI for tiny placements)
    base_work = base_img
    quad_work = [(float(x), float(y)) for (x, y) in quad_px]
    paste_info = _maybe_micro_roi(base_img, quad_work)
    if paste_info:
        base_work = paste_info["roi_up"]
        quad_work = paste_info["quad_up"]

    wW, wH = base_work.size
    # 1) Ensure logo has usable alpha (remove black/white screenshot backgrounds if needed)
    design_img = _cutout_guard(design_img)

    # Optional UI bend (curve) BEFORE perspective.
    # Supports both legacy slider `inp.curve` and the new 4-handle warp `inp.warp`.
    try:
        warp = getattr(inp, 'warp', None) or {}
        curve_legacy = float(getattr(inp, 'curve', 0.0) or 0.0)

        # New UI: mid-handles produce two curves
        curve_v = float((warp.get('bottom', 0.0) - warp.get('top', 0.0)) or 0.0) * 0.28
        curve_h = float((warp.get('right', 0.0) - warp.get('left', 0.0)) or 0.0) * 0.22

        # Keep backwards compatibility with the legacy "curve" slider (adds to vertical bend).
        curve_v = curve_v + curve_legacy

        # Apply in the same order as the frontend preview: vertical then horizontal.
        design_img = _warp_arc_vertical(design_img, curve_v)
        design_img = _warp_arc_horizontal(design_img, curve_h)
    except Exception:
        pass

    # 2) Project logo into quad using Pillow perspective transform (no OpenCV, no numpy) (no OpenCV, no numpy)
    def _solve_linear(A, B):
        # Gaussian elimination for 8x8
        n = len(B)
        M = [A[i][:] + [B[i]] for i in range(n)]
        for i in range(n):
            # pivot
            pivot = i
            for r in range(i+1, n):
                if abs(M[r][i]) > abs(M[pivot][i]):
                    pivot = r
            if abs(M[pivot][i]) < 1e-12:
                raise ValueError("Singular matrix")
            M[i], M[pivot] = M[pivot], M[i]
            # normalize row
            div = M[i][i]
            for c in range(i, n+1):
                M[i][c] /= div
            # eliminate
            for r in range(n):
                if r == i:
                    continue
                factor = M[r][i]
                if abs(factor) < 1e-12:
                    continue
                for c in range(i, n+1):
                    M[r][c] -= factor * M[i][c]
        return [M[i][n] for i in range(n)]

    def _find_perspective_coeffs(src_pts, dst_pts):
        # Solve for coeffs mapping dst -> src for PIL perspective
        A = []
        B = []
        for (x, y), (u, v) in zip(dst_pts, src_pts):
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            B.append(u)
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            B.append(v)
        return _solve_linear(A, B)

    src = [(0.0, 0.0), (float(design_img.width), 0.0),
           (float(design_img.width), float(design_img.height)), (0.0, float(design_img.height))]
    dst = [(float(quad_work[0][0]), float(quad_work[0][1])),
           (float(quad_work[1][0]), float(quad_work[1][1])),
           (float(quad_work[2][0]), float(quad_work[2][1])),
           (float(quad_work[3][0]), float(quad_work[3][1]))]

    coeffs = _find_perspective_coeffs(src, dst)

    overlay = Image.new("RGBA", (wW, wH), (0, 0, 0, 0))
    warped_logo = design_img.transform((wW, wH), Image.PERSPECTIVE, coeffs, resample=getattr(getattr(Image,'Resampling',Image),'BICUBIC'))
    overlay = Image.alpha_composite(overlay, warped_logo)

    # 3) Build guides for the model (DTF RULEBOOK V1):
    #    - CLEAN BASE (this is the image we edit)
    #    - SEPARATE OUTLINE GUIDE (reference only; must NOT appear in the final)
    #    - MASK for strict edit bounds (white=edit)
    guide_img = _draw_marker(base_work, quad_work)

    # Build edit mask.
    # IMPORTANT: mask MUST match the actual ink area, not the whole quad.
    # Otherwise Gemini may "fill" the quad with a matte/background.
    # We derive the mask from the warped logo alpha (after zipper/seam cuts).
    mask = None
    orig_mask = None

    
    # Preserve a zipper/seam line if it crosses the placement (helps jackets with center zipper).
    # We do this deterministically: cut a vertical stripe out of BOTH the edit mask and the warped logo alpha.
    try:
        stripe = _detect_zipper_stripe(base_work.convert("RGB"), quad_work)
        if stripe:
            sx = int(stripe["x"])
            sw = int(stripe["w"])
            yy0 = int(stripe.get("y0", 0))
            yy1 = int(stripe.get("y1", wH - 1))

            # If the zipper is OPEN (often shows background inside), widen the gap.
            try:
                bg_probe = base_work.convert("RGB").crop((max(0, sx - sw), yy0, min(wW, sx + sw), yy1))
                stat = ImageStat.Stat(bg_probe.convert("L"))
                # low variance + very bright/dark => likely background gap
                if (stat.stddev[0] < 18.0) and (stat.mean[0] > 210.0 or stat.mean[0] < 45.0):
                    sw = int(round(sw * 1.45))
            except Exception:
                pass

            x0 = max(0, sx - sw // 2)
            x1 = min(wW, sx + sw // 2 + 1)

            # Cut only inside the placement bbox (not full image height!)
            cut_h = max(1, yy1 - yy0)
            cut = Image.new("L", (x1 - x0, cut_h), 0)
            # mask is derived later from overlay alpha; no need to cut mask here.

            # also cut overlay alpha so logo naturally splits by zipper
            if overlay.mode != "RGBA":
                overlay = overlay.convert("RGBA")
            r, g, b, a = overlay.split()
            a.paste(Image.new("L", (x1 - x0, cut_h), 0), (x0, yy0))
            overlay = Image.merge("RGBA", (r, g, b, a))
    except Exception:
        pass
    # --- CENTER SEAM GAP (shirt placket / zipper / front opening) ---
    # --- CENTER SEAM GAP (AUTO) ---
    # Auto-cut a vertical no-ink stripe if placement crosses the garment center opening (placket/zipper).
    # This is engine-internal (no UI knob): it forces the print to split into two panels instead of bridging.
        if (str(os.getenv("PRINTS_CENTER_SEAM_GAP_ENABLE", "1")).strip() not in ("0","false","False","no","NO")) and ((profile.get("surface", "fabric") == "fabric") or (_normalize_mode(getattr(inp, "mode", "")) in ("dtf","sublimation","silkscreen","flex","vinyl","rubber","embroidery_panel","vyshyvanka_panel"))):
            rules = _load_garment_rules()
            dflt = (rules or {}).get("defaults", {}) or {}
            gap_px = max(10, int(round(wW * float(dflt.get("center_seam_gap_width_ratio", 0.022)))))

            # bbox of placement
            xs = [p[0] for p in quad_work]
            ys = [p[1] for p in quad_work]
            bx0, bx1 = int(min(xs)), int(max(xs))
            by0, by1 = int(min(ys)), int(max(ys))

            # only if wide/tall enough (avoid affecting tiny chest logos)
            w_ratio = (bx1 - bx0) / max(1.0, float(wW))
            h_ratio = (by1 - by0) / max(1.0, float(wH))
            min_w = float(dflt.get("min_bbox_width_ratio", 0.18))
            min_h = float(dflt.get("min_bbox_height_ratio", 0.12))

            crosses_center = (bx0 < (wW // 2) < bx1)
            only_if_cross = bool(dflt.get("only_if_crosses_center", True))
            if (w_ratio >= min_w) and (h_ratio >= min_h) and ((not only_if_cross) or crosses_center):
                mask, overlay = _apply_center_seam_gap(mask, overlay, gap_px=gap_px, bbox=(bx0, by0, bx1, by1), img_w=wW, img_h=wH)
    except Exception:
        pass





    # --- PREWARP (DTF V2.3): deterministic curvature using a relief/height map ---
    # Typography/sparse logos often end up flat due to model stochasticity.
    # We pre-warp the overlay within its alpha area using a displacement field from the base relief map,
    # then ask Gemini to integrate texture/shading without changing geometry.
    relief_img_pre = None
    try:
        relief_img_pre = _make_relief_map_from_base(base_work.convert("RGB"), blur_px=10.0, contrast=2.1)
        _fr = _overlay_fill_ratio(overlay)

        # Base strength by "print density" (typography/logos need stronger warp)
        _strength = 18.0 if (_fr < 0.18) else (16.0 if (_fr < 0.22) else (14.0 if (_fr < 0.50) else 18.0))

        # If the garment relief is weak (e.g., dark jacket flattened by background, or very smooth fabric),
        # push harder to make curvature visible and consistent.
        try:
            _rel_l = relief_img_pre.convert("L")
            _rel_arr = np.asarray(_rel_l, dtype=np.float32) / 255.0
            _energy = float(_rel_arr.std())
            if _energy < 0.06:
                _strength += 6.0
            elif _energy < 0.085:
                _strength += 3.0
        except Exception:
            pass

        # Optional env controls (safe defaults):
        # - PRINTS_PREWARP_ENABLE=0 disables fabric-aware prewarp
        # - PRINTS_PREWARP_STRENGTH_MULT scales deformation (e.g., 0.8..1.4)
        try:
            if str(os.getenv("PRINTS_PREWARP_ENABLE", "1")).strip() in ("0","false","False","no","NO"):
                raise RuntimeError("prewarp disabled")
            _mult = float(os.getenv("PRINTS_PREWARP_STRENGTH_MULT", "1.0") or "1.0")
            if _mult <= 0:
                raise RuntimeError("prewarp mult <= 0")
            _strength *= _mult
        except Exception:
            _strength = 0.0
        # Pre-warp disabled: keep DTF/text geometry crisp (no non-linear warp).
        # overlay = _prewarp_overlay_with_relief(overlay, relief_img_pre, strength_px=_strength)
    except Exception:
        pass

    def _img_to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    base_bytes = _img_to_png_bytes(base_work.convert("RGB"))
    # V2.5: Surface hint layer (pseudo normal/relief) derived from base.
    # Provides folds/seams/baffles guidance so the print deforms with the garment, not as a flat poster.
    try:
        _g = base_work.convert("RGB").convert("L")
        _e = _g.filter(ImageFilter.FIND_EDGES)
        _e = ImageOps.autocontrast(_e)
        _e = _e.filter(ImageFilter.GaussianBlur(2))
        # Boost subtle baffles/seams on dark fabrics.
        _e = ImageEnhance.Contrast(_e).enhance(2.2)
        surface_hint_img = _e.convert("RGB")
        surface_hint_bytes = _img_to_png_bytes(surface_hint_img)
    except Exception:
        surface_hint_bytes = None


    # V2.6: Seam/structure hint map (strong stitching/baffle/zipper lines) derived from base.
    # Helps the model "panelize" the print across seams/plackets instead of keeping the artwork perfectly rigid.
    try:
        _g2 = base_work.convert("RGB").convert("L")
        _s = _g2.filter(ImageFilter.FIND_EDGES)
        _s = ImageOps.autocontrast(_s)
        # Binarize to focus on structural lines.
        _s = _s.point(lambda p: 255 if p > 58 else 0)
        # Thicken seam lines a bit (more readable for the model).
        _s = _s.filter(ImageFilter.MaxFilter(5))
        # Slight blur to avoid harsh aliasing.
        _s = _s.filter(ImageFilter.GaussianBlur(0.8))
        seam_hint_img = _s.convert("RGB")
        seam_hint_bytes = _img_to_png_bytes(seam_hint_img)
    except Exception:
        seam_hint_bytes = None


    # V2.7: Edge/boundary hint map (hard garment boundaries like placket edges, zipper teeth borders, pocket edges).
    # This marks physical surface discontinuities so the model splits/terminates artwork at real edges instead of smoothing over them.
    try:
        _g3 = base_work.convert("RGB").convert("L")
        _h = _g3.filter(ImageFilter.FIND_EDGES)
        _h = ImageOps.autocontrast(_h)
        # Hard threshold to keep only strong boundaries.
        _h = _h.point(lambda p: 255 if p > 42 else 0)
        # Thicken boundaries for readability.
        _h = _h.filter(ImageFilter.MaxFilter(5))
        # Very light blur to avoid jagged aliasing.
        _h = _h.filter(ImageFilter.GaussianBlur(0.6))
        edge_hint_img = _h.convert("RGB")
        edge_hint_bytes = _img_to_png_bytes(edge_hint_img)
    except Exception:
        edge_hint_bytes = None

    if os.getenv('PRINTS_DEBUG') == '1':
        try:
            print(f"[prints][{run_id}] sha base={_sha256_bytes(base_bytes)} design={_sha256_bytes(design_bytes)} mask={_sha256_bytes(mask_bytes)}")
            if surface_hint_bytes:
                print(f"[prints][{run_id}] sha surface={_sha256_bytes(surface_hint_bytes)}")
            if seam_hint_bytes:
                print(f"[prints][{run_id}] sha seam={_sha256_bytes(seam_hint_bytes)}")
            if edge_hint_bytes:
                print(f"[prints][{run_id}] sha edge={_sha256_bytes(edge_hint_bytes)}")
        except Exception as _e:
            print(f"[prints][{run_id}] debug hash failed: {_e}")


    # --- MASK FROM OVERLAY ALPHA (ANTI-MATTE + POSITION LOCK) ---
    # Derive the editable mask from the *actual ink area* (overlay alpha) after all seam/zipper cuts.
    # This prevents Gemini from filling the entire quad with a rectangular matte/background.
    try:
        if overlay.mode != "RGBA":
            overlay = overlay.convert("RGBA")
        _a = overlay.split()[-1]
        _a_np = np.array(_a)
        _m_np = (_a_np > 10).astype(np.uint8) * 255
        # expand a bit to include small anti-aliased edges
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        _m_np = cv2.dilate(_m_np, _k, iterations=1)
        # soften edges slightly
        _m_np = cv2.GaussianBlur(_m_np, (0, 0), 0.8)
        orig_mask = Image.fromarray(_m_np).convert("L")
    except Exception:
        # Fallback: quad polygon mask (less strict; may allow matte).
        from PIL import ImageDraw
        orig_mask = Image.new("L", (wW, wH), 0)
        d = ImageDraw.Draw(orig_mask)
        d.polygon([(float(x), float(y)) for (x, y) in quad_work], fill=255)
        orig_mask = orig_mask.filter(ImageFilter.GaussianBlur(0.9))

    ref_bytes = base_bytes  # keep as explicit 'original base' reference
    guide_bytes = _img_to_png_bytes(guide_img)
    model_mask = orig_mask
    # Full-render is allowed only when placement_lock is OFF; otherwise we restrict edits to the placement mask.
    if getattr(inp, "render_mode", "full") == "full" and (not getattr(inp, "placement_lock", True)):
        model_mask = Image.new("L", orig_mask.size, 255)
    mask_bytes = _img_to_png_bytes(model_mask)
    overlay_bytes = _img_to_png_bytes(overlay)
    # DTF warp driver reference: helps Gemini bend/attach the whole print uniformly (even soft/pastel areas).
    try:
        warp_driver_img = _make_warp_driver(overlay, alpha_floor=0.70, blur_px=1.1)
        warp_driver_bytes = _img_to_png_bytes(warp_driver_img)
    except Exception:
        warp_driver_bytes = overlay_bytes
    # Ink carrier reference: for sparse/typographic logos, helps Gemini treat the whole print as ONE physical film.
    # Key: sparse typography needs a FILLED carrier over bbox; dense patches can use alpha-based carrier.
    try:
        bbox, fill_ratio = _alpha_bbox_and_fill_ratio(overlay)
        # Heuristic: typography/line-art tends to have low fill ratio.
        if fill_ratio < 0.22:
            ink_carrier_img = _make_ink_carrier_bbox(overlay, alpha_strength=110, margin_pct=0.12, blur_px=7.0)
        else:
            ink_carrier_img = _make_ink_carrier_from_overlay(overlay, alpha_strength=72, dilate_px=5, blur_px=2.0)
        ink_carrier_bytes = _img_to_png_bytes(ink_carrier_img)
    except Exception:
        ink_carrier_bytes = warp_driver_bytes

    # Fabric relief map reference from base: helps stabilize curvature across baffles.
    try:
        relief_img = relief_img_pre if (relief_img_pre is not None) else _make_relief_map_from_base(base_work.convert("RGB"), blur_px=8.0, contrast=1.9)
        relief_bytes = _img_to_png_bytes(relief_img)
    except Exception:
        relief_bytes = ref_bytes


    # Garment anchors map (seams/pocket/zipper edges) inside the mask: helps Gemini *not avoid* seams.
    try:
        _mask_l = mask_img.convert("L") if (mask_img is not None) else Image.new("L", base_work.size, 255)
        anchors_map = _make_garment_anchors_map(base_work.convert("RGB"), _mask_l, blur_px=1.0)
        anchors_bytes = _img_to_png_bytes(anchors_map)
    except Exception:
        anchors_bytes = relief_bytes

    # Composite reference (base + warped overlay) to reduce "outline-only" failures.
    try:
        _comp = base_work.convert("RGBA").copy()
        _comp.alpha_composite(overlay)
        comp_bytes = _img_to_png_bytes(_comp.convert("RGB"))
    except Exception:
        comp_bytes = ref_bytes


    if surface_hint_bytes is None:
        # Fallback to relief_bytes if surface hint extraction failed.
        surface_hint_bytes = relief_bytes

    if 'seam_hint_bytes' in locals() and seam_hint_bytes is None:
        # Fallback to surface hint (or relief) if seam hint extraction failed.
        seam_hint_bytes = surface_hint_bytes
    if 'edge_hint_bytes' in locals() and edge_hint_bytes is None:
        # Fallback to seam hint (or surface hint) if edge hint extraction failed.
        edge_hint_bytes = seam_hint_bytes if 'seam_hint_bytes' in locals() else surface_hint_bytes



    profile_text = (
        f"\n\nPRINT TYPE PROFILE:\n"
        f"- mode: {str(mode)}\n"
        f"- description: {profile.get('description')}\n"
        f"- thickness: {profile.get('thickness')}\n"
        f"- gloss: {profile.get('gloss')}\n"
        f"- deformation_follow: {profile.get('deformation_follow')}\n"
        f"- edge_softness: {profile.get('edge_softness')}\n"
        f"- edge_clearance_px: {profile.get('edge_clearance_px')}\n"
        f"- remove_underlying: {profile.get('remove_underlying')}\n"
    )
    # Add surface + freeform rules from YAML (optional)
    surface = (profile.get('surface') or 'fabric')
    if surface:
        profile_text += f"- surface: {surface}\n"
    rules_list = profile.get('rules') or []
    if isinstance(rules_list, (list, tuple)) and rules_list:
        profile_text += "PROFILE RULES:\n" + "\n".join([f"- {str(r).strip()}" for r in rules_list if str(r).strip()]) + "\n"
    use_relief_map = bool(profile.get('use_relief_map', str(surface).lower() != 'rigid'))
    use_anchors_map = bool(profile.get('use_anchors_map', str(surface).lower() != 'rigid'))
    # --- /PRINT TYPE PROFILE ---

    # --- NON-SEWN PRINT RULES (anti-sticker / fabric-integrated) ---
    mode_upper = str(mode).upper().strip() if mode else ""
    # "Ink-like" modes: applied/printed into the surface (not sewn, not rigid attach)
    INK_MODES = {"DTF", "SUBLIMATION", "PRINT", "INK", "TRANSFER", "HEAT_TRANSFER"}
    # Rigid/sewn/attached modes where thickness/background rules differ
    ATTACH_MODES = {"FABRIC_PATCH", "RUBBER_PATCH", "METAL_BADGE"}
    is_ink_mode = (mode_upper in INK_MODES) or (mode_upper and mode_upper not in ATTACH_MODES and "PATCH" not in mode_upper and "BADGE" not in mode_upper and "EMBROID" not in mode_upper)
    ink_rules = ""
    if is_ink_mode:
        ink_rules = (
            "\n\nINK-ON-FABRIC RULES (STRICT):\n"
            "- This is NOT a sticker/label/patch. Do NOT add any rectangular or solid background behind the logo.\n"
            "- Apply ONLY the logo shape itself. Any transparent area must remain the original fabric.\n"
            "- The logo must look printed/dyed into the fabric: preserve fabric texture, grain, wrinkles, and shading through the logo.\n"
            "- For quilted/puffer garments: quilting seams/channels must remain visible; the print follows the puff channels and bends naturally.\n"
            "- Do NOT create a flat white plate. Match local exposure and lighting so the print integrates realistically.\n"
        )
    # --- /NON-SEWN PRINT RULES ---

    # --- DTF SURFACE DISCONTINUITY RULES (pockets / zippers / seams) ---
    # When mode is DTF-like ink print, the design must behave as a physical ink layer across garment parts.
    # This block strengthens "fabric-aware" behavior on difficult garments (hoodie pockets, center zippers, panel seams).
    dtf_surface_rules = ""
    is_dtf_like = ("DTF" in mode_upper) or (mode_upper in {"PRINT", "TRANSFER", "HEAT_TRANSFER"})
    if is_ink_mode and is_dtf_like:
        # DTF must stay CRISP. We allow only photometric integration (lighting/shadows),
        # but FORBID non-linear geometric warping that would distort letters.
        dtf_surface_rules = (
            "\nDTF PHYSICAL INTEGRATION (HARD RULES):\n"
            "- Keep the design geometry EXACTLY as in the LOGO OVERLAY (no bending, no liquify, no stretching of letters).\n"
            "- You may apply ONLY subtle brightness/shadow modulation INSIDE the design to match fabric lighting (print-on-fabric look).\n"
            "- Preserve sharp edges and clean text; no blur, no smearing, no painterly redraw.\n"
            "- If a zipper/placket seam crosses the region, the design is naturally split by that seam (a thin gap), but each side stays sharp.\n"
            "- FORBIDDEN: duplicated logos, extra outlines/contours, emboss/vinyl shine, patch borders, stickers, or background plates.\n"
        )
    # --- /DTF SURFACE DISCONTINUITY RULES ---

    # --- FRAME LOCK / CROPPING RULES ---
    # placement_lock=True  => ABSOLUTE placement (do NOT move)
    # placement_lock=False => allow slight adjustment, but stay INSIDE the user rectangle
    _plock = bool(getattr(inp, "placement_lock", True))
    if _plock:
        frame_lock_rules = (
            "\nPLACEMENT CONSTRAINTS (HARD RULES):\n"
            "- The placement rectangle/quad is ABSOLUTE. Do NOT move, recenter, rotate, or rescale the design to fit.\n"
            "- Render the design exactly as positioned by the user, even if only part of it is visible.\n"
            "- If any part of the design falls outside the garment (background), it MUST be CROPPED/REMOVED, not moved back onto the garment.\n"
            "- The print MUST conform to the garment volume: seams, folds, and pocket edges (fabric-aware deformation).\n"
            "- The result must look like a real physical print applied before the photo was taken (not a flat overlay).\n"
            "- If the placement crosses seams, pocket edges, zippers, drawstrings, or layered parts, the print MUST split/occlude naturally: "
            "parts on top layers stay on those layers; parts under occluders are hidden; keep a tiny clearance along zippers/edges.\n"
            "- Never add any extra background plate/box behind the logo. Only the logo shape is applied.\n- If the design asset has its own background, treat that background as NOT PART OF THE LOGO (discard it).\n"
        )
    else:
        frame_lock_rules = (
            "\nPLACEMENT CONSTRAINTS (RELAXED):\n"
            "- Keep the print INSIDE the user rectangle/quad. You may slightly shift/scale/rotate within it (<=5%) ONLY if needed to follow seams/zipper/pocket edges.\n"
            "- Do NOT move the design outside the rectangle. Do NOT change its proportions or redraw.\n"
            "- If any part falls outside the garment (background), CROP/REMOVE it (never move it back).\n"
            "- Still follow garment volume: wrinkles, seams, pocket edges, zipper split/clearance.\n"
            "- Never add any extra background plate/box behind the logo. Only the logo shape is applied.\n- If the design asset has its own background, treat that background as NOT PART OF THE LOGO (discard it).\n"
        )
    # --- /FRAME LOCK / CROPPING RULES ---



    # --- PRINT MODE LOCK (prevents model from "switching" to other application types) ---
    mode_lock_rules = ""
    if mode_upper:
        if "DTF" in mode_upper:
            mode_lock_rules = (
                "\nPRINT MODE LOCK: DTF (DIRECT TO FILM)\n"
                "- This application is STRICTLY DTF ink transfer.\n"
                "- FORBIDDEN: vinyl/HTV/flex film look, patch, embroidery, silicone/rubber, emboss/deboss, engraving, foil, metallic, brooch/badge.\n"
                "- FORBIDDEN: outline-only / contour rendering. The logo MUST be solid filled where the overlay is filled.\n"
                "- Finish: matte or semi-matte ink on fabric surface. No glossy plastic shine, no bevel/3D edge.\n"
                "- Preserve overlay colors exactly. If the overlay is white, it must remain solid white (not gray, not outline).\n"
            )
        elif "PATCH" in mode_upper:
            mode_lock_rules = (
                "\nPRINT MODE LOCK: PATCH\n"
                "- This is a PATCH/ATTACHED OBJECT. It may have thickness and an edge, but must stay within the mask.\n"
                "- FORBIDDEN: converting into ink print, embroidery stitches, or metallic foil unless requested by the profile.\n"
            )
        elif "EMBROID" in mode_upper:
            mode_lock_rules = (
                "\nPRINT MODE LOCK: EMBROIDERY\n"
                "- This is EMBROIDERY (thread). Use stitched texture and thread direction.\n"
                "- FORBIDDEN: flat ink print, vinyl shine, embossed leather look, or metallic foil.\n"
            )
    # --- /PRINT MODE LOCK ---
    payload_mode = (os.getenv("PRINTS_GEMINI_PAYLOAD", "lite") or "lite").strip().lower()
    if payload_mode not in ("lite", "full"):
        payload_mode = "lite"

    input_desc = (
        "Input images:\n"
        "- BASE: the original photo (edit ONLY inside MASK).\n"
        "- MASK: white=editable, black=protected (outside MUST stay identical).\n"
        "- LOGO OVERLAY: the design already positioned/scaled/rotated to match the user's frame.\n"
    )
    if payload_mode == "lite":
        input_desc += "- HINT PACK (RGB): optional guidance maps (R=seams/edges, G=relief, B=anchors).\n"
    else:
        input_desc += "- HELPER MAPS: outline guide, relief, anchors, composite reference, etc.\n"

    prompt = (
        "You are an image editor.\n"
        + frame_lock_rules
        + mode_lock_rules
        + input_desc
        + "\nTask: Apply the logo STRICTLY inside the MASK, matching the LOGO OVERLAY placement exactly.\n"
        + "Rules:\n"
        + "- STRICT PLACEMENT LOCK: DO NOT re-center the design. DO NOT align to garment symmetry. DO NOT move it to typical chest/back/center positions.\n"
        + "- Match overlay position/size/rotation/perspective EXACTLY (pixel-perfect). No auto-adjustments.\n"
        + "- If the logo overlay contains any matte/background color, remove it completely. Print ONLY the artwork; background must NOT be printed.\n"
        + "- Keep garment structure: preserve seams/stitching/zipper lines; do not paint over them. If a zipper/placket crosses the region, the logo is naturally split by that seam (thin gap).\n"
        + "- Preserve the logo colors and transparency exactly as in the overlay (no recolor, no invert, no outline/contour rendering, no thickening strokes).\n"
        + "- Treat overlay pixels as the source of truth: transfer them onto fabric; DO NOT redraw or invent new paint.\n"
        + "- Do not modify anything outside the mask (outside MUST remain identical to BASE).\n"
        + "- Keep the same composition, person, clothing, and lighting as the base image.\n"
    ) + ink_rules + dtf_surface_rules + profile_text

    # If a TXT prompt template is configured for this mode, use it (keeps prompts out of code).
    try:
        # Mode label from UI may be like "Print (DTF)"; treat any mode containing "DTF" as DTF.
        is_dtf = ('DTF' in mode_upper)
        template_name = profile.get('prompt') or ('dtf.txt' if is_dtf else 'default.txt')
        prompt = _render_print_prompt(template_name, frame_lock_rules, mode_lock_rules, profile_text) + (ink_rules or '') + (dtf_surface_rules or '')
        prompt += f"\n\nRUN_ID: {run_id}\nAlways treat this as a fresh, independent run. Do not reuse any prior output.\n"
        if os.getenv('PRINTS_PROMPT_DEBUG', '').strip() in ('1', 'true', 'True', 'yes', 'YES'):
            try:
                head = (prompt or '').replace('\n', ' ')[:200]
                print(f"[prints] template={template_name} mode={inp.mode} head={head}")
            except Exception:
                pass
    except Exception:
        # fallback to the in-code prompt above
        pass

    # Surface-dependent references (rigid surfaces should NOT use fabric relief/anchors).
    if 'use_relief_map' in locals() and not use_relief_map:
        relief_bytes = ref_bytes
    if 'use_anchors_map' in locals() and not use_anchors_map:
        anchors_bytes = ref_bytes



    # --- Gemini payload size control ---
    # full: send many helper maps (legacy)
    # lite: send only what is needed for strict placement + realism
    
    def _pack_hint_rgb(*, seam_png: bytes, relief_png: bytes, anchors_png: bytes, target_size: tuple[int, int]) -> bytes:
        """Pack multiple grayscale hint maps into a single RGB image to reduce Gemini payload.
        R = seams/edges, G = relief/wrinkles, B = anchors/structure.
        """
        import io
        from PIL import Image

        def _to_l(img_bytes: bytes) -> Image.Image:
            im = Image.open(io.BytesIO(img_bytes)).convert("L")
            if im.size != target_size:
                im = im.resize(target_size, Image.BILINEAR)
            return im

        W0, H0 = target_size
        try:
            r = _to_l(seam_png)
        except Exception:
            r = Image.new("L", (W0, H0), 0)
        try:
            g = _to_l(relief_png)
        except Exception:
            g = Image.new("L", (W0, H0), 0)
        try:
            b = _to_l(anchors_png)
        except Exception:
            b = Image.new("L", (W0, H0), 0)

        rgb = Image.merge("RGB", (r, g, b))
        buf = io.BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _build_lite_prompt() -> str:
        mode_desc = (profile or {}).get("description") or (mode or "print")
        surface = (profile or {}).get("surface") or "surface"
        # Keep this short and unambiguous: fewer words, higher compliance.
        return (
            "PRINT INSERT TASK (STRICT)\n"
            f"MODE: {mode_desc}. SURFACE: {surface}.\n\n"
            "INPUT IMAGES ORDER:\n"
            "1) BASE (photo).\n"
            "2) MASK (edit zone): WHITE=can change, BLACK=must stay identical.\n"
            "3) OVERLAY (target design placement/size/rotation inside the mask).\n"
            "4) COMPOSITE-REF (BASE + OVERLAY already placed). This is the ground-truth placement.\n"
            "5) HINT-PACK RGB: R=seams/edges, G=wrinkles/relief, B=anchors/structure.\n\n"
            "HIGHEST PRIORITY RULES:\n"
            "- Outside MASK: keep pixel-perfect identical to BASE. Do not change anything.\n"
            "- Inside MASK: keep the design EXACTLY where COMPOSITE-REF/OVERLAY show it (pixel-perfect). NEVER move, re-center, or re-align. Do NOT move/scale/rotate/flip the design.\n"
            "- Treat the design as SOLID INK/DTF print (no outline, no stroke, no patch, no emboss).\n"
            "- Preserve the underlying surface realism: wrinkles, fabric weave, stitching, holes, texture must show through the print.\n"
            "- Respect structure: do not paint across zipper/placket; if seams indicate a split, leave a thin gap at the seam.\n"
            "- Keep background transparent: never add a rectangle/matte around the logo.\n- IMPORTANT: OVERLAY may contain a background (white/colored/gradient/rectangle). IGNORE it.\n- Use ONLY the visible subject pixels from OVERLAY (as if background were transparent).\n- If you see any rectangular plate from OVERLAY/COMPOSITE-REF, REMOVE it completely; apply only the subject.\n"
            "- Keep colors faithful to OVERLAY; avoid inventing extra shadows or changing garment color.\n\n"
            "OUTPUT: Return one edited image (same size as BASE)."
        )

    # Build parts list depending on payload_mode
    parts = None
    if payload_mode == "lite":
        prompt = _build_lite_prompt()
        hint_pack_bytes = _pack_hint_rgb(
            seam_png=seam_hint_bytes,
            relief_png=relief_bytes,
            anchors_png=anchors_bytes,
            target_size=(W, H),
        )
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(base_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(mask_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(overlay_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(comp_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(hint_pack_bytes)}},
        ]
    else:
        parts = [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(base_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(surface_hint_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(seam_hint_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(edge_hint_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(guide_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(ref_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(mask_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(relief_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(ink_carrier_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(warp_driver_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(anchors_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(overlay_bytes)}},
            {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(comp_bytes)}},
        ]

    body = {
        "generationConfig": {"temperature": 0.05, "topP": 0.9, "candidateCount": 1},
        "contents": [{
            "role": "user",
            "parts": parts,
        }],
    }
    def _safe_model_id(x: str) -> str:
        s = (x or "").strip()
        # Requests may choke on non-ASCII in URL path; enforce a safe model id.
        try:
            s.encode("ascii")
        except Exception:
            return os.getenv("PRINTS_GEMINI_MODEL_DEFAULT", "gemini-3-image")
        # Normalize possible inputs:
        # - "models/gemini-..." -> "gemini-..."
        # - "gemini-..." -> keep
        if s.startswith("models/gemini-"):
            s = s[len("models/"):]
        if s.startswith("gemini-"):
            return s
        # If someone passed displayName like "Nano Banana 2" — try extract gemini-* else fallback
        if "gemini-" in s:
            m = re.search(r"(gemini-[A-Za-z0-9.\-]+)", s)
            if m:
                return m.group(1)
        return os.getenv("PRINTS_GEMINI_MODEL_DEFAULT", "gemini-3-image")

    # Model selection: prefer Gemini Image 3 (stable image generation) but keep a safe fallback.
    # You can override via cfg.image_model or env PRINTS_GEMINI_MODEL_DEFAULT.
    image_model = _safe_model_id(getattr(cfg, 'image_model', '') or '')
    model_try = [image_model]
    # If caller didn't specify, try the legacy preview as a fallback.
    if image_model != "gemini-3.1-flash-image-preview":
        model_try.append("gemini-3.1-flash-image-preview")
    last_resp = None
    for _m in model_try:
        last_resp = post_generate_content(cfg.api_key, _m, body, timeout=180)
        # If transport error or model-not-found, try next.
        if isinstance(last_resp, dict) and (last_resp.get("__http_error__") or last_resp.get("ok") is False):
            _msg = (last_resp.get("text") or last_resp.get("error") or last_resp.get("message") or "").lower()
            if ("not found" in _msg) or ("model" in _msg and "not" in _msg):
                continue
        resp = last_resp
        break
    else:
        resp = last_resp

    # Robust error surface: do not lose Gemini error text (avoid bare RuntimeError).
    import json as _json  # local import to keep file-level deps unchanged
    if not isinstance(resp, dict):
        raise RuntimeError(f"Gemini bad response type: {type(resp).__name__}: {resp!r}")

    # gemini_rest marks transport/HTTP failures with "__http_error__".
    if resp.get("__http_error__") or (resp.get("ok") is False):
        msg = (
            resp.get("text")
            or resp.get("error")
            or resp.get("message")
            or resp.get("detail")
        )
        if not msg:
            try:
                msg = "Gemini HTTP error. resp=" + _json.dumps(resp, ensure_ascii=False)[:2000]
            except Exception:
                msg = "Gemini HTTP error. resp=<unserializable>"
        raise RuntimeError(str(msg))

    # Extract first returned image. Gemini may return multiple candidates; use a robust extractor.
    out_b64 = ""

    # If the model attempted a tool/function call (e.g., google:image_gen) the response can come back
    # as MALFORMED_FUNCTION_CALL. For print modes we must not fail the whole request — fallback.
    try:
        c0 = (resp.get('candidates') or [None])[0] or {}
        fr = (c0.get('finishReason') or '').upper()
        fm = (c0.get('finishMessage') or '')
        if fr == 'MALFORMED_FUNCTION_CALL' or ('MALFORMED_FUNCTION_CALL' in (fm or '')):
            out = apply_design(inp)
            try:
                out['ai'] = False
                out['via'] = 'deterministic_fallback'
                out['warning'] = 'Gemini returned MALFORMED_FUNCTION_CALL; used deterministic compositor.'
            except Exception:
                pass
            return out
    except Exception:
        pass

    try:
        out_b64 = _extract_first_image_b64(resp)
    except Exception:
        out_b64 = ""

    if not out_b64:
        # Surface useful diagnostics (finishReason / safetyRatings / possible text-only response)
        try:
            c0 = (resp.get("candidates") or [None])[0] or {}
            finish = c0.get("finishReason")
            safety = c0.get("safetyRatings")
            parts = ((c0.get("content") or {}).get("parts")) or []
            text_parts = []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    text_parts.append(p.get("text"))
            txt = (" ".join(text_parts)).strip()
        except Exception:
            finish, safety, txt = None, None, ""
        # Avoid dumping huge payloads; keep a short snippet.
        try:
            snippet = _json.dumps(resp, ensure_ascii=False)[:1500]
        except Exception:
            snippet = "<unserializable resp>"
        hint = (
            f"Gemini returned no image. "
            f"finishReason={finish!r} "
            f"safetyRatings={safety!r} "
            f"text={txt[:300]!r} "
            f"resp_snippet={snippet}"
        )
        raise RuntimeError(hint)

    out_bytes = base64.b64decode(out_b64)
    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
    try:
        _ov_sh = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
        out_img = _sharpen_print_area(out_img, _ov_sh)
    except Exception:
        pass
    # Heuristic retry: if the print looks too flat (not following fabric folds), do up to TWO stronger DTF conform retries.
    try:
        _ref_img0 = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        _ov0 = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
        mode_norm_local = _normalize_mode(mode)
        max_tries = 2 if ("dtf" in mode_norm_local) else 0
        for _attempt in range(max_tries):
            if not _flatness_failure(out_img, _ref_img0, _ov0):
                break
            try:
                ink_carrier_img2 = _make_ink_carrier_from_overlay(_ov0, alpha_strength=120, dilate_px=7, blur_px=2.6)
                ink_carrier_bytes2 = _img_to_png_bytes(ink_carrier_img2)
                prompt2 = prompt + "\n\nDTF RETRY (STRONGER CONFORM): Apply continuous curvature across the ENTIRE print block using the relief map; do not keep any part flat."
                import copy as _copy
                body2 = _copy.deepcopy(body)
                parts2 = body2["contents"][0]["parts"]
                # parts: [text, base, guide, ref, mask, relief, ink_carrier, warp_driver, overlay, comp]
                parts2[0] = {"text": prompt2}
                # ink_carrier is index 6
                parts2[6] = {"inlineData": {"mimeType": "image/png", "data": bytes_to_b64(ink_carrier_bytes2)}}
                resp2 = post_generate_content(cfg.api_key, image_model, body2, timeout=180)
                out_b64_2 = _extract_first_image_b64(resp2)
                if out_b64_2:
                    out_bytes = base64.b64decode(out_b64_2)
                    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
            except Exception:
                pass
    except Exception:
        pass
# Heuristic guard: sometimes Gemini "traces" the logo as an outline instead of preserving the fill.
    # If we detect edge-only change inside the overlay mask, we do ONE stricter retry (still placement-locked).
    try:
        _ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        _ov = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
        _msk = Image.open(io.BytesIO(mask_bytes)).convert("L")

        def _outline_only_failure(out_img_rgb: Image.Image, ref_rgb: Image.Image, ov_rgba: Image.Image, msk_l: Image.Image) -> bool:
            # Build alpha mask from warped overlay (fallback to provided mask).
            a = ov_rgba.split()[-1] if ov_rgba.mode == "RGBA" else msk_l
            a = a.resize(out_img_rgb.size, Image.NEAREST)
            ref_rgb = ref_rgb.resize(out_img_rgb.size, Image.BILINEAR)
            # Inner region (erode) and edge ring.
            a_inner = a.filter(ImageFilter.MinFilter(5))
            a_edge = ImageChops.subtract(a.filter(ImageFilter.MaxFilter(5)), a_inner)
            # Binarize
            inner = a_inner.point(lambda p: 255 if p > 180 else 0)
            edge = a_edge.point(lambda p: 255 if p > 40 else 0)
            # If there is almost no inner area, skip.
            if ImageStat.Stat(inner).sum[0] < 255 * 200:
                return False
            # Measure change vs ref in inner and edge areas.
            diff = ImageChops.difference(out_img_rgb, ref_rgb).convert("L")
            inner_mean = ImageStat.Stat(diff, mask=inner).mean[0]
            edge_mean = ImageStat.Stat(diff, mask=edge).mean[0] if ImageStat.Stat(edge).sum[0] > 0 else 0.0
            # Outline failure: edges changed noticeably, but the inner fill barely changed.
            return (edge_mean >= max(12.0, inner_mean * 1.8)) and (inner_mean < 10.0)

        if _outline_only_failure(out_img, _ref_img, _ov, _msk):
            prompt_retry = (prompt + "\n\n" +
                "CRITICAL (RETRY): Do NOT convert the logo into line art/outline. "
                "The filled areas of the logo must remain SOLID and match the LOGO OVERLAY exactly. "
                "Copy the overlay pixels inside the mask and only add subtle fabric shading. ")
            body["contents"][0]["parts"][0]["text"] = prompt_retry
            body["generationConfig"] = {"temperature": 0.0, "topP": 0.9}
            resp2 = post_generate_content(cfg.api_key, image_model, body, timeout=180)
            try:
                out_b64_2 = _extract_first_image_b64(resp2)
                out_bytes2 = _b64_to_bytes(out_b64_2)
                if out_bytes2:
                    out_img = Image.open(io.BytesIO(out_bytes2)).convert("RGB")
            except Exception:
                pass
    except Exception:
        pass

    # Keep the model's full output (for optional full re-render mode)
    model_full = out_img.copy() if out_img is not None else None

    # If we worked in a zoomed ROI, paste the edited ROI back into the full base image.
    if paste_info:
        rx0, ry0, rx1, ry1 = paste_info["rx0"], paste_info["ry0"], paste_info["rx1"], paste_info["ry1"]
        scale = float(paste_info["scale"] or 1.0)
        target_size = (int(rx1 - rx0), int(ry1 - ry0))
        if scale != 1.0:
            out_roi = out_img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            out_roi = out_img
        full = base_img.convert("RGB").copy()
        full.paste(out_roi, (int(rx0), int(ry0)))
        out_img = full

    used_full_render = False
    if getattr(inp, "render_mode", "full") == "full" and model_full is not None:
        # Accept full re-render only if the model preserved the scene outside the print area.
        try:
            cand_rgb = _resize_like(model_full, base_img).convert("RGB")
            base_rgb0 = base_img.convert("RGB")
            m_img = orig_mask
            # Expand the protected (inside) region so strong fabric-aware deformation near edges
            # does not fail outside-area validation.
            try:
                m_img = m_img.filter(ImageFilter.MaxFilter(17))  # ~8px dilation
            except Exception:
                pass
            m_arr = np.array(m_img, dtype=np.uint8)
            outside = m_arr < 10
            if outside.any():
                diff = np.abs(np.array(base_rgb0, dtype=np.int16) - np.array(cand_rgb, dtype=np.int16))
                mean = float(diff[outside].mean())
                mx = int(diff[outside].max())
            else:
                mean, mx = 0.0, 0
            # Loose thresholds: allow mild re-lighting / denoise, but block scene rewrites.
            if mean <= 8.0 and mx <= 60:
                out_img = cand_rgb
                used_full_render = True
        except Exception:
            used_full_render = False



    # --- POST-PROCESS: HARD PLACEMENT (quad) + GARMENT CROP + ZIPPER GAP ---
    try:
        from PIL import ImageDraw, ImageChops, ImageStat

        base_rgb = base_img.convert("RGB")
        full_w, full_h = base_rgb.size

        # polygon mask from quad (hard constraint)
        poly_mask = Image.new("L", (full_w, full_h), 0)
        ImageDraw.Draw(poly_mask).polygon([(float(x), float(y)) for (x, y) in quad_px], fill=255)

        # work bbox (expanded) for local analysis
        x0, y0, x1, y1 = _bbox_of_pts(quad_px)
        pad = int(max(8, 0.06 * max(x1 - x0, y1 - y0)))
        bx0 = max(0, int(x0) - pad); by0 = max(0, int(y0) - pad)
        bx1 = min(full_w, int(x1) + pad); by1 = min(full_h, int(y1) + pad)
        bbox = (bx0, by0, bx1, by1)

        # estimate background color from image corners
        def _corner_mean(img, x, y, s=14):
            patch = img.crop((x, y, min(full_w, x + s), min(full_h, y + s)))
            st = ImageStat.Stat(patch)
            return tuple(int(v) for v in st.mean[:3])

        corners = [
            _corner_mean(base_rgb, 0, 0),
            _corner_mean(base_rgb, full_w - 14, 0),
            _corner_mean(base_rgb, 0, full_h - 14),
            _corner_mean(base_rgb, full_w - 14, full_h - 14),
        ]
        bg = tuple(int(sum(c[i] for c in corners) / len(corners)) for i in range(3))

        # garment mask: pixels sufficiently different from background
        roi = base_rgb.crop(bbox)
        bg_img = Image.new("RGB", roi.size, bg)
        diff = ImageChops.difference(roi, bg_img).convert("L")

        # threshold tuned to work on white/gray studio backgrounds
        # use bg noise estimate from corners inside ROI to adapt a bit
        small = diff.resize((max(32, diff.size[0] // 4), max(32, diff.size[1] // 4)))
        st = ImageStat.Stat(small)
        mean_val = float(st.mean[0])
        std_val = float(st.stddev[0])
        thr = int(max(18, min(60, mean_val + 2.8 * std_val)))

        garment_roi = diff.point(lambda p, t=thr: 255 if p > t else 0)

        # cleanup (close small holes / smooth edges)
        garment_roi = garment_roi.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        garment_roi = garment_roi.filter(ImageFilter.GaussianBlur(0.6)).point(lambda p: 255 if p > 64 else 0)

        garment_mask = Image.new("L", (full_w, full_h), 0)
        garment_mask.paste(garment_roi, (bx0, by0))

        
        # zipper + pocket edge detection inside ROI (approx)
        # Goal: prevent ink crossing center zipper/placket and pocket openings (diagonal seams/zip pockets).
        zipper_mask = Image.new("L", (full_w, full_h), 0)
        pocket_mask = Image.new("L", (full_w, full_h), 0)
        try:
            edges = roi.convert("L").filter(ImageFilter.FIND_EDGES)
            e_small_w = min(220, max(80, edges.size[0] // 2))
            e_small_h = min(220, max(80, edges.size[1] // 2))
            e_small = edges.resize((e_small_w, e_small_h))
            e_bin = e_small.point(lambda p: 255 if p > 40 else 0)

            data = list(e_bin.getdata())

            # ---- zipper: dominant vertical edge (bias to center) ----
            col_sum = [0] * e_small_w
            for yy in range(e_small_h):
                row_off = yy * e_small_w
                for xx in range(e_small_w):
                    if data[row_off + xx]:
                        col_sum[xx] += 1

            best_x = max(range(e_small_w), key=lambda i: col_sum[i])
            best_score = col_sum[best_x] / float(e_small_h)

            cx0 = e_small_w // 2
            cwin_list = list(range(max(0, cx0 - 2), min(e_small_w, cx0 + 3)))
            center_score = (sum(col_sum[i] for i in cwin_list) / max(1.0, float(len(cwin_list)))) / float(e_small_h)

            use_x = None
            thr_zip = float(os.getenv("PRINTS_ZIPPER_THR", "0.18") or "0.18")
            if center_score > thr_zip:
                use_x = cx0
            elif best_score > max(thr_zip, 0.22):
                if 0.15 * e_small_w <= best_x <= 0.85 * e_small_w:
                    use_x = best_x

            if use_x is not None:
                stripe_w = int(max(4, 0.022 * (bx1 - bx0)))
                cx = bx0 + int(use_x / float(max(1, e_small_w - 1)) * (bx1 - bx0))
                zx0 = max(bx0, cx - stripe_w // 2)
                zx1 = min(bx1, cx + stripe_w // 2 + 1)
                ImageDraw.Draw(zipper_mask).rectangle([zx0, by0, zx1, by1], fill=255)
                zipper_mask = zipper_mask.filter(ImageFilter.MaxFilter(5))

            # ---- pockets: strong diagonal lines in lower half of ROI ----
            def _draw_diag_line(mask_img, kind, c_val, width_px):
                pts = []
                w = e_small_w - 1
                h = e_small_h - 1
                if kind == "t1":
                    c = float(c_val)
                    cand = [(c, 0.0), (c - h, float(h)), (0.0, c), (float(w), c - w)]
                else:
                    k = float(c_val)
                    cand = [(k, 0.0), (k + h, float(h)), (0.0, -k), (float(w), float(w) - k)]
                for x, y in cand:
                    if 0.0 <= x <= float(w) and 0.0 <= y <= float(h):
                        pts.append((x, y))
                if len(pts) < 2:
                    return
                best = None
                best_d = -1.0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dx = pts[i][0] - pts[j][0]
                        dy = pts[i][1] - pts[j][1]
                        d = dx * dx + dy * dy
                        if d > best_d:
                            best_d = d
                            best = (pts[i], pts[j])
                if not best:
                    return
                (x1s, y1s), (x2s, y2s) = best

                def _mx(x):
                    return bx0 + float(x) / float(max(1, w)) * float(bx1 - bx0)
                def _my(y):
                    return by0 + float(y) / float(max(1, h)) * float(by1 - by0)

                p1 = (_mx(x1s), _my(y1s))
                p2 = (_mx(x2s), _my(y2s))
                ImageDraw.Draw(mask_img).line([p1, p2], fill=255, width=int(width_px))

            bins_len = int(e_small_w + e_small_h + 4)
            b1 = [0] * bins_len
            b2 = [0] * bins_len
            for yy in range(e_small_h):
                row_off = yy * e_small_w
                for xx in range(e_small_w):
                    if data[row_off + xx]:
                        b1[xx + yy] += 1
                        b2[xx - yy + (e_small_h - 1)] += 1

            c1 = max(range(len(b1)), key=lambda i: b1[i])
            c2 = max(range(len(b2)), key=lambda i: b2[i])

            def _len_t1(c):
                x_min = max(0, c - (e_small_h - 1))
                x_max = min((e_small_w - 1), c)
                return max(1, x_max - x_min + 1)

            def _len_t2(k):
                kk = k - (e_small_h - 1)
                x_min = max(0, kk)
                x_max = min((e_small_w - 1), kk + (e_small_h - 1))
                return max(1, x_max - x_min + 1)

            s1 = float(b1[c1]) / float(_len_t1(c1))
            s2 = float(b2[c2]) / float(_len_t2(c2))

            thr_pocket = float(os.getenv("PRINTS_POCKET_EDGE_THR", "0.55") or "0.55")
            min_cnt = int(float(os.getenv("PRINTS_POCKET_EDGE_MINCNT", "70") or "70"))

            def _passes_lower(kind, c_val):
                xmid = float(e_small_w - 1) * 0.5
                if kind == "t1":
                    ymid = float(c_val) - xmid
                else:
                    kk = float(c_val) - float(e_small_h - 1)
                    ymid = xmid - kk
                return (0.45 * (e_small_h - 1)) <= ymid <= (0.98 * (e_small_h - 1))

            stripe_w2 = int(max(3, 0.018 * (bx1 - bx0)))
            if (b1[c1] >= min_cnt) and (s1 >= thr_pocket) and _passes_lower("t1", c1):
                _draw_diag_line(pocket_mask, "t1", c1, stripe_w2)
            if (b2[c2] >= min_cnt) and (s2 >= thr_pocket) and _passes_lower("t2", c2):
                _draw_diag_line(pocket_mask, "t2", c2, stripe_w2)

            if pocket_mask.getbbox():
                pocket_mask = pocket_mask.filter(ImageFilter.MaxFilter(7))
        except Exception:
            pass

# combine: inside placement AND on garment (if garment mask is reliable) AND not on zipper.
        # For non-garment items (caps, hard goods), garment_mask can be empty/too small because background is similar.
        # In that case we fall back to poly_mask only.
        combined = ImageChops.multiply(poly_mask, garment_mask)
        try:
            import numpy as _np
            gm = _np.array(garment_mask, dtype="uint8")
            # coverage of garment mask inside placement bbox
            bx = poly_mask.getbbox()
            if bx:
                x0,y0,x1,y1 = bx
                region = gm[y0:y1, x0:x1]
                cov = float(region.mean())/255.0  # 0..1
            else:
                cov = float(gm.mean())/255.0
            if cov < 0.05:
                combined = poly_mask.copy()
        except Exception:
            pass
        if zipper_mask.getbbox():
            combined = ImageChops.subtract(combined, zipper_mask)
        if pocket_mask.getbbox():
            combined = ImageChops.subtract(combined, pocket_mask)


        # soften mask edges to avoid jaggies/halos on hard logos
        try:
            combined = combined.filter(ImageFilter.GaussianBlur(radius=1))
            if used_full_render:
                # For full render we still preserve the base outside the placement area.
                # Let the model "break" the design inside, but keep identity outside.
                try:
                    combined = combined.filter(ImageFilter.MaxFilter(9))  # ~4px expand
                    combined = combined.filter(ImageFilter.GaussianBlur(radius=3.2))
                except Exception:
                    combined = combined.filter(ImageFilter.GaussianBlur(radius=2.0))
        except Exception:
            pass

        # If Gemini tried to move the logo away, restrict edits to our combined mask.
        # (We simply keep base outside mask, taking edited pixels only inside.)
        out_rgb = out_img.convert("RGB")

        # optional fallback: if edited region is basically unchanged, use overlay directly
        # (helps against repositioning)
        try:
            base_crop = base_rgb.crop(bbox)
            out_crop = out_rgb.crop(bbox)
            d = ImageChops.difference(base_crop, out_crop)
            dm = ImageStat.Stat(d).mean
            diff_mean = sum(dm) / max(1.0, len(dm))
        except Exception:
            diff_mean = 999.0

        if diff_mean < 0.35:
            # use the provided overlay (already warped to perspective) for ink-like modes
            try:
                overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
                ov_rgb = overlay_img.convert("RGB")
                ov_a = overlay_img.split()[-1]
                ov_mask = ImageChops.multiply(combined, ov_a)
                final = base_rgb.copy()
                final.paste(ov_rgb, mask=ov_mask)
                out_img = final
            except Exception:
                final = base_rgb.copy()
                final.paste(out_rgb, mask=combined)
                out_img = final
        else:
            final = base_rgb.copy()
            final.paste(out_rgb, mask=combined)
            out_img = final

    except Exception:
        # if anything goes wrong, keep Gemini output as-is
        pass
    # FINAL ENFORCEMENT (V2 PARITY LOCK):
    # Never allow changes outside the intended PRINT SHAPE.
    # Prefer the warped overlay alpha (design shape after cutout + warp),
    # fallback to quad polygon if overlay is unavailable.
    try:
        from PIL import ImageDraw
        base_rgb_final = base_img.convert("RGB")
        out_rgb_final = out_img.convert("RGB") if out_img is not None else base_rgb_final

        # 1) Prefer overlay alpha mask (most strict; removes any "plate"/background inside quad)
        pm = None
        try:
            _ov_final = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
            a = _ov_final.split()[-1]
            if a.size != base_rgb_final.size:
                a = a.resize(base_rgb_final.size, Image.NEAREST)
            pm = a
        except Exception:
            pm = None

        # 2) Fallback: build polygon mask from inp.quad (4 points)
        if pm is None:
            q = getattr(inp, "quad", None) or []
            pts = []
            try:
                if isinstance(q, (list, tuple)) and len(q) == 4 and isinstance(q[0], (list, tuple)):
                    pts = [(float(p[0]), float(p[1])) for p in q]
                elif isinstance(q, (list, tuple)) and len(q) == 8:
                    pts = [(float(q[i]), float(q[i+1])) for i in range(0, 8, 2)]
            except Exception:
                pts = []
            if len(pts) == 4:
                pm = Image.new("L", base_rgb_final.size, 0)
                dr = ImageDraw.Draw(pm)
                dr.polygon(pts, fill=255)

        if pm is not None:
            final = base_rgb_final.copy()
            final.paste(out_rgb_final, mask=pm)
            out_img = final
    except Exception:
        pass
    # --- /POST-PROCESS ---
    result_url = _save_png(out_img)
    thumb_url = _save_png(_make_thumb(out_img, 420))

    return {
        "resultUrl": result_url,
        "thumbUrl": thumb_url,
        "w": out_img.width,
        "h": out_img.height,
        "mode": inp.mode,
        "ai": True,
        "model": image_model,
    }