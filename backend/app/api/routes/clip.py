from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
import tempfile
import subprocess
import math
import base64
import json
import re
import os
import io
import mimetypes
from urllib.parse import urlparse
from uuid import uuid4

from PIL import Image, ImageDraw

from app.core.config import settings
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url
from app.engine.gemini_rest import post_generate_content

router = APIRouter()


class ClipImageIn(BaseModel):
    sceneId: str
    prompt: str
    style: str | None = "default"
    width: int | None = 1024
    height: int | None = 1024
    refs: "ClipImageRefsIn | None" = None
    sceneText: str | None = None


class ClipImageRefsIn(BaseModel):
    character: list[str] = Field(default_factory=list)
    location: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    propAnchorLabel: str | None = None
    sessionCharacterAnchor: str | None = None
    sessionLocationAnchor: str | None = None
    sessionStyleAnchor: str | None = None


class AudioSliceIn(BaseModel):
    sceneId: str
    t0: float
    t1: float
    audioUrl: str


def _ensure_assets_dir() -> None:
    ensure_static_dirs()


def _asset_url(filename: str) -> str:
    return asset_url(filename)


def _save_bytes_as_asset(raw: bytes, ext: str = "png") -> str:
    _ensure_assets_dir()
    ext = (ext or "png").lower().replace(".", "")
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        ext = "png"
    filename = f"clip_scene_{uuid4().hex}.{ext}"
    fpath = os.path.join(str(ASSETS_DIR), filename)
    with open(fpath, "wb") as f:
        f.write(raw)
    return _asset_url(filename)


def _resolve_audio_asset_path(audio_url: str) -> str | None:
    if not audio_url:
        return None

    parsed = urlparse(audio_url)
    path = parsed.path
    if path.startswith("/static/assets/"):
        filename = os.path.basename(path[len("/static/assets/"):])
    elif path.startswith("/assets/"):
        filename = os.path.basename(path[len("/assets/"):])
    else:
        return None

    if not filename:
        return None

    base = os.path.splitext(filename)[0]
    if not base:
        return None

    dirs = [ASSETS_DIR]
    names = [filename, base, f"{base}.mp3", f"{base}.wav", f"{base}.ogg", f"{base}.m4a"]
    seen = set()
    candidates: list[str] = []
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if p in seen:
                continue
            seen.add(p)
            candidates.append(p)

    for p in candidates:
        if os.path.isfile(p):
            return p

    return None


def _ffmpeg_audio_slice(input_path: str, output_path: str, t0: float, t1: float) -> tuple[bool, str]:
    dur = max(0.0, t1 - t0)
    if dur < 0.05:
        dur = 0.05

    first_cmd = [
        "ffmpeg", "-y",
        "-ss", str(t0),
        "-to", str(t1),
        "-i", input_path,
        "-c", "copy",
        output_path,
    ]
    try:
        first = subprocess.run(first_cmd, capture_output=True, text=True)
        if (
            first.returncode == 0
            and os.path.isfile(output_path)
            and os.path.getsize(output_path) > 1024
        ):
            return True, ""

        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(t0),
            "-t", str(dur),
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "192k",
            output_path,
        ]
        fallback = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if (
            fallback.returncode == 0
            and os.path.isfile(output_path)
            and os.path.getsize(output_path) > 1024
        ):
            return True, ""

        err = (fallback.stderr or first.stderr or "ffmpeg_failed").strip()
        return False, err[:500]
    except FileNotFoundError:
        return False, "ffmpeg_missing_install_and_add_to_PATH"


def _debug_audio_slice(audio_url: str, resolved_path: str | None) -> None:
    if (settings.PS_ENV or "").lower() != "dev":
        return

    candidate_debug = []
    parsed = urlparse(audio_url or "")
    path = parsed.path or ""
    if path.startswith("/static/assets/"):
        filename = os.path.basename(path[len("/static/assets/"):])
        base = os.path.splitext(filename)[0]
    elif path.startswith("/assets/"):
        filename = os.path.basename(path[len("/assets/"):])
        base = os.path.splitext(filename)[0]
    else:
        filename = ""
        base = ""

    if filename and base:
        dirs = [ASSETS_DIR]
        names = [filename, base, f"{base}.mp3", f"{base}.wav", f"{base}.ogg", f"{base}.m4a"]
        seen = set()
        for d in dirs:
            for n in names:
                p = os.path.join(d, n)
                if p in seen:
                    continue
                seen.add(p)
                candidate_debug.append(p)

    print("AUDIO SLICE DEBUG")
    print("audioUrl:", audio_url)
    print("resolved path:", resolved_path)
    print("ASSETS_DIR:", str(ASSETS_DIR))
    print("candidate paths (first 10):")
    for p in candidate_debug[:10]:
        print(" -", p, "exists=", os.path.isfile(p))


def _mock_scene_image(scene_id: str, width: int, height: int) -> str:
    _ensure_assets_dir()
    w = max(256, min(2048, int(width or 1024)))
    h = max(256, min(2048, int(height or 1024)))
    img = Image.new("RGB", (w, h), color=(44, 48, 58))
    draw = ImageDraw.Draw(img)
    text = f"MOCK\n{scene_id or 'scene'}"
    draw.multiline_text((32, 32), text, fill=(230, 235, 245), spacing=8)
    filename = f"clip_scene_mock_{uuid4().hex}.png"
    img.save(os.path.join(str(ASSETS_DIR), filename), format="PNG")
    return _asset_url(filename)


def _decode_gemini_image(resp: dict) -> tuple[bytes, str] | None:
    try:
        for cand in (resp.get("candidates") or []):
            content = (cand or {}).get("content") or {}
            for part in (content.get("parts") or []):
                inline = part.get("inlineData") or {}
                b64 = inline.get("data")
                mime = (inline.get("mimeType") or "image/png").lower()
                if isinstance(b64, str) and b64:
                    raw = base64.b64decode(b64)
                    ext = "jpg" if "jpeg" in mime or "jpg" in mime else "png"
                    return raw, ext
    except Exception:
        return None
    return None


def _normalize_ref_list(items, max_items: int = 8) -> list[str]:
    out = []
    if not items:
        return out
    for it in items:
        if isinstance(it, str):
            url = str(it).strip()
        elif isinstance(it, dict):
            url = str(it.get("url") or "").strip()
        else:
            url = str(getattr(it, "url", "") or "").strip()
        if url:
            out.append(url)
    return out[:max_items]


def _clean_anchor_label(label: str | None) -> str:
    v = str(label or "").strip()
    v = re.sub(r"\s+", " ", v)
    return v[:120]


def _build_prop_anchor(label: str | None) -> dict | None:
    cleaned = _clean_anchor_label(label)
    if not cleaned:
        return None
    return {
        "label": cleaned,
        "source": "ref",
    }


def _planner_input_signature(*, character_refs: list[str], location_refs: list[str], style_refs: list[str], props_refs: list[str], text: str, audio_url: str, mode: str, scenario_key: str, shoot_key: str, style_key: str, freeze_style: bool, want_lipsync: bool) -> str:
    signature_payload = {
        "characterRefs": character_refs,
        "locationRefs": location_refs,
        "styleRefs": style_refs,
        "propsRefs": props_refs,
        "text": str(text or "").strip(),
        "audioUrl": str(audio_url or "").strip(),
        "mode": str(mode or "").strip(),
        "settings": {
            "scenarioKey": str(scenario_key or "").strip(),
            "shootKey": str(shoot_key or "").strip(),
            "styleKey": str(style_key or "").strip(),
            "freezeStyle": bool(freeze_style),
            "wantLipSync": bool(want_lipsync),
        },
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _infer_prop_anchor_label(props_images: list[dict], api_key: str, model_used: str) -> str:
    if not props_images:
        return ""
    prompt = (
        "You must identify one single object shown across all reference photos. "
        "Treat all photos as different angles/details of the SAME object. "
        "Return STRICT JSON only: {\"label\":\"...\"}. "
        "Label must be short, stable, concrete, in English (2-6 words), no punctuation, no alternatives. "
        "If uncertain, output a stable fallback generic object label."
    )
    parts = [{"text": prompt}, *props_images]
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    resp = post_generate_content(api_key, model_used, body, timeout=60)
    raw = _extract_gemini_text(resp if isinstance(resp, dict) else {})
    parsed = _parse_json_from_text(raw)
    label = ""
    if isinstance(parsed, dict):
        label = _clean_anchor_label(parsed.get("label"))
    if not label:
        label = "anchored reference object"
    return label


def _enforce_prop_anchor_text(text: str, prop_anchor_label: str, *, lang: str) -> str:
    clean_text = str(text or "").strip()
    label = _clean_anchor_label(prop_anchor_label)
    if not label:
        return clean_text

    if lang == "ru":
        anchor_phrase = f"тот же предмет из референса ({label})"
        conflict_terms = [
            r"equipment\s+bag",
            r"generic\s+equipment",
            r"toolbox",
            r"backpack",
            r"\bbag\b",
            r"рюкзак",
            r"сумк[аеиу]",
            r"ящик\s+с\s+инструментами",
        ]
    else:
        anchor_phrase = f"the {label} from reference"
        conflict_terms = [
            r"equipment\s+bag",
            r"generic\s+equipment",
            r"toolbox",
            r"backpack",
            r"\bbag\b",
        ]

    out = clean_text
    for pattern in conflict_terms:
        out = re.sub(pattern, anchor_phrase, out, flags=re.I)

    if re.search(re.escape(label), out, flags=re.I) or re.search(r"from\s+reference|из\s+референса", out, flags=re.I):
        return out.strip()

    if not out:
        return anchor_phrase

    suffix = f" В кадре остаётся {anchor_phrase}." if lang == "ru" else f" Keep {anchor_phrase} visible."
    return (out + suffix).strip()


def _guess_image_mime(url: str, headers: dict, raw: bytes) -> str:
    header_mime = str((headers or {}).get("Content-Type") or "").split(";")[0].strip().lower()
    if header_mime.startswith("image/"):
        return header_mime

    guessed, _ = mimetypes.guess_type(url or "")
    guessed = (guessed or "").lower()
    if guessed.startswith("image/"):
        return guessed

    try:
        fmt = (Image.open(io.BytesIO(raw)).format or "").lower()
    except Exception:
        fmt = ""
    if fmt == "jpeg":
        return "image/jpeg"
    if fmt:
        return f"image/{fmt}"
    return "image/jpeg"


def _load_reference_image_inline(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.content
        if not raw:
            return None
        mime = _guess_image_mime(url, dict(r.headers), raw)
        return {
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        }
    except Exception:
        return None


class RefUrlItem(BaseModel):
    url: str


class BrainRefsIn(BaseModel):
    character: list[RefUrlItem] = []
    location: list[RefUrlItem] = []
    props: list[RefUrlItem] = []
    style: RefUrlItem | list[RefUrlItem] | None = None
    propAnchorLabel: str | None = None


class BrainIn(BaseModel):
    audioUrl: str | None = None
    text: str | None = None
    mode: str | None = None

    # brain settings (optional)
    scenarioKey: str | None = None   # e.g. "beat_rhythm" | "song_meaning"
    shootKey: str | None = None      # e.g. "cinema"
    styleKey: str | None = None      # e.g. "realism"
    freezeStyle: bool | None = None

    # refs (urls)
    refs: BrainRefsIn | None = None
    propAnchorLabel: str | None = None
    characterRefs: list[RefUrlItem] | None = None
    character_refs: list[str] | None = None
    locationRefs: list[RefUrlItem] | None = None
    propsRefs: list[RefUrlItem] | None = None
    styleRef: RefUrlItem | None = None

    # legacy single-url refs (backward compatibility)
    refCharacter: str | None = None
    refLocation: str | None = None
    refStyle: str | None = None
    refItems: str | None = None

    # informational (optional)
    audioType: str | None = None     # "song" | "bg"
    textType: str | None = None      # "lyrics" | "story" | "notes"
    wantLipSync: bool | None = None


def _build_session_world_anchors(*, text: str, character_refs: list[str], location_refs: list[str], style_refs: list[str], style_key: str) -> dict[str, str]:
    text_l = (text or "").strip().lower()
    style_hint = (style_key or "").strip()

    world_cue_to_anchor = [
        ("stage", "a live performance stage environment with visible rigging, stage floor, and audience-facing orientation"),
        ("concert", "a live concert venue with performance staging, crowd space, and show-ready production design"),
        ("club", "an intimate music club venue with performance area, audience zone, and nightlife atmosphere"),
        ("theater", "a theater performance venue with a formal stage, controlled house lighting, and audience seating"),
        ("audience", "a performance venue world with audience space, stage focus, and event atmosphere"),
        ("spotlight", "a stage-oriented performance world where lighting rigs and spotlight focus define the environment"),
        ("crowd", "a live event venue with crowd area, performer focal point, and concert-like staging"),
        ("factory", "an industrial factory environment with heavy structural materials and utilitarian spatial design"),
        ("warehouse", "a large warehouse environment with raw industrial textures and open utilitarian layout"),
        ("forest", "a forest environment with natural vegetation, trees, and layered outdoor depth"),
        ("beach", "a beachside environment with open shoreline space, coastal textures, and horizon depth"),
        ("subway", "an urban subway environment with platforms, transit architecture, and underground infrastructure"),
        ("rooftop", "a rooftop environment above the city with open skyline views and elevated urban context"),
        ("church", "a church interior environment with sacred architecture and formal spatial symmetry"),
        ("bar", "a bar or lounge environment with seating, counter zones, and social nightlife mood"),
        ("studio", "a controlled studio environment for performance or filming with production equipment context"),
        ("alley", "a narrow city alley environment with enclosed urban walls and gritty street texture"),
        ("street", "an urban street environment with road geometry, surrounding buildings, and city circulation"),
        ("city", "a city environment with urban density, architecture, and active metropolitan context"),
        ("desert", "a desert environment with arid terrain, open horizon, and dry atmospheric conditions"),
        ("mountain", "a mountain environment with elevated terrain, expansive natural scale, and rugged topography"),
        ("office", "an office environment with workspaces, desks, and structured professional interior layout"),
        ("school", "a school environment with classroom/campus context and academic institutional design"),
        ("hospital", "a hospital environment with clinical infrastructure, medical spaces, and sterile interior language"),
        ("room", "an interior room environment defined by enclosed architecture and localized scene staging"),
    ]
    style_cue_to_anchor = [
        ("dimly lit", "dim, low-key lighting with selective visibility and restrained exposure"),
        ("single spotlight", "single-spotlight performance lighting with strong subject isolation and high contrast"),
        ("spotlight", "spotlight-driven lighting mood with strong subject focus and controlled falloff"),
        ("concert lighting", "concert-style dynamic stage lighting with performance-driven highlights and atmosphere"),
        ("stage lights", "stage-lighting visual language with directional beams and show-like illumination rhythm"),
        ("club lighting", "club-style lighting atmosphere with nightlife contrast and color-accent illumination"),
        ("neon", "neon-accented visual style with saturated practical glows and urban nighttime energy"),
        ("smoky", "smoky atmospheric look with suspended particulates and softened depth transitions"),
        ("warm light", "warm-toned lighting palette with amber highlights and inviting contrast"),
        ("cold light", "cool-toned lighting palette with blue/cyan bias and crisp emotional distance"),
        ("moody", "moody cinematic styling with controlled contrast, emotional shadows, and restrained brightness"),
        ("cinematic", "cinematic visual language with intentional contrast, composition, and narrative lighting"),
        ("dark atmosphere", "dark atmospheric styling with low-key exposure and dramatic tonal separation"),
        ("dramatic lighting", "dramatic lighting with strong chiaroscuro, shaped highlights, and emotional contrast"),
        ("fog", "fog-rich atmosphere with volumetric depth and softened distant detail"),
        ("haze", "haze-based atmosphere with gentle diffusion, light bloom, and layered depth"),
    ]

    text_world_anchor = ""
    text_style_anchor = ""
    for cue, anchor in world_cue_to_anchor:
        if cue in text_l:
            text_world_anchor = anchor
            break
    for cue, anchor in style_cue_to_anchor:
        if cue in text_l:
            text_style_anchor = anchor
            break

    if character_refs:
        character_anchor = "same exact person identity as character reference images"
    else:
        if any(word in text_l for word in ["woman", "girl", "her", "she"]):
            character_anchor = "a solitary woman in her early 30s with short dark hair, expressive eyes, and a dark winter coat"
        else:
            character_anchor = "a solitary man in his early 30s with short dark hair and a trimmed beard, wearing a dark winter coat"

    if location_refs:
        location_anchor = "same exact world/location identity as location reference images"
    else:
        location_anchor = text_world_anchor or "a narrow European winter street with old brick buildings and wet cobblestone pavement"

    if style_refs:
        style_anchor = "same exact style identity from style reference images: weather, season, palette, atmosphere, and lighting mood"
    else:
        style_anchor = text_style_anchor or style_hint or "cold cinematic realism, muted winter palette, overcast sky, wet reflective pavement, atmospheric haze"

    return {
        "character": character_anchor,
        "location": location_anchor,
        "style": style_anchor,
    }


def _inject_session_world_anchors(prompt: str, anchors: dict[str, str]) -> str:
    base = (prompt or "").strip()
    anchor_text = (
        "SESSION WORLD ANCHORS:\n"
        f"Character anchor: {anchors.get('character', '')}\n"
        f"Location anchor: {anchors.get('location', '')}\n"
        f"Style anchor: {anchors.get('style', '')}\n\n"
        "These anchors define the persistent identity of the clip world and must remain unchanged across all frames."
    )
    return f"{base}\n\n{anchor_text}" if base else anchor_text


def _extract_gemini_text(resp: dict) -> str:
    try:
        cands = resp.get("candidates") or []
        if not cands:
            return ""
        content = (cands[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _parse_json_from_text(s: str) -> dict | None:
    if not s:
        return None

    def _balance_json_tail(chunk: str) -> str:
        stack = []
        in_string = False
        escape = False
        for ch in chunk:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack and ((ch == "]" and stack[-1] == "[") or (ch == "}" and stack[-1] == "{")):
                    stack.pop()
        if in_string:
            chunk += '"'
        for opener in reversed(stack):
            chunk += "]" if opener == "[" else "}"
        return chunk

    s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s.strip(), flags=re.M)
    s2 = re.sub(r"\s*```\s*$", "", s2, flags=re.M)
    m = re.search(r"\{[\s\S]*\}", s2)
    chunks_to_try = [m.group(0)] if m else []

    first_brace = s2.find("{")
    if first_brace >= 0:
        tail = s2[first_brace:]
        last_closed = max(tail.rfind("}"), tail.rfind("]"))
        if last_closed > 0:
            chunks_to_try.append(tail[: last_closed + 1])
        chunks_to_try.append(tail)

    seen = set()
    for chunk in chunks_to_try:
        if not chunk or chunk in seen:
            continue
        seen.add(chunk)
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk), _balance_json_tail(chunk)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def _combined_error_text(resp: dict | None) -> str:
    if not isinstance(resp, dict):
        return ""
    parts = [
        resp.get("text"),
        resp.get("error"),
        resp.get("detail"),
    ]
    out = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            out.append(part)
        else:
            out.append(json.dumps(part, ensure_ascii=False))
    return "\n".join([x for x in out if x]).strip()


def _is_model_unsupported_error(text: str) -> bool:
    s = (text or "").lower()
    needles = [
        "not found for api version",
        "not supported for generatecontent",
        "model not found",
    ]
    return any(n in s for n in needles)


def _pick_fallback_model(model_used: str | None) -> str:
    model = (model_used or "").strip()
    for candidate in ("gemini-2.5-flash", "gemini-2.0-flash"):
        if candidate and candidate != model:
            return candidate
    return "gemini-2.5-flash"


def get_audio_duration(url: str) -> float:
    """Получаем длительность аудио через ffprobe"""
    try:
        if os.path.isfile(url):
            path = url
            temp_path = None
        else:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(r.content)
                path = f.name
                temp_path = path

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        dur = float((result.stdout or "").strip())
        if math.isfinite(dur) and dur > 0:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return float(dur)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return 30.0
    except Exception:
        return 30.0


def _probe_audio_duration(path: str) -> float | None:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        dur = float((result.stdout or "").strip())
        if math.isfinite(dur) and dur > 0:
            return float(dur)
    except Exception:
        return None
    return None


def _load_audio_for_planner(audio_url: str | None) -> tuple[float, bytes | None, str, dict]:
    duration: float | None = None
    audio_mime = "audio/mpeg"
    debug = {
        "inputAudioUrl": audio_url or None,
        "resolvedPath": None,
        "audioBytesFound": False,
        "audioBytesSource": "none",
        "audioMime": audio_mime,
        "durationSec": None,
        "durationSource": "unknown",
        "audioLoadError": None,
        "hint": "",
    }

    if not audio_url:
        debug["durationSource"] = "default_no_audio"
        debug["hint"] = "audio_url_missing"
        return 30.0, None, audio_mime, debug

    resolved_path = _resolve_audio_asset_path(audio_url)
    if resolved_path and os.path.isfile(resolved_path):
        debug["resolvedPath"] = resolved_path
        ext = (os.path.splitext(resolved_path)[1] or "").lower()
        if ext == ".wav":
            audio_mime = "audio/wav"
        elif ext == ".ogg":
            audio_mime = "audio/ogg"
        elif ext == ".m4a":
            audio_mime = "audio/mp4"
        duration = _probe_audio_duration(resolved_path)
        if duration is not None:
            debug["durationSec"] = duration
            debug["durationSource"] = "local_ffprobe"
        try:
            with open(resolved_path, "rb") as f:
                audio_bytes = f.read()
            if audio_bytes:
                debug["audioBytesFound"] = True
                debug["audioBytesSource"] = "local_asset"
                debug["audioMime"] = audio_mime
                debug["hint"] = "audio_loaded_from_local_asset"
                return float(duration or 30.0), audio_bytes, audio_mime, debug
        except Exception as e:
            debug["audioLoadError"] = f"local_asset_read_failed:{str(e)[:180]}"

    try:
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        header_mime = str(r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if header_mime:
            audio_mime = header_mime
        audio_bytes = r.content
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                probed = _probe_audio_duration(tmp_path)
                if probed is not None:
                    duration = probed
                    debug["durationSec"] = duration
                    debug["durationSource"] = "http_ffprobe"
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            debug["audioBytesFound"] = True
            debug["audioBytesSource"] = "http"
            debug["audioMime"] = audio_mime
            debug["hint"] = "audio_loaded_over_http"
            return float(duration or 30.0), audio_bytes, audio_mime, debug
    except Exception as e:
        debug["audioLoadError"] = f"http_audio_load_failed:{str(e)[:180]}"

    if duration is not None:
        debug["durationSec"] = duration
        if debug["durationSource"] == "unknown":
            debug["durationSource"] = "ffprobe_without_audio_bytes"
    else:
        debug["durationSource"] = "default_fallback"
    debug["hint"] = "audio_not_found_or_unreachable_planner_built_without_audio_bytes"
    return float(duration or 30.0), None, audio_mime, debug


def _validate_storyboard_timeline(duration: float, scenes: list[dict]) -> tuple[bool, str | None, list[str]]:
    if not scenes:
        return False, "scenes_empty", []
    tol_edge = 0.75
    tol_touch = 0.3
    max_gap = 0.75
    warnings: list[str] = []

    starts = [float(scene.get("start") or 0.0) for scene in scenes]
    if starts != sorted(starts):
        return False, "timeline_unsorted", warnings

    sorted_scenes = scenes

    for idx, scene in enumerate(sorted_scenes):
        start = float(scene.get("start") or 0.0)
        end = float(scene.get("end") or 0.0)
        if start < -tol_edge:
            return False, f"timeline_scene_start_oob_at_{idx}", warnings
        if end > float(duration) + tol_edge:
            return False, f"timeline_scene_end_oob_at_{idx}", warnings

    first_start = float(sorted_scenes[0].get("start") or 0.0)
    last_end = float(sorted_scenes[-1].get("end") or 0.0)

    if abs(first_start - 0.0) > tol_edge:
        return False, "timeline_bad_start", warnings
    if abs(last_end - float(duration)) > tol_edge:
        return False, "timeline_bad_end", warnings

    for idx in range(1, len(sorted_scenes)):
        prev_end = float(sorted_scenes[idx - 1].get("end") or 0.0)
        cur_start = float(sorted_scenes[idx].get("start") or 0.0)
        delta = cur_start - prev_end
        if delta < -tol_touch:
            return False, f"timeline_overlap_at_{idx}", warnings
        if delta > max_gap:
            return False, f"timeline_gap_at_{idx}", warnings
        if abs(delta) > tol_touch:
            warnings.append(f"timeline_micro_gap_at_{idx}")
    return True, None, warnings


def _format_audio_analysis_summary(audio_analysis: dict) -> str:
    duration = float(audio_analysis.get("duration") or 0.0)
    bpm = float(audio_analysis.get("bpm") or 0.0)
    downbeats = audio_analysis.get("downbeats") or []
    vocal_phrases = audio_analysis.get("vocalPhrases") or []
    energy_peaks = audio_analysis.get("energyPeaks") or []
    sections = audio_analysis.get("sections") or []

    section_lines = []
    for sec in sections[:6]:
        sec_type = str(sec.get("type") or "section")
        sec_start = float(sec.get("start") or 0.0)
        sec_end = float(sec.get("end") or 0.0)
        section_lines.append(f"{sec_type}({sec_start:.2f}-{sec_end:.2f})")

    phrase_lines = []
    for phr in vocal_phrases[:6]:
        p0 = float(phr.get("start") or 0.0)
        p1 = float(phr.get("end") or 0.0)
        phrase_lines.append(f"{p0:.2f}-{p1:.2f}")

    peak_lines = [f"{float(t):.2f}" for t in energy_peaks[:8]]

    summary = "\nAUDIO ANALYSIS:"
    summary += f"\nduration={duration:.2f}"
    summary += f"\nbpm={bpm:.0f}" if bpm > 0 else "\nbpm=0"
    summary += f"\ndownbeats={len(downbeats)}"
    summary += f"\nvocalPhrases={len(vocal_phrases)}"
    summary += f"\nenergyPeaks={len(energy_peaks)}"
    summary += "\nsections=" + (", ".join(section_lines) if section_lines else "none")
    if phrase_lines:
        summary += "\nvocalPhrases(first6):\n" + "\n".join(phrase_lines)
    if peak_lines:
        summary += "\nenergyPeaks(first8):\n" + "\n".join(peak_lines)
    return summary


def _fallback_plan(duration: float, text: str | None):
    scene_len = 5.0
    scene_count = max(1, math.ceil(duration / scene_len))
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()] if text else []
    chunks = []
    if lines:
        step = max(1, math.floor(len(lines) / scene_count))
        for i in range(scene_count):
            part = lines[i * step : (i + 1) * step]
            chunks.append(" ".join(part))
    else:
        for i in range(scene_count):
            chunks.append(f"Scene {i+1}")

    scenes = []
    t = 0.0
    for i in range(scene_count):
        ch = chunks[i] if i < len(chunks) else ""
        t1 = min(duration, t + scene_len)
        scenes.append({
            "id": f"s{i+1:02d}",
            "start": float(t),
            "end": float(t1),
            "why": "резервная нарезка по равным сегментам",
            "sceneText": ch,
            "imagePrompt": f"Кинематографичная сцена: {ch}",
            "videoPrompt": "Кинематографичное движение камеры, драматичный свет, зерно плёнки",
            "audioType": "mixed",
            "sceneType": "visual_rhythm",
            "hasVocals": False,
            "isLipSync": False,
            "lyricFragment": "",
            "timingReason": "резервная нарезка на равные по длительности отрезки",
            "beatAnchor": "bar_start",
            "performanceType": "cinematic_visual",
            "shotType": "wide",
        })
        t = t1
        if t >= duration:
            break
    # ensure last end == duration
    if scenes:
        scenes[-1]["end"] = float(duration)
    return scenes


def _normalize_scenes(duration: float, scenes: list[dict]) -> list[dict]:
    """Ensure scenes are valid and cover full duration."""
    out = []
    allowed_product_views = {"hero", "wide", "side", "detail", "interaction", "macro"}
    for i, s in enumerate(scenes or []):
        try:
            t0 = float(s.get("start", s.get("t0", 0.0)))
            t1 = float(s.get("end", s.get("t1", 0.0)))
        except Exception:
            continue
        if not (math.isfinite(t0) and math.isfinite(t1)):
            continue
        if t1 <= t0:
            continue
        audio_type = str(s.get("audioType") or "mixed")
        scene_type = str(s.get("sceneType") or "visual_rhythm")
        has_vocals = bool(s.get("hasVocals") is True)
        is_lipsync = bool(s.get("isLipSync") is True or s.get("lipSync") is True)
        lyric_fragment = str(s.get("lyricFragment") or "").strip()
        timing_reason = str(s.get("timingReason") or s.get("why") or "")

        performance_type = str(s.get("performanceType") or "cinematic_visual")
        shot_type = str(s.get("shotType") or "")
        product_view = str(s.get("productView") or "").strip().lower()
        if product_view not in allowed_product_views:
            product_view = ""

        wants_lipsync = is_lipsync or scene_type == "lipSync"
        missing_vocal_phrase = not lyric_fragment
        instrumental_slice = audio_type == "instrumental" or not has_vocals
        seg_duration = t1 - t0
        too_short_lipsync = seg_duration < 1.0
        short_with_lyric = seg_duration < 1.5 and bool(lyric_fragment)
        if wants_lipsync and (instrumental_slice or missing_vocal_phrase or too_short_lipsync or short_with_lyric):
            only_missing_phrase_issue = missing_vocal_phrase and audio_type != "instrumental" and has_vocals
            if not only_missing_phrase_issue:
                has_vocals = False
            is_lipsync = False
            scene_type = "vocal" if (audio_type != "instrumental" and has_vocals) else "visual_rhythm"
            performance_type = "cinematic_visual"
            if shot_type == "mouth_closeup":
                shot_type = "medium"
            if too_short_lipsync and missing_vocal_phrase:
                fallback_reason = "lipSync disabled: segment too short and lyricFragment is empty"
            elif short_with_lyric:
                fallback_reason = "lipSync disabled: segment too short for a coherent vocal phrase"
            else:
                fallback_reason = "lipSync disabled: vocal phrase not confirmed for this segment"
            timing_reason = f"{timing_reason}; {fallback_reason}" if timing_reason else fallback_reason

        normalized_scene = {
            "id": str(s.get("id") or f"s{i+1:02d}"),
            "start": round(t0, 2),
            "end": round(t1, 2),
            "why": str(s.get("why") or ""),
            "sceneText": str(s.get("sceneText") or ""),
            "imagePrompt": str(s.get("imagePrompt") or s.get("prompt") or s.get("sceneText") or ""),
            "videoPrompt": str(s.get("videoPrompt") or ""),
            "audioType": audio_type,
            "sceneType": scene_type,
            "hasVocals": has_vocals,
            "isLipSync": is_lipsync,
            "lyricFragment": lyric_fragment,
            "timingReason": timing_reason,
            "beatAnchor": str(s.get("beatAnchor") or ""),
            "performanceType": performance_type,
            "shotType": shot_type,
        }
        if product_view:
            normalized_scene["productView"] = product_view
        out.append(normalized_scene)
    if not out:
        return out
    # clamp and sort
    out.sort(key=lambda x: x["start"])
    # clamp to [0,duration]
    for s in out:
        s["start"] = max(0.0, min(float(duration), float(s["start"])))
        s["end"] = max(0.0, min(float(duration), float(s["end"])))
        if s["end"] <= s["start"]:
            s["end"] = min(float(duration), s["start"] + 0.5)
    # force first start 0 and last end duration (soft)
    out[0]["start"] = 0.0
    out[-1]["end"] = float(duration)
    # remove overlaps / make monotonic
    for i in range(1, len(out)):
        if out[i]["start"] < out[i-1]["end"]:
            out[i]["start"] = out[i-1]["end"]
            if out[i]["end"] <= out[i]["start"]:
                out[i]["end"] = min(float(duration), out[i]["start"] + 0.5)
    out[-1]["end"] = float(duration)
    return out


def _minimum_scene_count_for_repair(duration: float) -> int:
    if duration >= 60:
        return 10
    if duration >= 45:
        return 8
    if duration >= 30:
        return 7
    if duration >= 15:
        return 5
    return 3


def _validate_planner_scenes_quality(duration: float, scenario_key: str, scenes: list[dict]) -> dict:
    scene_count = len(scenes or [])
    empty_scene_text_count = 0
    empty_image_prompt_count = 0
    empty_video_prompt_count = 0
    empty_core_scene_count = 0

    for scene in scenes or []:
        scene_text = str(scene.get("sceneText") or "").strip()
        image_prompt = str(scene.get("imagePrompt") or "").strip()
        video_prompt = str(scene.get("videoPrompt") or "").strip()

        is_scene_text_empty = not scene_text
        is_image_prompt_empty = not image_prompt
        is_video_prompt_empty = not video_prompt

        if is_scene_text_empty:
            empty_scene_text_count += 1
        if is_image_prompt_empty:
            empty_image_prompt_count += 1
        if is_video_prompt_empty:
            empty_video_prompt_count += 1
        if is_scene_text_empty and is_image_prompt_empty and is_video_prompt_empty:
            empty_core_scene_count += 1

    warnings: list[str] = []
    rejected_reasons: list[str] = []
    scenario = (scenario_key or "").strip().lower()
    min_clip_scenes_for_repair = _minimum_scene_count_for_repair(duration) if scenario == "clip" else 0
    is_weak_clip_plan = bool(scenario == "clip" and scene_count < min_clip_scenes_for_repair)
    if scenario == "clip":
        if duration >= 12 and scene_count < 2:
            warnings.append("scene_count_below_min_for_12s")
        if duration >= 20 and scene_count < 3:
            warnings.append("scene_count_below_min_for_20s")
        if duration >= 30 and scene_count < 4:
            warnings.append("scene_count_below_min_for_30s")
        if scene_count < min_clip_scenes_for_repair:
            warnings.append(f"scene_count_below_repair_min_for_clip:{scene_count}<{min_clip_scenes_for_repair}")
        if is_weak_clip_plan:
            warnings.append("weak_clip_plan")

    if scene_count == 1:
        only = scenes[0]
        coverage = max(0.0, float(only.get("end") or 0.0) - float(only.get("start") or 0.0))
        only_scene_text = str(only.get("sceneText") or "").strip()
        only_image_prompt = str(only.get("imagePrompt") or "").strip()
        only_video_prompt = str(only.get("videoPrompt") or "").strip()
        only_core_empty = not only_scene_text and not only_image_prompt and not only_video_prompt

        if duration > 0 and (coverage / duration) >= 0.9 and only_core_empty:
            rejected_reasons.append("single_scene_covers_almost_entire_track")

    if empty_scene_text_count > 0:
        warnings.append("has_empty_sceneText")
    if empty_image_prompt_count > 0:
        warnings.append("has_empty_imagePrompt")
    if empty_video_prompt_count > 0:
        warnings.append("has_empty_videoPrompt")

    if scene_count == 0:
        rejected_reasons.append("empty_scenes")
    if scene_count > 0 and empty_core_scene_count > (scene_count / 2):
        rejected_reasons.append("more_than_half_scenes_empty_core_fields")

    rejected_reason = ",".join(rejected_reasons) if rejected_reasons else None
    return {
        "scenario": scenario,
        "sceneCount": scene_count,
        "emptySceneTextCount": empty_scene_text_count,
        "emptyImagePromptCount": empty_image_prompt_count,
        "emptyVideoPromptCount": empty_video_prompt_count,
        "emptyCoreSceneCount": empty_core_scene_count,
        "warnings": warnings,
        "rejectedReason": rejected_reason,
        "repairRetryUsed": False,
        "weakClipPlan": is_weak_clip_plan,
    }


def _build_planning_semantics(
    text: str,
    scenario_key: str,
    audio_type_hint: str,
    text_type_hint: str,
    want_lipsync: bool,
    character_refs: list[str],
    location_refs: list[str],
    props_refs: list[str],
    style_key: str,
) -> dict:
    text_l = (text or "").lower()
    style_l = (style_key or "").lower()
    hint_audio_l = (audio_type_hint or "").lower()
    hint_text_l = (text_type_hint or "").lower()

    product_keywords = [
        "прод", "товар", "описан", "аппарат", "product", "commercial", "sale", "selling", "welding",
    ]
    is_product_text = bool(text_l) and any(k in text_l for k in product_keywords)

    text_types = []
    if hint_text_l:
        text_types.append(hint_text_l)
    if is_product_text:
        text_types.extend(["commercial description", "product narrative"])
    if not text_types and text_l:
        text_types.append("story")

    has_song_vocals = hint_audio_l in {"song", "song_with_vocals", "vocals"} or want_lipsync
    audio_type = "song_with_vocals" if has_song_vocals else (hint_audio_l or "mixed")

    has_character = bool(character_refs)
    has_location = bool(location_refs)
    has_style = bool(style_key)
    product_ref_count = len(props_refs)
    product_mode = bool(product_ref_count and (is_product_text or "product" in hint_text_l or "commercial" in hint_text_l))

    props_role = "multi-angle product reference" if product_mode and product_ref_count > 1 else "generic props"
    mode_key = (scenario_key or "").strip().lower()
    if mode_key == "clip" and product_mode:
        mode_interpretation = "clip_product_performance"
    elif mode_key == "clip":
        mode_interpretation = "music_driven_visual_montage"
    else:
        mode_interpretation = "generic_storyboard"

    return {
        "textType": text_types,
        "audioType": audio_type,
        "storySource": "TEXT" if text_l else "AUDIO",
        "timingSource": "AUDIO" if mode_key == "clip" else "TEXT",
        "speechSource": "AUDIO" if has_song_vocals else ("TEXT" if text_l else "NONE"),
        "audioRole": ["emotion source", "rhythm source", "timing source"] if has_song_vocals else ["timing source"],
        "propsRole": props_role,
        "productMode": product_mode,
        "productRefCount": product_ref_count,
        "hasCharacter": has_character,
        "hasLocation": has_location,
        "hasStyle": has_style,
        "modeInterpretation": mode_interpretation,
        "styleApplication": "historical_world_modern_product" if "18" in style_l and product_mode else "default",
    }


@router.post("/clip/plan")
def clip_plan(payload: BrainIn):
    """Gemini-first clip planner: Gemini analyzes audio/text/refs and returns strict JSON storyboard."""
    text = (payload.text or "").strip()
    mode = (getattr(payload, "mode", None) or payload.scenarioKey or "clip").strip().lower() or "clip"

    duration, audio_bytes, audio_mime, audio_debug = _load_audio_for_planner(payload.audioUrl)



    refs_obj = payload.refs
    character_refs = _normalize_ref_list((refs_obj.character if refs_obj else None))
    if not character_refs:
        character_refs = _normalize_ref_list(payload.characterRefs)
    if not character_refs:
        character_refs = _normalize_ref_list(payload.character_refs)

    location_refs = _normalize_ref_list((refs_obj.location if refs_obj else None))
    if not location_refs:
        location_refs = _normalize_ref_list(payload.locationRefs)

    props_refs = _normalize_ref_list((refs_obj.props if refs_obj else None))
    if not props_refs:
        props_refs = _normalize_ref_list(payload.propsRefs)

    style_refs = []
    if refs_obj and getattr(refs_obj, "style", None):
        style_value = refs_obj.style
        if isinstance(style_value, list):
            style_refs = _normalize_ref_list(style_value)
        else:
            u = str(getattr(style_value, "url", "") or "").strip()
            if u:
                style_refs = [u]
    if not style_refs and payload.styleRef:
        u = str(getattr(payload.styleRef, "url", "") or "").strip()
        if u:
            style_refs = [u]

    if payload.refCharacter:
        character_refs.append(str(payload.refCharacter).strip())
    if not location_refs and payload.refLocation:
        location_refs = [str(payload.refLocation).strip()]
    if not props_refs and payload.refItems:
        props_refs = [str(payload.refItems).strip()]
    if not style_refs and payload.refStyle:
        style_refs = [str(payload.refStyle).strip()]

    character_refs = list(dict.fromkeys([url for url in character_refs if url]))[:8]
    location_refs = list(dict.fromkeys([url for url in location_refs if url]))[:8]
    props_refs = list(dict.fromkeys([url for url in props_refs if url]))[:8]
    style_refs = list(dict.fromkeys([url for url in style_refs if url]))[:1]

    scenario_key = (payload.scenarioKey or "").strip()
    shoot_key = (payload.shootKey or "").strip()
    style_key = (payload.styleKey or "").strip()
    freeze_style = bool(payload.freezeStyle)
    want_lipsync = bool(payload.wantLipSync)

    session_world_anchors = _build_session_world_anchors(
        text=text,
        character_refs=character_refs,
        location_refs=location_refs,
        style_refs=style_refs,
        style_key=style_key,
    )

    planner_input_signature = _planner_input_signature(
        character_refs=character_refs,
        location_refs=location_refs,
        style_refs=style_refs,
        props_refs=props_refs,
        text=text,
        audio_url=payload.audioUrl or "",
        mode=mode,
        scenario_key=scenario_key,
        shoot_key=shoot_key,
        style_key=style_key,
        freeze_style=freeze_style,
        want_lipsync=want_lipsync,
    )

    input_state_debug = {
        "characterRefCount": len(character_refs),
        "locationRefCount": len(location_refs),
        "styleRefCount": len(style_refs),
        "propsRefCount": len(props_refs),
        "textPresent": bool(text),
        "audioPresent": bool(payload.audioUrl),
        "mode": mode,
        "signature": planner_input_signature,
    }

    character_images = []
    for ref_url in character_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            character_images.append(inline_part)

    location_images = []
    for ref_url in location_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            location_images.append(inline_part)

    props_images = []
    for ref_url in props_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            props_images.append(inline_part)

    print("CLIP DEBUG character_refs:", character_refs)
    print("CLIP DEBUG attached character images:", len(character_images))
    print("CLIP DEBUG location_refs:", location_refs)
    print("CLIP DEBUG attached location images:", len(location_images))

    refs_debug = {
        "characterRefCount": len(character_refs),
        "characterImagesAttached": len(character_images),
        "locationRefCount": len(location_refs),
        "locationImagesAttached": len(location_images),
        "styleRefCount": len(style_refs),
        "propsRefCount": len(props_refs),
        "propsImagesAttached": len(props_images),
    }

    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "code": "GEMINI_API_KEY_MISSING",
                "detail": "Gemini API key is missing for clip planning",
                "plannerDebug": {
                    "inputState": input_state_debug,
                    "refsDebug": refs_debug,
                },
            },
        )

    prop_anchor_label = _clean_anchor_label(
        getattr(refs_obj, "propAnchorLabel", None) or getattr(payload, "propAnchorLabel", None)
    )
    prop_anchor_source = "payload" if prop_anchor_label else "fallback"
    if props_images and not prop_anchor_label:
        anchor_model = (getattr(settings, "GEMINI_VISION_MODEL", None) or "gemini-1.5-flash").strip()
        prop_anchor_label = _infer_prop_anchor_label(props_images, api_key, anchor_model)
        prop_anchor_source = "inferred" if prop_anchor_label else "fallback"

    prop_anchor = _build_prop_anchor(prop_anchor_label)
    if prop_anchor:
        prop_anchor["source"] = prop_anchor_source
    refs_debug["propAnchor"] = prop_anchor
    refs_debug["propAnchorLabel"] = prop_anchor_label or None
    refs_debug["propAnchorSource"] = prop_anchor_source

    style_anchor = (
        "season, weather, color palette and cinematic visual language must be taken directly from style reference images"
        if style_refs
        else ((payload.styleKey or "").strip() or "neutral cinematic realism")
    )
    lighting_anchor = (
        "light direction, softness, exposure and color temperature must match the lighting implied by style reference images"
        if style_refs
        else "environment-driven cinematic lighting derived from the location and world state"
    )
    location_anchor = (
        "architecture style, street geometry, paving materials and environmental aging must match location reference images"
        if location_refs
        else "coherent single-location environment"
    )
    environment_anchor = "weather, atmosphere, surface materials and environmental mood must remain stable across scenes"
    weather_anchor = (
        "weather state must be taken directly from style reference images and remain unchanged across scenes"
        if style_refs
        else "coherent stable weather state across scenes"
    )
    surface_anchor = (
        "ground/surface state must be taken directly from style reference images, including snow traces, wetness, reflections, and material condition"
        if style_refs
        else "coherent stable ground and surface condition across scenes"
    )
    has_visual_inputs = bool(audio_bytes or character_images or location_images or props_images)
    if has_visual_inputs:
        model_used = getattr(settings, "GEMINI_VISION_MODEL", None) or "gemini-1.5-flash"
    else:
        model_used = getattr(settings, "GEMINI_TEXT_MODEL", None) or "gemini-2.5-flash"
    model_used = model_used.strip()

    system_rules = f"""You are a professional music video director and editor.
Build the clip storyboard directly from audio/text/refs with strict continuity and rhythm logic.
Return ONLY valid JSON object, no markdown, no explanation, no code fences.

Hard rules:
- Analyze audio yourself: BPM, sections, vocal phrases, energy events.
- Cover full track from 0 to track.durationSec with no gaps and no overlap between scenes.
- Scene durations should be logical: 1-2 sec only for fast inserts, 2-4 sec common, 4-6 sec atmospheric.
- Scene boundaries should align with beat accents, section transitions, and vocal phrase boundaries.
- Do not invent random disconnected scenes.
- Maintain continuity: same character identity, same world/location logic, same style language.
- If refs are provided, refs are source of truth and have priority over free imagination.
- lipSyncText must be non-empty only when there is real vocal phrase in that scene.
- If no audio is available, still build coherent storyboard from text+refs.
- If no text/refs, still build coherent storyboard from audio only.

TEXT NODE PRIORITY RULE:

If TEXT is provided, it is the primary narrative source for the clip.

TEXT must define:

- what happens in each scene
- event progression across scenes
- scene-specific actions and interactions
- character intention
- emotional beats

AUDIO PRIORITY RULE:

AUDIO defines:

- timing
- rhythm
- intensity
- segmentation
- vocal emphasis

AUDIO must not overwrite or replace TEXT narrative content.
If both TEXT and AUDIO are present:

- TEXT defines story content
- AUDIO defines pacing and cut structure

NO GENERIC PLANNER REWRITE:

Do not rewrite TEXT narrative into generic cinematic fallback.
Do not collapse specific actions/events into neutral atmospheric portraits.
If TEXT specifies concrete actions, gestures, interactions, events, or dramatic progression,
preserve them explicitly in the scene list.

TEXT WORLD OVERRIDE RULE:

If location/style refs are absent, but TEXT explicitly defines a concrete world, venue, or environment,
derive the session world from TEXT instead of generic planner fallback world initialization.

TEXT STYLE OVERRIDE RULE:

If style refs are absent, but TEXT explicitly defines lighting, performance mood, venue atmosphere,
or visual styling cues, derive the style anchor from TEXT instead of generic fallback style.

GENERIC FALLBACK LIMIT:

Generic planner fallback world/style may only be used when user input does not define
a concrete world or concrete style cues.

MODE INTERPRETATION:

The storyboard engine supports two narrative modes:

CLIP MODE
ADVERTISEMENT MODE

Each mode has different narrative priorities.

CLIP MODE NARRATIVE RULE:

When mode = CLIP:

The storyboard must be driven primarily by the song lyrics and musical emotion.

Lyrics define:

- narrative meaning
- character motivation
- emotional transitions
- symbolic story moments

Music defines:

- rhythm
- pacing
- scene duration
- cut timing

Scenes must visualize the emotional meaning of the lyrics.

Each scene should interpret a phrase or emotional moment from the song.

Reference images only define:

- character appearance
- environment
- visual style

They must NOT override the lyrical narrative.

PROP ROLE IN CLIP MODE:

Props must not become the central narrative subject unless the lyrics explicitly reference the object.

Props are environmental elements used by the character.

Avoid product-style shots such as:

- isolated prop hero shots
- repeated product close-ups
- scenes where the prop dominates the composition

The focus must remain on the character and the story.

LYRIC INTERPRETATION RULE:

If lyrics text is available,
each scene should interpret a phrase or emotional fragment from the lyrics.

Scenes should represent:

- emotional meaning
- metaphor
- character decisions
- narrative progression

The storyboard must feel like a visual interpretation of the song.

ADVERTISEMENT MODE RULE:

When mode = ADVERTISEMENT:

The product becomes the central subject of the visual narrative.

Scenes must highlight:

- product visibility
- product interaction
- product functionality
- product hero moments

The product may appear prominently in multiple shots.

ADVERTISEMENT AUDIO MODE:

If advertisement audio narration is provided,
scene structure should follow the spoken marketing script.

Each scene should illustrate the feature or benefit described in the narration.

MODE PRIORITY SWITCH:

When mode = CLIP:

lyrics and emotional narrative override props and product focus.

When mode = ADVERTISEMENT:

product focus overrides lyrical or narrative interpretation.

MASTER WORLD CONTEXT (session-level):
- Character: from character refs if present
- Location: from location refs if present
- Style: from style refs if present
- Prop anchor: {prop_anchor_label or "none"}
All scenes must respect this world context.

SESSION WORLD ANCHORS:

Character anchor: {session_world_anchors["character"]}

Location anchor: {session_world_anchors["location"]}

Style anchor: {session_world_anchors["style"]}

Lighting anchor: {lighting_anchor}

Environment anchor: {environment_anchor}

Weather anchor: {weather_anchor}

Surface anchor: {surface_anchor}

All scenes must inherit these anchors.

Do not change these anchors between scenes.

STYLE-DEFINED ENVIRONMENT STATE:

If style reference images are present, they define:

- season
- weather state
- lighting mood
- color palette
- surface condition
- atmospheric mood

These style-defined environmental states must remain stable across all scenes.

Do not reinterpret or weaken them in later scenes.

If the style references imply winter snow, snow traces and cold winter atmosphere,
do not switch later scenes to generic wet cloudy weather without snow.

WEATHER STATE LOCK:

Weather must remain the same across the whole session unless explicitly changed by text.

If the style reference implies:

- snow
- winter cold
- overcast winter weather

then all scenes must preserve that same weather state.

Do NOT switch between:

- snow
- rain
- dry cloudy weather
- neutral weather

unless explicitly requested.

Weather continuity includes:

- presence/absence of snow
- snow traces on roofs and ground
- wetness level
- atmospheric coldness

SURFACE STATE LOCK:

Ground and surface conditions must remain visually consistent across scenes.

Maintain:

- same pavement material
- same wetness level
- same snow traces
- same reflection behavior
- same environmental wear

If the first scene shows wet cobblestone with snow traces,
later scenes must preserve that same surface logic.

GLOBAL ENVIRONMENT STATE:

Style reference images define the global environment state for the entire session.

This includes:

- season
- weather
- lighting mood
- color palette
- atmospheric conditions
- ground surface state

These properties must remain constant across all scenes.

Camera framing or shot type (wide, medium, close-up, macro) must NOT weaken these environmental constraints.

VISIBLE WEATHER LOCK:

If snow is part of the style-defined environment state,
snow must remain visible in every frame.

Snow accumulation or snow traces must remain visible on at least some of:

- ground edges
- rooftops
- pavement gaps
- horizontal surfaces
- street borders
- environmental surfaces

Do not reduce snowy winter state into generic wet cold weather.

Visible weather cues must remain present even in close-up and macro shots.

SUBJECT RELIGHTING RULE:

Character lighting must be derived entirely from the environment.

Do not preserve lighting baked into character reference images.

The generated subject must match the environment in:

- light direction
- color temperature
- ambient bounce light
- shadow softness
- exposure
- atmospheric haze

The character must not look studio-lit inside an outdoor cinematic environment.

PHYSICAL SUBJECT INTEGRATION:

The character must appear physically present inside the same world as the background.

Match:

- ambient depth
- edge contrast
- environmental color bounce
- surface reflections
- ground contact shadows
- local atmospheric perspective

Do not render the character as:
- pasted
- composited
- cut out
- separately lit
- cleaner than the environment

The subject must feel photographed in the same place and lighting conditions as the environment.

SUBJECT AND PROP ENVIRONMENT MATCH:

Character and prop must inherit the same environmental qualities as the scene.

Match:

- ambient haze
- dust
- smoke diffusion
- reflected dirty light
- floor color bounce
- local contrast softness
- environmental color contamination

Do not render the character or prop as cleaner, sharper, or separately lit than the environment.

Character and prop must feel physically present in the same air, same light, and same atmosphere as the scene.

NO CUTOUT / NO COMPOSITE LOOK:

Do not render the character or prop as:

- pasted
- composited
- cut out
- sticker-like
- separately exposed
- separately color-graded

They must feel captured inside the same environment,
with the same atmospheric depth and lighting logic.

Edges, contrast, color temperature, and softness must match the environment.

CHARACTER INTEGRATION LOCK:

The character must inherit:

- local ambient light
- environmental shadow softness
- atmospheric haze
- reflected floor color
- industrial / urban environmental contamination when applicable

Do not keep the character unnaturally clean or studio-like if the world is dusty, hazy, smoky, wet, dirty, snowy, or industrial.

The subject must feel photographed in the same environment,
not inserted afterward.

ATMOSPHERIC DEPTH RULE:

All visible elements must be affected by the same atmosphere.

Apply consistent haze, moisture, light scattering and atmospheric depth to:

- background
- character
- props

Do not keep the subject artificially crisp if the environment is soft, hazy, cold, wet, snowy, or diffuse-lit.

PROP SIZE CLASS LOCK:

The prop must belong to a stable real-world size class across all scenes.

The object must not change its physical class between frames.

Example:
A portable welding machine must remain portable-welder sized in every frame.

It must not become:
- oversized
- generator-sized
- miniaturized
- enlarged to fit composition
- distorted in apparent volume

The prop must remain physically plausible relative to the human body.

HARD PROP SIZE CLASS LOCK:

The prop belongs to a fixed real-world size class.

If the prop is a portable welding machine,
its physical class is compact carryable equipment,
approximately small-suitcase class.

It must never become:

- oversized
- generator-sized
- floor-machine sized
- enlarged to dominate the frame
- visually inflated to reveal more detail

The prop must keep the same physical size class across all scenes.

BODY-RELATIVE SCALE REFERENCE:

The prop must remain consistent relative to the human body.

Use stable body-relative references such as:

- hand grip
- shin height
- knee level
- lower leg size
- forearm carry scale

Do not change prop size class between:

- wide shots
- medium shots
- close-ups
- macro shots

Framing must not justify scaling the object larger or smaller.

ANATOMIC ANCHORING:

Object scale must be anchored relative to the human body.

Use stable body-relative proportions such as:

- knee height
- lower leg height
- hand-carryable size
- forearm / torso relation

The prop must keep the same body-relative scale across all shots.

Do not resize the object just because the framing changes.

MACRO CONTEXT LOCK:

In close-up or macro shots, the environment state must remain visible through the surface context.

If the wide-shot environment is snowy wet cobblestone street,
then close-up shots must preserve that same surface logic.

Macro shots must not forget:
- snow traces
- wetness
- pavement material
- winter environment cues

Close framing must not weaken global world continuity.

SESSION WORLD CONSISTENCY RULES:

All scenes must look like they belong to the SAME continuous world.

Do NOT treat scenes as independent illustrations.

Maintain continuity of:

- lighting conditions
- weather
- architectural language
- street geometry
- color palette
- environmental mood

Scenes must feel like different camera shots from the same film scene,
not different locations or times.

Lighting, atmosphere and architecture must remain consistent.

IMPORTANT:
Use the FIRST generated scene as the baseline world state.
All following scenes must inherit the same visual environment.

LIGHTING CONTINUITY:

All scenes must share the same lighting logic.

The first scene defines the lighting conditions.

If the first scene implies:

- overcast sky
- diffuse winter light
- cold color temperature

then all following scenes must maintain the SAME lighting conditions.

Do NOT switch to:

- sunny light
- warm sunset light
- studio lighting
- dramatic spotlight lighting

unless explicitly specified in refs or text.

Lighting must remain consistent across the whole storyboard.

Maintain:

- same shadow softness
- same exposure level
- same color temperature
- same light direction

LOCATION CONTINUITY LOCK:

If a location reference exists,
all scenes must appear to take place in the SAME environment.

Do not generate completely different streets,
cities or architectural styles.

Maintain continuity of:

- building materials
- architectural era
- street width
- pavement type
- environmental aging

Scenes must feel like different camera positions
within the same district or street world.

Camera angle may change,
but the environment must remain recognizably the same.

STRICT OBJECT LOCK:
- If props refs exist, they define one anchored prop identity for the whole session.
- Treat multiple props photos as different angles/details of the same object.
- Never reinterpret, replace, rename, generalize, or downgrade anchored prop identity.
- If scene wording conflicts with prop anchor identity, prop anchor identity wins.

PROP INTEGRATION LOCK:

The prop must be physically integrated into the scene.

Ensure that the prop:

- matches scene lighting
- matches scene exposure
- matches color temperature
- matches perspective
- matches scale relative to the character

The prop must NOT appear:

- pasted
- floating
- overly clean compared to the environment
- composited from another image

The prop must interact naturally with the character:

- realistic hand grip
- correct weight orientation
- correct physical contact

The prop must visually belong to the environment.

PROP INTEGRATION HARD LOCK:

The prop must be integrated into the environment with the same:

- ambient light
- shadow softness
- color temperature
- reflected floor color
- atmospheric softness
- dirt / haze / smoke context

Do not render the prop as a clean product render inside a dirty scene.

The prop must visually belong to the same world as the floor, air, and surrounding light.

ENVIRONMENTAL CONTAMINATION LOCK:

If the environment contains:

- dust
- smoke
- industrial haze
- wet reflections
- cold fog
- snow residue
- dirty floor bounce

then character and prop must inherit that environmental contamination visually.

They must not look isolated from the environmental conditions.

SOURCE PRIORITY RULES

Use the following source priority:

1. character reference images define exact person identity
2. location reference images define exact world/location identity
3. style reference images define season, weather, palette, atmosphere, and visual language
4. props reference images define exact object identity
5. scene text defines action, emotion, placement, interaction, and narrative meaning
6. audio defines timing, rhythm, energy, lipsync structure, and scene intensity
7. shoot mode defines camera language
8. styleKey is only a fallback when no style reference images are present
9. free imagination is allowed only when no higher-priority source defines that element

Higher-priority sources must never be overridden by lower-priority ones.

PER-SOURCE INTERPRETATION LOCKS

CHARACTER refs:
- text may change pose/action/emotion
- text must not change who the person is

LOCATION refs:
- text may change position within the same place
- text must not change the place itself

STYLE refs:
- text may change dramatic emphasis
- text must not replace season/weather/palette defined by style refs

PROPS refs:
- text may describe prop use/placement
- text must not rename or replace the object

PROP SCALE LOCK:

The prop must preserve the same real-world physical size class across all scenes.

The object must remain physically plausible relative to the human body.

Do not:

- enlarge it
- shrink it
- exaggerate it
- miniaturize it
- distort its real-world scale between shots

The prop must keep stable human-relative scale in every frame.

Example:
If the prop is a portable welding machine,
it must remain portable-welder sized in every scene,
not suitcase-sized in one scene and generator-sized in another.

PROP PHYSICAL CONSISTENCY:

Keep consistent:

- size relative to hands
- size relative to torso/legs
- grip logic
- weight impression
- handle/cable behavior
- ground contact behavior

The prop must not look weightless, oversized, undersized, or physically inconsistent between scenes.

If the prop is handheld,
its scale must remain realistically liftable by the character.

AUDIO:
- may control scene timing, pacing, emotion intensity, and lipsync
- must not redefine character/location/prop identity

SHOOT MODE:
- may control camera framing and movement language
- must not redefine world identity or character identity

STYLE KEY:
- use only if style refs are absent
- if style refs exist, style refs win

REFERENCE PRIORITY RULES

If character reference images are attached:
- Describe the SAME person from the reference images.
- Do not invent another man/woman.
- Do not change gender.
- Do not replace the outfit unless the story explicitly requests a wardrobe change.
- Do not invent a different hairstyle, age, or body type.
- All scenes must refer to the same exact person from the reference images.

If location reference images are attached:
- Describe the SAME environment from the reference images.
- Do not replace the setting with another room, street, or world.
- Architecture, mood, and setting must come from the reference images.

If reference images are attached, they override free imagination.

CHARACTER CONFLICT RESOLUTION

If scene text conflicts with character reference identity:
- Character refs always win.
- Conflicting text about gender, facial identity, age, hairstyle, clothing identity, or visible accessories must be ignored.
- Do not mix contradictory identity signals.
- Do not partially preserve incorrect text claims when they contradict character refs.
- Example: if text says "girl" but the character reference clearly shows a man, describe the man from reference and ignore the incorrect text identity label.

REFERENCE DETAIL ACCURACY

- Describe only details that are clearly supported by reference images.
- Do not invent accessories, wearable items, or carried objects that are not clearly visible.
- If a detail is ambiguous, do not state it as fact.
- Prefer omission over hallucination.

CLOTHING DETAIL INTERPRETATION RULES

- Hoodie drawstrings, garment cords, seams, folds, logo edges, shadows, and fabric details must not be misidentified as headphones, necklaces, wires, or accessories.
- Clothing details must remain clothing details unless clearly identifiable as separate objects.
- Logos must remain logos and must not be turned into separate accessories.

NO INVENTED ACCESSORIES RULE

Do not add headphones, glasses, jewelry, bags, backpacks, hats, watches, necklaces, or other accessories unless:
- They are clearly visible in reference images, or
- They are explicitly defined by a higher-priority reference node.

Scene text alone must not invent small visual accessories when character refs contradict or do not support them.

Response schema (all keys required):
{{
  "track": {{"durationSec": number, "bpm": number, "timeSignature": string, "energyProfile": string}},
  "sections": [{{"start": number, "end": number, "type": string, "energy": string}}],
  "vocalPhrases": [{{"start": number, "end": number, "text": string}}],
  "energyEvents": [{{"time": number, "type": string, "description": string}}],
  "scenes": [{{
    "id": "scene_001",
    "start": number,
    "end": number,
    "sceneType": string,
    "shotPurpose": string,
    "visualDescription": string,
    "visualPrompt": string,
    "lipSyncText": string,
    "camera": string,
    "motion": string,
    "reason": string
  }}]
}}

CHARACTER IDENTITY LOCK

If character reference images are provided:
- All images represent the SAME person
- This character must appear in every scene
- Do not redesign or replace the character
- Maintain identical facial identity
- Maintain same age, gender, hair, body type
- Treat these images as the source of truth

All scenes must describe the SAME character.

REFERENCE UNDERSTANDING RULES

Character reference images:
- All images depict the SAME person
- Use this character in every scene
- Do not change gender
- Do not change facial identity
- Clothing from reference images should remain consistent unless the story explicitly changes it
- Do not invent new hairstyles or body types
- Avoid generic invented phrases when references are specific

Location reference images:
- These images define the environment of the clip
- Scenes should take place in this world
- Architecture and atmosphere should match these references
- Avoid generic environment wording that ignores the reference details

STYLE REFERENCE RULES
- If style reference images are attached, they define season, atmosphere, palette, texture, weather, environment mood, and overall visual styling.
- Do not ignore style references.
- If style references indicate winter / snow / cold season / icy environment, scenes must reflect that visually.
- Do not default to neutral weather or generic city mood when style references specify a distinct season or atmosphere.

PROPS REFERENCE RULES
- If props reference images are attached, they define key objects of the scene.
- If there is only one props reference, treat it as a primary prop.
- Do not omit the prop when the scene can logically include it.
- Scene descriptions and visual prompts must explicitly mention the prop whenever relevant.
- Avoid treating props as optional decoration when they are clearly intended as key scene objects.

PROP PRIORITY RULES

If props reference images are attached:
- Props refs define exact object identity.
- Scene text may describe prop action, role, placement, or interaction.
- Scene text must not replace or rename the object into a different item.
- Object identity comes from refs, not from text.
- Example enforcement: if the prop ref is a welding machine, it must remain a welding machine and must not become a backpack, bag, suitcase, toolbox, speaker, generator, or generic equipment case.

If props refs are absent:
- Props may be inferred from scene text.

When references are present:
- Scene descriptions must explicitly describe the same man/woman from the reference images.
- Scene descriptions must explicitly describe the same environment from the reference images.
- Do not output generic placeholders like "young woman in a room" when references indicate a different person/place.
- When style refs exist, visualDescription and visualPrompt must explicitly reflect the style-defining season, atmosphere, weather, palette, and texture.
- When props refs exist, visualDescription and visualPrompt must explicitly mention and integrate the key prop in relevant scenes.
- When props refs exist, visualDescription and visualPrompt must preserve exact prop identity from refs and must never replace or rename the prop based on scene text.
- visualDescription and visualPrompt must not include invented small accessories or unsupported wardrobe details.
- If an accessory is uncertain, omit it and do not guess.

If reference images exist they override imagination.

IMPORTANT LANGUAGE ENFORCEMENT

All human-readable descriptive output must be written in Russian.

The following fields must ALWAYS be in Russian:
- visualDescription
- reason
- camera
- motion
- lipSyncText
- sections.type
- sections.energy
- vocalPhrases.text
- energyEvents.description

Only visualPrompt may remain in English because it is intended for image generation.

If any of the required descriptive fields are returned in English, the output is invalid.
"""

    user_input = {
        "mode": mode,
        "shootMode": payload.shootKey or payload.mode or "",
        "styleKey": payload.styleKey or "",
        "audioUrl": payload.audioUrl or "",
        "audioDurationHintSec": duration,
        "text": text,
        "refs": {
            "character": character_refs,
            "location": location_refs,
            "style": style_refs,
            "props": props_refs,
        },
        "propAnchor": prop_anchor,
    }

    parts = [{"text": system_rules}]

    if character_images:
        parts.append({"text": "Character reference images. All images depict the SAME main character."})
        parts.extend(character_images)

    if location_images:
        parts.append({"text": "Location reference images. These images define the world and environment of the clip."})
        parts.extend(location_images)

    if props_images:
        parts.append({"text": "Props reference images. All images depict the SAME single object identity from different angles/details."})
        parts.extend(props_images)
        parts.append({"text": f"Session prop anchor label: {prop_anchor_label}"})

    parts.append({"text": "Input payload:\n" + json.dumps(user_input, ensure_ascii=False)})

    if audio_bytes:
        parts.append({
            "inlineData": {
                "mimeType": audio_mime,
                "data": base64.b64encode(audio_bytes).decode("ascii")
            }
        })

    generation_config = {
        "temperature": 0.2,
        "responseMimeType": "application/json",
    }

    def _call_gemini(request_parts, model_name: str):
        body = {
            "contents": [{"role": "user", "parts": request_parts}],
            "generationConfig": generation_config,
        }
        resp = post_generate_content(api_key, model_name, body, timeout=120)
        raw = _extract_gemini_text(resp if isinstance(resp, dict) else {})
        parsed = _parse_json_from_text(raw)
        return resp, raw, parsed

    def _resolve_timeline_duration(plan: dict) -> float:
        track = plan.get("track") or {}
        try:
            gemini_track_duration = float(track.get("durationSec"))
            if not math.isfinite(gemini_track_duration) or gemini_track_duration <= 0:
                gemini_track_duration = None
        except Exception:
            gemini_track_duration = None

        duration_source = str(audio_debug.get("durationSource") or "")
        has_real_audio_duration = duration_source in {"local_ffprobe", "http_ffprobe", "ffprobe_without_audio_bytes"}
        if has_real_audio_duration and duration > 0:
            return float(duration)
        if gemini_track_duration is not None:
            return float(gemini_track_duration)
        if duration > 0:
            return float(duration)
        return 30.0

    def _validate_plan(plan: dict) -> tuple[bool, str | None]:
        if not isinstance(plan, dict):
            return False, "response_not_json_object"
        track = plan.get("track")
        scenes = plan.get("scenes")
        if not isinstance(track, dict):
            return False, "track_missing"
        if not isinstance(scenes, list) or not scenes:
            return False, "scenes_missing_or_empty"
        for idx, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                return False, f"scene_{idx}_not_object"
            try:
                start = float(scene.get("start"))
                end = float(scene.get("end"))
            except Exception:
                return False, f"scene_{idx}_invalid_time"
            if not (start < end):
                return False, f"scene_{idx}_start_not_less_than_end"
            visual_prompt = str(scene.get("visualPrompt") or "").strip()
            visual_desc = str(scene.get("visualDescription") or "").strip()
            if not (visual_prompt or visual_desc):
                return False, f"scene_{idx}_visual_empty"
        return True, None

    retry_used = False
    validation_warnings: list[str] = []
    validation_rejected_reason: str | None = None

    resp, raw_text, parsed = _call_gemini(parts, model_used)
    err_text = _combined_error_text(resp if isinstance(resp, dict) else {})
    if _is_model_unsupported_error(err_text):
        model_used = _pick_fallback_model(model_used)
        resp, raw_text, parsed = _call_gemini(parts, model_used)

    is_valid, reason = _validate_plan(parsed)
    if is_valid:
        timeline_duration = _resolve_timeline_duration(parsed)
        timeline_ok, timeline_reason, timeline_warnings = _validate_storyboard_timeline(timeline_duration, parsed.get("scenes") or [])
        validation_warnings.extend(timeline_warnings)
        if not timeline_ok:
            is_valid = False
            reason = timeline_reason

    if not is_valid:
        retry_used = True
        validation_warnings = []
        retry_parts = parts + [{"text": f"Previous output invalid ({reason}). Return ONLY one valid JSON object matching required schema."}]
        resp, raw_text, parsed = _call_gemini(retry_parts, model_used)
        is_valid, reason = _validate_plan(parsed)
        if is_valid:
            timeline_duration = _resolve_timeline_duration(parsed)
            timeline_ok, timeline_reason, timeline_warnings = _validate_storyboard_timeline(timeline_duration, parsed.get("scenes") or [])
            validation_warnings.extend(timeline_warnings)
            if not timeline_ok:
                is_valid = False
                reason = timeline_reason

    validation_rejected_reason = reason if not is_valid else None

    if not is_valid:
        err = _combined_error_text(resp if isinstance(resp, dict) else {}) or raw_text or reason or "invalid_gemini_json"
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "code": "CLIP_PLAN_VALIDATION_FAILED",
                "detail": str(err)[:1200],
                "modelUsed": model_used,
                "hint": reason,
                "plannerDebug": {
                    "audio": audio_debug,
                    "inputState": input_state_debug,
                    "model": {
                        "modelUsed": model_used,
                        "hasVisualInputs": has_visual_inputs,
                        "hasVisualRefsAttached": bool(character_images or location_images or props_images),
                    },
                    "refsDebug": refs_debug,
                    "validation": {
                        "scenario": mode,
                        "sceneCount": len((parsed or {}).get("scenes") or []),
                        "rejectedReason": validation_rejected_reason,
                        "repairRetryUsed": retry_used,
                        "warnings": validation_warnings,
                    },
                },
            },
        )

    plan = parsed
    track = dict(plan.get("track") or {})
    audio_duration = _resolve_timeline_duration(plan)
    track["durationSec"] = audio_duration
    scenes = plan.get("scenes") or []

    normalized_scenes = []
    for idx, s in enumerate(scenes):
        start = float(s.get("start"))
        end = float(s.get("end"))
        visual_prompt = str(s.get("visualPrompt") or "").strip()
        visual_desc = str(s.get("visualDescription") or "").strip()
        lip_sync_text = str(s.get("lipSyncText") or "").strip()
        lyric_fragment = str(s.get("lyricFragment") or lip_sync_text).strip()
        video_prompt = str(s.get("videoPrompt") or visual_prompt or visual_desc).strip()
        reason_text = str(s.get("reason") or "").strip()
        if prop_anchor_label:
            visual_prompt = _enforce_prop_anchor_text(visual_prompt, prop_anchor_label, lang="en")
            video_prompt = _enforce_prop_anchor_text(video_prompt, prop_anchor_label, lang="en")
            visual_desc = _enforce_prop_anchor_text(visual_desc, prop_anchor_label, lang="ru")
            reason_text = _enforce_prop_anchor_text(reason_text, prop_anchor_label, lang="ru")
        scene_type = str(s.get("sceneType") or "visual_rhythm").strip() or "visual_rhythm"
        normalized_scenes.append({
            **s,
            "id": str(s.get("id") or f"scene_{idx + 1:03d}"),
            "start": start,
            "end": end,
            "prompt": visual_prompt or visual_desc,
            "sceneText": visual_desc,
            "imagePrompt": visual_prompt,
            "videoPrompt": video_prompt,
            "why": reason_text,
            "sceneType": scene_type,
            "isLipSync": bool(lip_sync_text),
            "lipSyncText": lip_sync_text,
            "lyricFragment": lyric_fragment,
        })

    return {
        "ok": True,
        "engine": "gemini",
        "modelUsed": model_used,
        "fallbackUsed": False,
        "hint": None if audio_bytes else "plan_built_without_audio_bytes",
        "audioDuration": audio_duration,
        "track": track,
        "sections": plan.get("sections") if isinstance(plan.get("sections"), list) else [],
        "vocalPhrases": plan.get("vocalPhrases") if isinstance(plan.get("vocalPhrases"), list) else [],
        "energyEvents": plan.get("energyEvents") if isinstance(plan.get("energyEvents"), list) else [],
        "scenes": normalized_scenes,
        "propAnchor": prop_anchor,
        "sessionWorldAnchors": {
            "character": session_world_anchors["character"],
            "location": session_world_anchors["location"],
            "style": session_world_anchors["style"],
        },
        "plannerDebug": {
            "audio": audio_debug,
            "inputState": input_state_debug,
            "model": {
                "modelUsed": model_used,
                "hasVisualInputs": has_visual_inputs,
                "hasVisualRefsAttached": bool(character_images or location_images or props_images),
            },
            "refsDebug": refs_debug,
            "validation": {
                "scenario": mode,
                "sceneCount": len(normalized_scenes),
                "rejectedReason": validation_rejected_reason,
                "repairRetryUsed": retry_used,
                "warnings": validation_warnings,
            },
        },
    }


@router.post("/clip/image")
def clip_image(payload: ClipImageIn):
    scene_id = (payload.sceneId or "").strip()
    prompt = (payload.prompt or "").strip()
    style = (payload.style or "default").strip()

    if not scene_id:
        return JSONResponse(status_code=400, content={"ok": False, "code": "BAD_REQUEST", "hint": "sceneId_required"})
    if not prompt:
        return JSONResponse(status_code=400, content={"ok": False, "code": "BAD_REQUEST", "hint": "prompt_required"})

    width = max(256, min(2048, int(payload.width or 1024)))
    height = max(256, min(2048, int(payload.height or 1024)))
    scene_text = (payload.sceneText or "").strip()
    refs_obj = payload.refs
    character_refs = _normalize_ref_list(getattr(refs_obj, "character", None))
    location_refs = _normalize_ref_list(getattr(refs_obj, "location", None))
    style_refs = _normalize_ref_list(getattr(refs_obj, "style", None))
    props_refs = _normalize_ref_list(getattr(refs_obj, "props", None))
    prop_anchor_label = _clean_anchor_label(getattr(refs_obj, "propAnchorLabel", None))
    session_character_anchor = str(getattr(refs_obj, "sessionCharacterAnchor", "") or "").strip()
    session_location_anchor = str(getattr(refs_obj, "sessionLocationAnchor", "") or "").strip()
    session_style_anchor = str(getattr(refs_obj, "sessionStyleAnchor", "") or "").strip()

    character_images = []
    for ref_url in character_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            character_images.append(inline_part)

    location_images = []
    for ref_url in location_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            location_images.append(inline_part)

    style_images = []
    for ref_url in style_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            style_images.append(inline_part)

    props_images = []
    for ref_url in props_refs:
        inline_part = _load_reference_image_inline(ref_url)
        if inline_part:
            props_images.append(inline_part)

    prop_anchor_source = "payload" if prop_anchor_label else "fallback"
    if props_images and not prop_anchor_label:
        api_key_for_anchor = (settings.GEMINI_API_KEY or "").strip()
        anchor_model = (getattr(settings, "GEMINI_VISION_MODEL", None) or "gemini-1.5-flash").strip()
        if api_key_for_anchor:
            prop_anchor_label = _infer_prop_anchor_label(props_images, api_key_for_anchor, anchor_model)
            prop_anchor_source = "inferred" if prop_anchor_label else "fallback"

    refs_debug = {
        "characterRefCount": len(character_refs),
        "characterImagesAttached": len(character_images),
        "locationRefCount": len(location_refs),
        "locationImagesAttached": len(location_images),
        "styleRefCount": len(style_refs),
        "styleImagesAttached": len(style_images),
        "propsRefCount": len(props_refs),
        "propsImagesAttached": len(props_images),
        "propAnchorLabel": prop_anchor_label or None,
        "propAnchorSource": prop_anchor_source,
        "sessionCharacterAnchor": session_character_anchor or None,
        "sessionLocationAnchor": session_location_anchor or None,
        "sessionStyleAnchor": session_style_anchor or None,
    }

    style_anchor = (
        "season, weather, color palette and cinematic visual language must be taken directly from style reference images"
        if style_refs
        else ((style or "").strip() or "world-coherent cinematic realism")
    )
    lighting_anchor = (
        "light direction, softness, exposure and color temperature must match the lighting implied by style reference images"
        if style_refs
        else "environment-driven cinematic lighting derived from the location and world state"
    )
    location_anchor = (
        "architecture style, street geometry, paving materials and environmental aging must match location reference images"
        if location_refs
        else "coherent single-location environment"
    )
    environment_anchor = "weather, atmosphere, surface materials and environmental mood must remain stable across scenes"
    weather_anchor = (
        "weather state must be taken directly from style reference images and remain unchanged across scenes"
        if style_refs
        else "coherent stable weather state across scenes"
    )
    surface_anchor = (
        "ground/surface state must be taken directly from style reference images, including snow traces, wetness, reflections, and material condition"
        if style_refs
        else "coherent stable ground and surface condition across scenes"
    )

    has_visual_refs_attached = bool(character_images or location_images or style_images or props_images)
    # Normalize aspect label for prompt
    if height > width:
        aspect_ratio = "9:16"
    elif width > height:
        aspect_ratio = "16:9"
    else:
        aspect_ratio = "1:1"

    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        image_url = _mock_scene_image(scene_id, width, height)
        return {
            "ok": True,
            "sceneId": scene_id,
            "imageUrl": image_url,
            "engine": "mock",
            "hint": "no_gemini_key",
            "modelUsed": None,
            "refsDebug": refs_debug,
        }

    try:
        model = settings.GEMINI_IMAGE_MODEL or "gemini-2.5-flash-image-preview"

        system_prompt = (
            "You are a professional film director, cinematographer and visual production designer creating scenes for a cinematic music video. "
            "All scenes belong to the same continuous world and moment in time. "
            "GLOBAL WORLD CONTINUITY: preserve consistency for time of day, lighting conditions, sky brightness and color, street light intensity, ambient brightness, atmospheric haze/fog, and environmental color grading. "
            "WEATHER CONSISTENCY: maintain consistent snow/rain/fog/wind, snow coverage, wet/dry surfaces, atmospheric particles, and visible breath in cold air when applicable. "
            "LOCATION WORLD LOCK: keep architecture style, building proportions, street layout, materials/textures, signage style, and cultural environment as the same location. "
            "TIME PERIOD CONSISTENCY: architecture, vehicles, clothing, signage, and technology must remain in the same historical era. "
            "REFERENCE RULES: use all provided references as source of truth. Character references define the same person. Location references define the same world and architecture. Style references define weather, season, palette, atmosphere, and cinematic language. Props references define key objects. "
            "SOURCE PRIORITY RULES: Character references define who the person is. Location references define where the scene exists. Style references define season, weather, palette, atmosphere and visual language. Props references define exact object identity. Scene text defines action, emotion, narrative, interaction and placement. Visual prompt defines composition and shot content. Audio (if available) defines timing, rhythm, intensity, and lipsync energy. Shoot mode defines camera language. styleKey/style field is fallback style only when no style references exist. Free imagination is lowest priority and is allowed only when no higher-priority source defines that element. "
            "If any lower-priority input conflicts with higher-priority references, higher-priority references win. Higher-priority sources must never be overridden by lower-priority ones. "
            "Character refs cannot be overridden by scene text. Location refs cannot be overridden by scene text. Style refs cannot be overridden by scene text or generic visual prompt. Props refs cannot be overridden by scene text or generic visual prompt. "
            "WORLD LIGHTING PRIORITY: location/world/style references define the lighting model, atmosphere, palette, and environmental state. Character and prop references must adapt to that world state. Characters must not preserve reference-image lighting. Props must not preserve reference-image lighting. Props never define scene lighting. "
            "If style refs are absent, styleKey/style may influence the image as fallback. "
            "PROP PRIORITY RULES: if props reference images are attached, props refs define exact object identity. Scene text may describe how the object is used and where it is placed, but must not redefine, replace, or rename what the object is. If text conflicts with props refs, props refs win. If props refs are absent, text may define scene objects. "
            "STRICT OBJECT LOCK: The prop reference image defines the exact prop identity for this session. The prop must remain the same object across all scenes. Never reinterpret, replace, rename, generalize, or downgrade it into another object. "
            "CHARACTER IDENTITY LOCK: preserve facial structure, hairstyle, body proportions, skin tone, facial hair, gender, and age appearance. Do not redesign the person. "
            "CHARACTER DETAIL LOCK: preserve clothing type/colors, logos/brand marks, accessories, hairstyle, and carried items unless scene text explicitly changes wardrobe. "
            "PROP CONSISTENCY: maintain prop design, materials, dimensions, cables/attachments, brand markings, and wear/texture. Do not redesign props. "
            "PHYSICAL SCALE CONSISTENCY: maintain realistic human-relative scale; handheld objects must remain realistically liftable and consistent across scenes. "
            "BACKGROUND CHARACTER CONTROL: background people may appear but must remain subtle and non-distracting. "
            "WORLD DETAIL CONSISTENCY: keep vegetation, street furniture, parked vehicles, shop signs, decorations, snow accumulation, and ground texture consistent without random major changes. "
            "CINEMATIC STYLE CONSISTENCY: preserve coherent color grading, lighting mood, contrast, film atmosphere, and lens feel across scenes. "
            "SUBJECT INTEGRATION: character and objects must match environment in lighting direction/intensity, color temperature, reflections, ambient light, and shadows. "
            "GROUND CONTACT: ensure believable physical interaction with surfaces via contact shadows, footprints/compression, and wet reflections when appropriate. "
            "CINEMATIC ATMOSPHERE: use natural depth of field, atmospheric perspective, subtle haze/light scattering, realistic materials/textures, and filmic grading. Avoid plastic skin, flat lighting, and synthetic artifacts. "
            "FINAL RULE: generate ONE cinematic still frame that looks like real footage from a professional film production, never an artificial collage. "
            "WORLD CONSISTENCY: Maintain the same environment, lighting and visual style as previous scenes from the storyboard. The world state must remain stable. Do not change weather, lighting, architecture, season, or color palette. Treat this frame as another camera shot from the same film scene. "
            "ENVIRONMENT CONTINUITY: The environment must remain visually consistent. Maintain same street type, same architectural style, same weather conditions, same surface materials, and same atmosphere. The viewer should feel that all frames belong to the same real location. "
            f"SESSION WORLD ANCHORS:\n"
            f"Style anchor: {style_anchor}\n"
            f"Lighting anchor: {lighting_anchor}\n"
            f"Location anchor: {location_anchor}\n"
            f"Environment anchor: {environment_anchor}\n\n"
            f"Weather anchor: {weather_anchor}\n"
            f"Surface anchor: {surface_anchor}\n\n"
            "Use these anchors as global constraints. "
            "All generated frames must obey these anchors. "
            "Do not reinterpret them. "
            "STYLE-DEFINED ENVIRONMENT STATE: If style reference images are present, they define season, weather state, lighting mood, color palette, surface condition, and atmospheric mood. These style-defined environmental states must remain stable across all scenes. Do not reinterpret or weaken them in later scenes. If the style references imply winter snow, snow traces and cold winter atmosphere, do not switch later scenes to generic wet cloudy weather without snow. "
            "WEATHER STATE LOCK: Weather must remain the same across the whole session unless explicitly changed by text. If the style reference implies snow, winter cold, or overcast winter weather, then all scenes must preserve that same weather state. Do NOT switch between snow, rain, dry cloudy weather, and neutral weather unless explicitly requested. Weather continuity includes presence/absence of snow, snow traces on roofs and ground, wetness level, and atmospheric coldness. "
            "SURFACE STATE LOCK: Ground and surface conditions must remain visually consistent across scenes. Maintain same pavement material, same wetness level, same snow traces, same reflection behavior, and same environmental wear. If the first scene shows wet cobblestone with snow traces, later scenes must preserve that same surface logic. "
            "GLOBAL ENVIRONMENT STATE: Style reference images define the global environment state for the entire session. This includes season, weather, lighting mood, color palette, atmospheric conditions, and ground surface state. These properties must remain constant across all scenes. Camera framing or shot type (wide, medium, close-up, macro) must NOT weaken these environmental constraints. "
            "VISIBLE WEATHER LOCK: If snow is part of the style-defined environment state, snow must remain visible in every frame. Snow accumulation or snow traces must remain visible on at least some of ground edges, rooftops, pavement gaps, horizontal surfaces, street borders, and environmental surfaces. Do not reduce snowy winter state into generic wet cold weather. Visible weather cues must remain present even in close-up and macro shots. "
            "SUBJECT RELIGHTING RULE: Character lighting must be derived entirely from the environment. Do not preserve lighting baked into character reference images. The generated subject must match the environment in light direction, color temperature, ambient bounce light, shadow softness, exposure, and atmospheric haze. The character must not look studio-lit inside an outdoor cinematic environment. "
            "PHYSICAL SUBJECT INTEGRATION: The character must appear physically present inside the same world as the background. Match ambient depth, edge contrast, environmental color bounce, surface reflections, ground contact shadows, and local atmospheric perspective. Do not render the character as pasted, composited, cut out, separately lit, or cleaner than the environment. The subject must feel photographed in the same place and lighting conditions as the environment. "
            "SUBJECT AND PROP ENVIRONMENT MATCH: Character and prop must inherit the same environmental qualities as the scene. Match ambient haze, dust, smoke diffusion, reflected dirty light, floor color bounce, local contrast softness, and environmental color contamination. Do not render the character or prop as cleaner, sharper, or separately lit than the environment. Character and prop must feel physically present in the same air, same light, and same atmosphere as the scene. "
            "NO CUTOUT / NO COMPOSITE LOOK: Do not render the character or prop as pasted, composited, cut out, sticker-like, separately exposed, or separately color-graded. They must feel captured inside the same environment, with the same atmospheric depth and lighting logic. Edges, contrast, color temperature, and softness must match the environment. "
            "CHARACTER INTEGRATION LOCK: The character must inherit local ambient light, environmental shadow softness, atmospheric haze, reflected floor color, and industrial/urban environmental contamination when applicable. Do not keep the character unnaturally clean or studio-like if the world is dusty, hazy, smoky, wet, dirty, snowy, or industrial. The subject must feel photographed in the same environment, not inserted afterward. "
            "ATMOSPHERIC DEPTH RULE: All visible elements must be affected by the same atmosphere. Apply consistent haze, moisture, light scattering and atmospheric depth to background, character, and props. Do not keep the subject artificially crisp if the environment is soft, hazy, cold, wet, snowy, or diffuse-lit. "
            "PROP SCALE LOCK: The prop must preserve the same real-world physical size class across all scenes. The object must remain physically plausible relative to the human body. Do not enlarge it, shrink it, exaggerate it, miniaturize it, or distort its real-world scale between shots. The prop must keep stable human-relative scale in every frame. Example: if the prop is a portable welding machine, it must remain portable-welder sized in every scene, not suitcase-sized in one scene and generator-sized in another. "
            "PROP SIZE CLASS LOCK: The prop must belong to a stable real-world size class across all scenes. The object must not change its physical class between frames. Example: a portable welding machine must remain portable-welder sized in every frame. It must not become oversized, generator-sized, miniaturized, enlarged to fit composition, or distorted in apparent volume. The prop must remain physically plausible relative to the human body. "
            "HARD PROP SIZE CLASS LOCK: The prop belongs to a fixed real-world size class. If the prop is a portable welding machine, its physical class is compact carryable equipment, approximately small-suitcase class. It must never become oversized, generator-sized, floor-machine sized, enlarged to dominate the frame, or visually inflated to reveal more detail. The prop must keep the same physical size class across all scenes. "
            "BODY-RELATIVE SCALE REFERENCE: The prop must remain consistent relative to the human body. Use stable body-relative references such as hand grip, shin height, knee level, lower leg size, and forearm carry scale. Do not change prop size class between wide shots, medium shots, close-ups, and macro shots. Framing must not justify scaling the object larger or smaller. "
            "CAMERA DISTANCE RULE: Changes in framing must come from camera distance, not object scaling. Wide shots, medium shots, close-ups, and macro shots must NOT change the physical size of the prop. The prop must remain the same real-world object size. When a shot becomes closer, the camera moves closer to the object instead of enlarging it. Do not increase prop size to reveal detail. Framing changes must be achieved through camera movement, lens choice, or crop — not object scaling. "
            "PROP DETAIL RULE: When a shot requires more visible detail of the prop, reveal detail through camera proximity, lighting, and focus. Do NOT increase the object's physical size. "
            "HUMAN HAND SCALE RULE: Handheld props must remain consistent with human hand size. If a prop is carried by one hand, its dimensions must remain believable for a single-hand grip. Do not enlarge handheld props beyond realistic carryable scale. "
            "ANATOMIC ANCHORING: Object scale must be anchored relative to the human body. Use stable body-relative proportions such as knee height, lower leg height, hand-carryable size, and forearm/torso relation. The prop must keep the same body-relative scale across all shots. Do not resize the object just because the framing changes. "
            "MACRO CONTEXT LOCK: In close-up or macro shots, the environment state must remain visible through the surface context. If the wide-shot environment is snowy wet cobblestone street, then close-up shots must preserve that same surface logic. Macro shots must not forget snow traces, wetness, pavement material, and winter environment cues. Close framing must not weaken global world continuity. "
            "PROP INTEGRATION HARD LOCK: The prop must be integrated into the environment with the same ambient light, shadow softness, color temperature, reflected floor color, atmospheric softness, and dirt/haze/smoke context. Do not render the prop as a clean product render inside a dirty scene. The prop must visually belong to the same world as the floor, air, and surrounding light. "
            "PROP RELIGHTING RULE: Do not preserve lighting baked into prop reference images. Prop references define object identity, category, silhouette, material cues, and usage only. Every prop must be fully relit by the current scene environment and must match world light direction, color temperature, ambient bounce, atmospheric diffusion, environmental reflections, shadow softness, shadow direction, and environment color contamination. "
            "ENVIRONMENTAL CONTAMINATION LOCK: If the environment contains dust, smoke, industrial haze, wet reflections, cold fog, snow residue, or dirty floor bounce, then character and prop must inherit that environmental contamination visually. They must not look isolated from the environmental conditions. "
            "SURFACE INTERACTION RULE: Any object placed on ground, floor, table, or other support surface must show physical contact with that surface. Require contact shadows, plausible contact pressure, subtle local bounce or reflections when appropriate, and slight dust/dirt grounding when appropriate. No floating look and no clean studio isolation inside dirty, industrial, snowy, or wet environments. "
            "PROP CATEGORY AND SCALE LOCK: Prop references define object class (portable_tool, handheld_object, device, furniture, large_machine, vehicle, environment_object). Keep category-faithful shape, function, and human-relative scale across all shots. Portable and handheld props must keep realistic body-relative size and must not randomly grow or shrink between scenes. "
            "CLIP WORLD LOCK: Every shot in one clip must belong to one shared world identity. Preserve the same lighting logic, atmosphere, weather state, palette, material response, and dust/fog/snow/rain state across shots. Shot variation is allowed, but world identity must remain unchanged. "
            "PROP PHYSICAL CONSISTENCY: Keep consistent size relative to hands, size relative to torso/legs, grip logic, weight impression, handle/cable behavior, and ground contact behavior. The prop must not look weightless, oversized, undersized, or physically inconsistent between scenes. If the prop is handheld, its scale must remain realistically liftable by the character. "
            "Scene text may be Russian and visual prompt may be English. Use both when available: visual prompt defines composition/action, and scene text defines narrative context and emotion."
        )

        parts = [{"text": system_prompt}]

        if character_images:
            parts.append({"text": "Character reference images. All depict the SAME main character."})
            parts.extend(character_images)

        if location_images:
            parts.append({"text": "Location reference images. These define the same world/environment."})
            parts.extend(location_images)

        if style_images:
            parts.append({"text": "Style reference images. These define season, weather, palette, texture, atmosphere, and overall visual language. Apply them explicitly and visibly in the final frame."})
            parts.extend(style_images)

        if props_images:
            parts.append({"text": "Props reference images. These are key scene objects. Keep them prominent when relevant; if only one prop is attached, treat it as primary and do not omit it."})
            parts.extend(props_images)
            parts.append({"text": "The prop identity is defined by the reference images and must not be replaced."})
            if prop_anchor_label:
                parts.append({"text": f"Session prop anchor label: {prop_anchor_label}. Keep exactly this prop identity."})

        if prop_anchor_label:
            prompt = _enforce_prop_anchor_text(prompt, prop_anchor_label, lang="en")
            scene_text = _enforce_prop_anchor_text(scene_text, prop_anchor_label, lang="ru")

        prompt = (
            _inject_session_world_anchors(
                prompt,
                {
                    "character": session_character_anchor or "coherent single-character identity across all scenes",
                    "location": session_location_anchor or location_anchor,
                    "style": session_style_anchor or style_anchor,
                },
            )
            + "\n\n"
            + f"Lighting anchor: {lighting_anchor}\n"
            + f"Environment anchor: {environment_anchor}\n"
            + f"Weather anchor: {weather_anchor}\n"
            + f"Surface anchor: {surface_anchor}\n\n"
            "PHYSICAL SCALE RULES:\n\n"
            "Keep the prop at the same realistic real-world size across all frames.\n"
            "The object must remain physically plausible relative to the person.\n"
            "Do not change object scale between shots.\n\n"
            "WEATHER / SURFACE RULES:\n\n"
            "Keep the same weather state and surface condition as defined by the style references.\n"
            "Do not remove snow if snow is part of the style-defined world state.\n"
            "Do not switch surface logic between snowy, wet, and dry unless explicitly requested.\n\n"
            "SUBJECT / SCALE / ATMOSPHERE RULES:\n\n"
            "Keep the character fully integrated into the environment.\n"
            "Match subject lighting to the scene.\n"
            "Preserve visible weather cues from the style-defined world state.\n"
            "Keep the prop at the same realistic real-world size class across all frames.\n"
            "Do not resize the prop for composition.\n"
            "Maintain the same surface logic in wide, medium, close-up and macro shots.\n\n"
            "SUBJECT / PROP REALISM RULES:\n\n"
            "Keep the character and prop fully integrated into the environment.\n"
            "Do not allow pasted or cutout appearance.\n"
            "Match local atmosphere, dirty light, ambient haze, reflections, and contrast softness.\n"
            "Keep the prop at the same compact real-world size class across all frames.\n"
            "Do not enlarge the prop for visibility or composition.\n\n"
            "HARD WORLD RELIGHTING RULES:\n\n"
            "Location/style/world references are the lighting authority for this frame.\n"
            "Character and prop references define identity only and must be fully relit by world lighting and atmosphere.\n"
            "Never preserve reference-image lighting for props or characters.\n"
            "Any grounded object must show contact shadows and believable surface interaction.\n"
            "All visible objects must inherit environmental color contamination, haze, and weather response.\n"
            "Keep one world identity across clip shots: stable lighting logic, palette, atmosphere, weather, and material response."
        )

        scene_payload = {
            "sceneId": scene_id,
            "style": style,
            "styleKey": style,
            "aspectRatio": aspect_ratio,
            "resolution": f"{width}x{height}",
            "sceneText": scene_text,
            "visualPrompt": prompt,
            "propAnchorLabel": prop_anchor_label or None,
            "sessionCharacterAnchor": session_character_anchor or None,
            "sessionLocationAnchor": session_location_anchor or None,
            "sessionStyleAnchor": session_style_anchor or None,
            "styleAnchor": style_anchor,
            "lightingAnchor": lighting_anchor,
            "locationAnchor": location_anchor,
            "environmentAnchor": environment_anchor,
            "weatherAnchor": weather_anchor,
            "surfaceAnchor": surface_anchor,
        }
        parts.append({"text": "Scene payload:\n" + json.dumps(scene_payload, ensure_ascii=False)})

        body = {
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        resp = post_generate_content(api_key, model, body, timeout=120)
        decoded = _decode_gemini_image(resp if isinstance(resp, dict) else {})
        if decoded:
            raw, ext = decoded
            image_url = _save_bytes_as_asset(raw, ext)
            return {
                "ok": True,
                "sceneId": scene_id,
                "imageUrl": image_url,
                "engine": "gemini",
                "modelUsed": model,
                "refsDebug": refs_debug,
            }

        image_url = _mock_scene_image(scene_id, width, height)
        return {
            "ok": True,
            "sceneId": scene_id,
            "imageUrl": image_url,
            "engine": "mock",
            "hint": "gemini_no_image",
            "modelUsed": model,
            "refsDebug": refs_debug,
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "code": "BAD_REQUEST", "hint": str(e)[:300]})
    except Exception as e:
        try:
            image_url = _mock_scene_image(scene_id, width, height)
            return {
                "ok": True,
                "sceneId": scene_id,
                "imageUrl": image_url,
                "engine": "mock",
                "hint": f"gemini_error:{str(e)[:200]}",
                "modelUsed": model if 'model' in locals() else None,
                "refsDebug": refs_debug,
            }
        except Exception:
            return JSONResponse(status_code=500, content={"ok": False, "code": "IMAGE_GENERATION_FAILED", "hint": str(e)[:300]})


@router.post("/clip/audio-slice")
def clip_audio_slice(payload: AudioSliceIn):
    scene_id = (payload.sceneId or "").strip()
    if not scene_id:
        return JSONResponse(status_code=400, content={"ok": False, "code": "BAD_REQUEST", "hint": "sceneId_required"})

    t0 = round(float(payload.t0), 3)
    t1 = round(float(payload.t1), 3)
    if t0 < 0:
        return JSONResponse(status_code=400, content={"ok": False, "code": "bad_t0", "hint": "t0_must_be_non_negative"})
    if t1 <= t0:
        return JSONResponse(status_code=400, content={"ok": False, "code": "bad_range", "hint": "t1_must_be_greater_than_t0"})
    if (t1 - t0) > 300.0:
        return JSONResponse(status_code=400, content={"ok": False, "code": "slice_too_long", "hint": "max_slice_sec_300"})

    path = _resolve_audio_asset_path(payload.audioUrl)
    if not path:
        _debug_audio_slice(payload.audioUrl, path)
        return JSONResponse(status_code=400, content={"ok": False, "code": "invalid_audioUrl", "hint": "audioUrl_must_point_to_/static/assets/<file>_or_/assets/<file>"})

    _ensure_assets_dir()
    safe_scene = re.sub(r"[^a-zA-Z0-9_-]", "_", scene_id) or "scene"
    t0_ms = int(round(t0 * 1000))
    t1_ms = int(round(t1 * 1000))
    filename = f"clip_audio_{safe_scene}_{t0_ms}_{t1_ms}_{uuid4().hex[:8]}.mp3"
    output_path = os.path.join(str(ASSETS_DIR), filename)

    ok, err = _ffmpeg_audio_slice(path, output_path, t0, t1)
    if not ok:
        _debug_audio_slice(payload.audioUrl, path)
        return JSONResponse(status_code=500, content={"ok": False, "code": "slice_failed", "hint": err})

    return {
        "ok": True,
        "audioUrl": payload.audioUrl,
        "audioSliceUrl": _asset_url(filename),
        "sliceUrl": _asset_url(filename),
        "t0": t0,
        "t1": t1,
        "duration": round(t1 - t0, 3),
        "audioSliceBackendDurationSec": round(t1 - t0, 3),
    }
