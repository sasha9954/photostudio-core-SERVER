from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import tempfile
import subprocess
import math
import base64
import json
import re
import os
from urllib.parse import urlparse
from uuid import uuid4

from PIL import Image, ImageDraw

from app.core.config import settings
from app.engine.gemini_rest import post_generate_content

router = APIRouter()

ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "assets")
)
ASSETS_DIR_APP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "app", "static", "assets")
)
ASSETS_DIR_ALT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "static", "assets")
)


class ClipImageIn(BaseModel):
    sceneId: str
    prompt: str
    style: str | None = "default"
    width: int | None = 1024
    height: int | None = 1024


class AudioSliceIn(BaseModel):
    sceneId: str
    t0: float
    t1: float
    audioUrl: str


def _ensure_assets_dir() -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)


def _asset_url(filename: str) -> str:
    base = (settings.PUBLIC_BASE_URL or "http://127.0.0.1:8000").rstrip("/")
    return f"{base}/static/assets/{filename}"


def _save_bytes_as_asset(raw: bytes, ext: str = "png") -> str:
    _ensure_assets_dir()
    ext = (ext or "png").lower().replace(".", "")
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        ext = "png"
    filename = f"clip_scene_{uuid4().hex}.{ext}"
    fpath = os.path.join(ASSETS_DIR, filename)
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

    dirs = [ASSETS_DIR_APP, ASSETS_DIR, ASSETS_DIR_ALT]
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
        dirs = [ASSETS_DIR_APP, ASSETS_DIR, ASSETS_DIR_ALT]
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
    print("ASSETS_DIR_APP:", ASSETS_DIR_APP)
    print("ASSETS_DIR:", ASSETS_DIR)
    print("ASSETS_DIR_ALT:", ASSETS_DIR_ALT)
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
    img.save(os.path.join(ASSETS_DIR, filename), format="PNG")
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


class BrainIn(BaseModel):
    audioUrl: str | None = None
    text: str | None = None

    # brain settings (optional)
    scenarioKey: str | None = None   # e.g. "beat_rhythm" | "song_meaning"
    shootKey: str | None = None      # e.g. "cinema"
    styleKey: str | None = None      # e.g. "realism"
    freezeStyle: bool | None = None

    # refs (urls) - optional, just for prompt context
    refCharacter: str | None = None
    refLocation: str | None = None
    refStyle: str | None = None

    # informational (optional)
    audioType: str | None = None     # "song" | "bg"
    textType: str | None = None      # "lyrics" | "story" | "notes"


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
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(r.content)
            path = f.name

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
        return 30.0
    except Exception:
        return 30.0


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
            "why": "fallback slicing",
            "sceneText": ch,
            "imagePrompt": f"Cinematic scene: {ch}",
            "videoPrompt": "Cinematic camera movement, dramatic lighting, film grain"
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
        out.append({
            "id": str(s.get("id") or f"s{i+1:02d}"),
            "start": round(t0, 2),
            "end": round(t1, 2),
            "why": str(s.get("why") or ""),
            "sceneText": str(s.get("sceneText") or ""),
            "imagePrompt": str(s.get("imagePrompt") or s.get("prompt") or s.get("sceneText") or ""),
            "videoPrompt": str(s.get("videoPrompt") or ""),
        })
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


@router.post("/clip/plan")
def clip_plan(payload: BrainIn):
    """SMART ScenePlan: returns timecoded scenes across whole audio."""
    text = (payload.text or "").strip()

    duration = 30.0
    audio_bytes = None
    audio_mime = "audio/mpeg"

    if payload.audioUrl:
        duration = get_audio_duration(payload.audioUrl)
        try:
            r = requests.get(payload.audioUrl, timeout=30)
            r.raise_for_status()
            audio_bytes = r.content
        except Exception:
            audio_bytes = None

    # If no key -> fallback
    if not (settings.GEMINI_API_KEY or "").strip():
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "modelUsed": None,
            "fallbackUsed": False,
            "hint": "no_gemini_key",
            "error": {
                "code": "GENERATION_FAILED",
                "hint": "no_gemini_key",
                "modelUsed": None,
                "fallbackUsed": False,
            },
        }

    scenario_key = (payload.scenarioKey or "beat_rhythm").strip()
    shoot_key = (payload.shootKey or "cinema").strip()
    style_key = (payload.styleKey or "realism").strip()
    freeze = bool(payload.freezeStyle)

    rules = f"""Ты — режиссёр монтажа музыкального клипа.
Нужно построить SMART storyboard по треку и (если дан) тексту.
Цель: умные таймкоды смены сцен по смыслу вокала/слов и по ритму/биту.

ОБЯЗАТЕЛЬНО:
- Верни ТОЛЬКО JSON (без пояснений, без Markdown).
- Длительность трека: ~{duration:.1f} секунд.
- Сцены должны покрывать ВЕСЬ таймлайн от 0 до {duration:.1f}.
- Делай ~6–9 сцен на 30–60 сек (адаптируй под длительность).
- Переходы делай на музыкальных акцентах/снейре (каждый 2-й или 4-й удар), но не дроби бессмысленно.
- Если есть вокал/слова — границы сцен ставь на смысловых фразах/переходах (куплет/припев/бридж).
- Стиль: {style_key}. Съёмка: {shoot_key}. FreezeStyle: {freeze}.
- Если текста нет — всё равно делай осмысленный клиповый план по музыке.

JSON СХЕМА:
{{
  "audioDuration": number,
  "scenes": [
    {{
      "id": "s01",
      "start": number,
      "end": number,
      "why": "коротко почему тут переход",
      "sceneText": "что происходит в кадре",
      "imagePrompt": "промт для генерации картинки",
      "videoPrompt": "промт движения камеры/анимации (3–5 сек)"
    }}
  ]
}}

ВАЖНО:
- start/end в секундах, с 1–2 знаками после запятой.
- end строго > start.
- Последняя сцена end = {duration:.1f}.
"""

    ref_hints = []
    if payload.refCharacter:
        ref_hints.append("Есть реф персонажа (character reference).")
    if payload.refLocation:
        ref_hints.append("Есть реф локации (location reference).")
    if payload.refStyle:
        ref_hints.append("Есть реф стиля (style reference).")

    extra = ""
    if ref_hints:
        extra += "\n" + " ".join(ref_hints)
    if text:
        extra += "\nТЕКСТ/СМЫСЛ (может быть история или слова песни):\n" + text[:4000]

    parts = [{"text": rules + extra}]

    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        parts.append({"inlineData": {"mimeType": audio_mime, "data": b64}})

    generation_config = {
        "temperature": 0.25,
        "topP": 0.9,
        "maxOutputTokens": 4096,
    }

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }

    model_used = settings.GEMINI_VISION_MODEL if audio_bytes else settings.GEMINI_TEXT_MODEL
    fallback_used = False

    # Call Gemini with safe model fallback on unsupported-model errors
    resp = post_generate_content(settings.GEMINI_API_KEY, model_used, body, timeout=120)
    combined_error = _combined_error_text(resp if isinstance(resp, dict) else None)
    first_model = model_used
    first_error = combined_error[:1500] if combined_error else None
    first_was_unsupported = _is_model_unsupported_error(first_error or "")
    if first_was_unsupported:
        model_used = _pick_fallback_model(model_used)
        fallback_used = True
        resp = post_generate_content(settings.GEMINI_API_KEY, model_used, body, timeout=120)
        combined_error = _combined_error_text(resp if isinstance(resp, dict) else None)

    error_hint = (combined_error or "")[:1500] if combined_error else None

    # If http error BUT body may contain JSON in text -> try parse it before fallback
    if isinstance(resp, dict) and resp.get("__http_error__"):
        raw_text = resp.get("text") or ""
        j = _parse_json_from_text(raw_text)
        if isinstance(j, dict) and isinstance(j.get("scenes"), list):
            scenes = _normalize_scenes(duration, j.get("scenes") or [])
            if scenes:
                return {
                    "ok": True,
                    "engine": "gemini_partial",
                    "audioDuration": duration,
                    "scenes": scenes,
                    "modelUsed": model_used,
                    "fallbackUsed": fallback_used,
                    "hint": "http_error_but_parsed_json",
                }
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": error_hint or raw_text[:1500],
            "error": {
                "code": "MODEL_UNSUPPORTED" if first_was_unsupported else "GENERATION_FAILED",
                "hint": error_hint or raw_text[:1500],
                "modelUsed": model_used,
                "fallbackUsed": fallback_used,
                "firstAttempt": {
                    "modelUsed": first_model,
                    "hint": first_error,
                } if fallback_used else None,
            },
        }

    # Normal parse
    text_out = _extract_gemini_text(resp if isinstance(resp, dict) else {})
    j = _parse_json_from_text(text_out)
    if (not isinstance(j, dict) or "scenes" not in j) and '"scenes"' in (text_out or "") and "{" in (text_out or ""):
        retry_parts = [
            {"text": rules + extra + "\n\nReturn ONLY valid JSON, no markdown, no code fences, ensure all braces closed."}
        ]
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode("ascii")
            retry_parts.append({"inlineData": {"mimeType": audio_mime, "data": b64}})
        retry_body = {
            "contents": [{"role": "user", "parts": retry_parts}],
            "generationConfig": generation_config,
        }
        retry_resp = post_generate_content(settings.GEMINI_API_KEY, model_used, retry_body, timeout=120)
        retry_text = _extract_gemini_text(retry_resp if isinstance(retry_resp, dict) else {})
        j = _parse_json_from_text(retry_text)
        if isinstance(retry_resp, dict):
            retry_error_hint = _combined_error_text(retry_resp)
            if retry_error_hint:
                error_hint = (retry_error_hint or "")[:1500]
        text_out = retry_text or text_out

    if not isinstance(j, dict) or "scenes" not in j:
        scenes = _fallback_plan(duration, text)
        hint = (text_out or error_hint or "")[:1500] or None
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": hint,
            "error": {
                "code": "MODEL_UNSUPPORTED" if first_was_unsupported else "GENERATION_FAILED",
                "hint": hint,
                "modelUsed": model_used,
                "fallbackUsed": fallback_used,
                "firstAttempt": {
                    "modelUsed": first_model,
                    "hint": first_error,
                } if fallback_used else None,
            },
        }

    scenes = _normalize_scenes(duration, j.get("scenes") or [])
    if not scenes:
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": "empty_scenes",
            "error": {
                "code": "GENERATION_FAILED",
                "hint": "empty_scenes",
                "modelUsed": model_used,
                "fallbackUsed": fallback_used,
            },
        }

    return {
        "ok": True,
        "engine": "gemini",
        "audioDuration": duration,
        "scenes": scenes,
        "modelUsed": model_used,
        "fallbackUsed": fallback_used,
        "hint": None,
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
        return {"ok": True, "sceneId": scene_id, "imageUrl": image_url, "engine": "mock", "hint": "no_gemini_key"}

    try:
        model = settings.GEMINI_IMAGE_MODEL or "gemini-2.5-flash-image-preview"
        body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": f"Create one cinematic frame for storyboard scene {scene_id}. Style: {style}. Aspect ratio: {aspect_ratio}. Resolution: {width}x{height}. {prompt}"}],
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        resp = post_generate_content(api_key, model, body, timeout=120)
        decoded = _decode_gemini_image(resp if isinstance(resp, dict) else {})
        if decoded:
            raw, ext = decoded
            image_url = _save_bytes_as_asset(raw, ext)
            return {"ok": True, "sceneId": scene_id, "imageUrl": image_url, "engine": "gemini"}

        image_url = _mock_scene_image(scene_id, width, height)
        return {"ok": True, "sceneId": scene_id, "imageUrl": image_url, "engine": "mock", "hint": "gemini_no_image"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"ok": False, "code": "BAD_REQUEST", "hint": str(e)[:300]})
    except Exception as e:
        try:
            image_url = _mock_scene_image(scene_id, width, height)
            return {"ok": True, "sceneId": scene_id, "imageUrl": image_url, "engine": "mock", "hint": f"gemini_error:{str(e)[:200]}"}
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
    output_path = os.path.join(ASSETS_DIR, filename)

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
    }
