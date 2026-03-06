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
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url
from app.engine.gemini_rest import post_generate_content

router = APIRouter()


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


class RefUrlItem(BaseModel):
    url: str


class BrainRefsIn(BaseModel):
    character: list[RefUrlItem] = []
    location: list[RefUrlItem] = []
    props: list[RefUrlItem] = []
    style: RefUrlItem | None = None


class BrainIn(BaseModel):
    audioUrl: str | None = None
    text: str | None = None

    # brain settings (optional)
    scenarioKey: str | None = None   # e.g. "beat_rhythm" | "song_meaning"
    shootKey: str | None = None      # e.g. "cinema"
    styleKey: str | None = None      # e.g. "realism"
    freezeStyle: bool | None = None

    # refs (urls)
    refs: BrainRefsIn | None = None
    characterRefs: list[RefUrlItem] | None = None
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


def _load_audio_for_planner(audio_url: str | None) -> tuple[float, bytes | None, str, dict]:
    duration = 30.0
    audio_mime = "audio/mpeg"
    debug = {
        "inputAudioUrl": audio_url or None,
        "resolvedPath": None,
        "audioBytesFound": False,
        "audioBytesSource": "none",
        "hint": "",
    }

    if not audio_url:
        debug["hint"] = "audio_url_missing"
        return duration, None, audio_mime, debug

    resolved_path = _resolve_audio_asset_path(audio_url)
    if resolved_path and os.path.isfile(resolved_path):
        debug["resolvedPath"] = resolved_path
        duration = get_audio_duration(resolved_path)
        try:
            with open(resolved_path, "rb") as f:
                audio_bytes = f.read()
            if audio_bytes:
                debug["audioBytesFound"] = True
                debug["audioBytesSource"] = "local_asset"
                debug["hint"] = "audio_loaded_from_local_asset"
                return duration, audio_bytes, audio_mime, debug
        except Exception:
            pass

    try:
        duration = get_audio_duration(audio_url)
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        audio_bytes = r.content
        if audio_bytes:
            debug["audioBytesFound"] = True
            debug["audioBytesSource"] = "http"
            debug["hint"] = "audio_loaded_over_http"
            return duration, audio_bytes, audio_mime, debug
    except Exception:
        pass

    debug["hint"] = "audio_not_found_or_unreachable_planner_built_without_audio_bytes"
    return duration, None, audio_mime, debug


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
    """SMART ScenePlan: returns timecoded scenes across whole audio."""
    text = (payload.text or "").strip()

    duration, audio_bytes, audio_mime, audio_debug = _load_audio_for_planner(payload.audioUrl)
    empty_validation_debug = {
        "scenario": "clip",
        "sceneCount": 0,
        "emptySceneTextCount": 0,
        "emptyImagePromptCount": 0,
        "emptyVideoPromptCount": 0,
        "emptyCoreSceneCount": 0,
        "warnings": [],
        "rejectedReason": None,
        "repairRetryUsed": False,
    }

    scenario_key = "clip"
    shoot_key = (payload.shootKey or "cinema").strip()
    style_key = (payload.styleKey or "realism").strip()
    freeze = bool(payload.freezeStyle)
    audio_type_hint = (payload.audioType or "").strip().lower()
    text_type_hint = (payload.textType or "").strip().lower()
    want_lipsync = bool(payload.wantLipSync)

    def _normalize_ref_list(items, max_items: int = 5):
        out = []
        if not items:
            return out
        for it in items:
            url = str(getattr(it, "url", "") or "").strip()
            if url:
                out.append(url)
        return out[:max_items]

    refs_obj = payload.refs
    character_refs = _normalize_ref_list((refs_obj.character if refs_obj else None) or payload.characterRefs, 5)
    location_refs = _normalize_ref_list((refs_obj.location if refs_obj else None) or payload.locationRefs, 5)
    props_refs = _normalize_ref_list((refs_obj.props if refs_obj else None) or payload.propsRefs, 5)
    style_ref = str((refs_obj.style.url if refs_obj and refs_obj.style else "") or (payload.styleRef.url if payload.styleRef else "") or "").strip()

    # legacy compatibility
    if not character_refs and payload.refCharacter:
        character_refs = [str(payload.refCharacter).strip()]
    if not location_refs and payload.refLocation:
        location_refs = [str(payload.refLocation).strip()]
    if not props_refs and payload.refItems:
        props_refs = [str(payload.refItems).strip()]
    if not style_ref and payload.refStyle:
        style_ref = str(payload.refStyle).strip()

    planning_semantics = _build_planning_semantics(
        text=text,
        scenario_key=scenario_key,
        audio_type_hint=audio_type_hint,
        text_type_hint=text_type_hint,
        want_lipsync=want_lipsync,
        character_refs=character_refs,
        location_refs=location_refs,
        props_refs=props_refs,
        style_key=style_key,
    )

    product_distribution_enabled = bool(
        planning_semantics.get("productMode") and planning_semantics.get("productRefCount", 0) > 1
    )

    planner_debug_base = {
        "audio": audio_debug,
        "planningSemantics": planning_semantics,
        "productDistributionEnabled": product_distribution_enabled,
    }

    # If no key -> fallback
    if not (settings.GEMINI_API_KEY or "").strip():
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "plannerDebug": {**planner_debug_base, "validation": empty_validation_debug},
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

    rules = f"""Ты — режиссёр монтажа музыкального клипа.
Нужно построить SMART storyboard по треку и (если дан) тексту.
Цель: умные таймкоды смены сцен по смыслу вокала/слов и по ритму/биту.

ОБЯЗАТЕЛЬНО:
- Верни ТОЛЬКО JSON (без пояснений, без Markdown).
- Длительность трека: ~{duration:.1f} секунд.
- Сцены должны покрывать ВЕСЬ таймлайн от 0 до {duration:.1f}.
- Делай ~6–9 сцен на 30–60 сек (адаптируй под длительность).
- Сначала классифицируй тип аудио для каждой сцены: instrumental | song_with_vocals | speech | mixed.
- Для каждой сцены укажи sceneType: visual_rhythm | vocal | lipSync.
- Переходы делай на музыкальных акцентах/снейре (каждый 2-й или 4-й удар), но не дроби бессмысленно.
- Если есть вокал/слова — границы сцен ставь на смысловых фразах/переходах (куплет/припев/бридж).
- Стиль: {style_key}. Съёмка: {shoot_key}. FreezeStyle: {freeze}.
- Если текста нет — всё равно делай осмысленный клиповый план по музыке.
- Все текстовые поля в JSON возвращай ТОЛЬКО на русском языке.
- Даже если песня/текст не на русском, служебные поля planner всё равно должны быть на русском.
- Поля why, sceneText, lyricFragment, timingReason, imagePrompt, videoPrompt — всегда только русский.

КОНТИНЬЮИТИ И LOCK-ПРАВИЛА (ОБЯЗАТЕЛЬНО):
A) Если есть styleRef: это STYLE LOCK (высший приоритет).
- Это глобальный стиль всего ролика.
- Все сцены, персонажи, локации и предметы должны быть стилизованы под styleRef.
- Нельзя смешивать несвязанные стили при наличии styleRef.

B) Если есть characterRefs[]: это IDENTITY LOCK.
- Все изображения characterRefs описывают одного персонажа.
- Во всех сценах сохраняй лицо, образ и идентичность персонажа.
- Нельзя менять персонажа между сценами.

C) Если есть locationRefs[]: это WORLD LOCK.
- Если locationRefs несколько, построй единый world profile и логичные зоны мира.
- Нельзя распределять locationRefs по сценам 1-к-1 без общей логики.
- Если locationRefs один — все сцены в рамках одного мира.
- Если locationRefs нет — придумай мир, но сохрани единый стиль.

D) Если есть propsRefs[]: это PROPS LOCK.
- Используй эти предметы как важные props в сценах.
- Не теряй предметы случайно между сценами.
- Не обязательно показывать все предметы в каждом кадре.

E) Если текста нет:
- Строй свободный нарратив внутри зафиксированного мира (free narrative inside locked world).
- Нельзя допускать случайные смысловые, локационные и стилистические скачки.

F) Если текст есть:
- Текст — это story guidance по смыслу сцен.
- Но текст не должен ломать continuity: identity/location/style lock сохраняются обязательно.

ПРАВИЛА ПО ВОКАЛУ И LIPSYNC:
A) ПРИОРИТЕТ АНАЛИЗА ВОКАЛА:
- Если textType_hint=lyrics ИЛИ audioType_hint=song ИЛИ wantLipSync=true,
  сначала найди вокальные фразы (границы строк/фраз/дыхания),
  и только потом уточняй переходы по ритму/биту.
- В этих режимах нельзя строить план только от ритма, игнорируя вокальные фразы.

B) Если в аудио есть вокал:
- различай инструментальные и вокальные отрезки;
- отмечай hasVocals=true только там, где реально слышен голос.

- Если wantLipSync=true и в аудио есть вокал:
  MUST включить минимум 1 lipSync сцену,
  предпочтительно 1–2 lipSync сцены в подходящих местах трека;
  каждая lipSync сцена должна быть вокруг ЦЕЛЬНОЙ вокальной фразы.

C) Если сцена lipSync (isLipSync=true или sceneType=lipSync):
- выбирай t0/t1 только вокруг целой вокальной фразы;
- начало не должно попадать в середину слова;
- конец не должен обрывать слово/слог;
- segment должен покрывать цельную исполняемую фразу, пригодную для синхронизации губ;
- segment нельзя делать слишком коротким, если фраза ещё продолжается;
- нельзя резать в середине слова, слога или дыхания;
- предпочитай законченные микро-фразы, а не обрезанные фрагменты;
- для lipSync тайминга используй ЧИСЛЕННЫЕ ГРАНИЦЫ:
  t0 = start_of_vocal_phrase - 0.15..0.30 sec,
  t1 = end_of_vocal_phrase + 0.10..0.25 sec;
- prefer duration roughly 2.0–6.0 sec when possible;
- avoid ultra-short segments unless the vocal phrase is truly short;
- if phrase is longer, prefer complete expressive fragment rather than clipped fragment;
- t0 should align just before audible phrase onset;
- t1 should align just after phrase resolution / breath / phrase tail;
- lyricFragment должен содержать короткий фрагмент исполняемой фразы;
- lyricFragment должен описывать именно тот фрагмент, который реально исполняется в диапазоне t0/t1;
- sceneText и videoPrompt ОБЯЗАНЫ явно описывать singing performance;
- в videoPrompt укажи эмоцию, интенсивность, дистанцию камеры и mouth-visible framing.

- ЖЁСТКАЯ СОГЛАСОВАННОСТЬ ПОЛЕЙ ДЛЯ LIPSYNC:
  если isLipSync=true, то:
  * sceneType MUST быть "lipSync"
  * hasVocals MUST быть true
  * performanceType MUST быть "singing_performance"
  * shotType MUST быть одним из: medium | closeup | mouth_closeup
  * sceneText/videoPrompt MUST описывать singing performance

D) Если lipSync=false:
- ориентируй t0/t1 на ритм, бит, переходы, дропы и изменение энергии;
- выбирай музыкально цельные куски, не режь между сильными долями без причины.

E) Для sceneType=visual_rhythm:
- segment должен начинаться и заканчиваться на музыкально устойчивой точке;
- избегай случайного реза между сильными долями;
- если есть явный переход / drop / accent / bar boundary — предпочитай его как границу;
- prefer stable boundaries near 2-beat / 4-beat / bar-like accents;
- avoid jittery timing such as meaningless 0.7–1.1 sec cuts unless explicitly justified.

F) КАЧЕСТВО И СОГЛАСОВАННОСТЬ ПОЛЕЙ:
- Если sceneType="lipSync":
  * lyricFragment MUST NOT be empty
  * hasVocals MUST be true
  * performanceType MUST be "singing_performance"
  * shotType MUST be one of: medium | closeup | mouth_closeup
  * timingReason должен объяснять, почему этот кусок удобен для lip-sync
- Если sceneType="visual_rhythm":
  * lyricFragment may be empty
  * shotType может быть широким (wide/medium/other non-lipsync framing)
  * timingReason должен объяснять музыкальную причину выбора границ

JSON СХЕМА:
{{
  "audioDuration": number,
  "scenes": [
    {{
      "id": "s01",
      "start": number,
      "end": number,
      "why": "коротко почему тут переход",
      "audioType": "instrumental | song_with_vocals | speech | mixed",
      "sceneType": "visual_rhythm | vocal | lipSync",
      "hasVocals": true,
      "isLipSync": false,
      "lyricFragment": "короткий фрагмент вокальной фразы или пусто",
      "timingReason": "почему выбраны именно такие t0/t1",
      "beatAnchor": "например: downbeat_1 | snare_2_4 | vocal_phrase_start",
      "performanceType": "cinematic_visual | singing_performance | narrative_vocal",
      "shotType": "wide | medium | closeup | mouth_closeup",
      "productView": "hero | wide | side | detail | interaction | macro",
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
- Если isLipSync=true: sceneType="lipSync", hasVocals=true, performanceType="singing_performance",
  shotType из medium|closeup|mouth_closeup, и sceneText/videoPrompt про singing performance.
- Если productMode=true: можно указывать productView для сцены (hero|wide|side|detail|interaction|macro).
- Если productMode=false: не добавляй поле productView.
"""

    if scenario_key == "clip":
        rules += """

КОНЦЕПЦИЯ РЕЖИМА "CLIP":
Это музыкальный клип, состоящий из трёх типов сцен:
1) PERFORMANCE (lipSync)
2) RHYTHM MONTAGE
3) ATMOSPHERIC / STORY INSERTS
Planner должен смешивать их.

ЛОГИКА РАСПРЕДЕЛЕНИЯ СЦЕН ДЛЯ CLIP:
- LipSync дорогой: используй экономно и только как короткий performance-акцент.
- 10–25% сцен: lipSync / performance
- 40–60% сцен: rhythm montage
- 15–30% сцен: atmospheric / narrative inserts
- LipSync сцены НЕ должны идти подряд.

ВЫБОР LIPSYNC СЦЕН:
LipSync сцены добавляй только если одновременно соблюдено:
- слышен явный вокал;
- есть цельная вокальная фраза;
- это эмоционально сильный момент.
- Не превращай весь припев в одну длинную lipSync-сцену.
- Выбирай только лучший фрагмент припева / hook.
- Предпочтительная длительность lipSync: 3–5 сек.
- Допустимо 2–3 сек для очень сильной короткой фразы.
- Максимум обычно 5–6 сек.
Приоритетные точки:
- начало припева;
- главная строка;
- эмоциональный пик;
- переход в припев.

Для lipSync сцены обязательно:
- sceneType = "lipSync"
- hasVocals = true
- isLipSync = true
- performanceType = "singing_performance"
- shotType = medium | closeup | mouth_closeup

RHYTHM MONTAGE СЦЕНЫ:
- sceneType = "visual_rhythm"
- Границы должны совпадать с сильными долями, дропами и изменением энергии трека.
- Используй beatAnchor только из:
  downbeat | snare | drop | phrase_transition

ATMOSPHERIC / STORY INSERTS:
- sceneType = "vocal" или "visual_rhythm"
- Используй их, чтобы разбавить performance, показать эмоцию, добавить атмосферу и действие персонажа.

ОГРАНИЧЕНИЯ РЕЖИМА CLIP:
- Для клипа до 30 секунд: максимум 1 lipSync сцена.
- Для клипа 30–60 секунд: максимум 2 lipSync сцены, редко 3.
- LipSync сцены не должны идти подряд.
- Остальную часть припева показывай через rhythm montage / atmosphere / story inserts.
- Clip mode должен быть музыкальным клипом, а не karaoke и не talking head.
- Баланс для clip mode: немного lipSync, немного performance, много клипового монтажа, немного истории/атмосферы.
- Даже короткий clip должен ощущаться как плотный музыкальный монтаж с достаточным количеством смен сцен.
- Для ~30 секунд обычно целись в 7–8 сцен, если аудио не даёт очень сильного основания сделать меньше.
- Избегай плана из 2–3 крупных сцен для 30-секундного музыкального клипа.

РАБОТА С ФЛАГОМ wantLipSync:
- Если wantLipSync=false:
  * Строй клип в первую очередь по beat/rhythm/energy/transitions.
  * Текст (если есть) используй как слабую смысловую подсказку, не как жёсткий диктант.
  * Это монтажный музыкальный клип.
  * Во всех сценах isLipSync=false.

- Если wantLipSync=true:
  * Только performance/lipSync сцены строй по полным vocal phrases.
  * Нельзя обрывать фразу, слово или предложение в lipSync-сцене.
  * Остальные сцены между lipSync-сценами строй по beat/energy/montage/atmosphere/story inserts.
  * Клип должен оставаться музыкальным клипом, а не сплошным talking head.
"""

    ref_hints = []
    if style_ref:
        ref_hints.append("Есть styleRef (глобальный стиль, наивысший приоритет).")
    if character_refs:
        ref_hints.append(f"Есть characterRefs: {len(character_refs)} шт (identity lock).")
    if location_refs:
        ref_hints.append(f"Есть locationRefs: {len(location_refs)} шт (world lock).")
    if props_refs:
        ref_hints.append(f"Есть propsRefs: {len(props_refs)} шт (props lock).")

    extra = ""
    if ref_hints:
        extra += "\n" + " ".join(ref_hints)
    extra += "\nПРИОРИТЕТ ИСТОЧНИКОВ (строгий): STYLE > CHARACTER > LOCATION > PROPS > TEXT > AUDIO."
    if style_ref:
        extra += f"\nstyleRef: {style_ref[:500]}"
    if character_refs:
        extra += "\ncharacterRefs:\n" + "\n".join(f"- {u[:500]}" for u in character_refs)
    if location_refs:
        extra += "\nlocationRefs:\n" + "\n".join(f"- {u[:500]}" for u in location_refs)
    if props_refs:
        extra += "\npropsRefs:\n" + "\n".join(f"- {u[:500]}" for u in props_refs)

    extra += "\nВНУТРЕННЯЯ СЕМАНТИКА PLANNER (используй как канон):"
    extra += f"\n- textType: {', '.join(planning_semantics.get('textType') or []) or 'none'}"
    extra += f"\n- audioType: {planning_semantics.get('audioType')}"
    extra += f"\n- storySource: {planning_semantics.get('storySource')}"
    extra += f"\n- timingSource: {planning_semantics.get('timingSource')}"
    extra += f"\n- speechSource: {planning_semantics.get('speechSource')}"
    extra += f"\n- audioRole: {', '.join(planning_semantics.get('audioRole') or [])}"
    extra += f"\n- propsRole: {planning_semantics.get('propsRole')}"
    extra += f"\n- productMode: {planning_semantics.get('productMode')}"
    extra += f"\n- modeInterpretation: {planning_semantics.get('modeInterpretation')}"
    extra += "\nЕсли audioType=song_with_vocals: НЕ превращай TEXT в разговорный диалог персонажа."
    extra += "\nLip sync в таком случае только singing/performance, либо полностью визуальный performance без разговорной речи."
    if planning_semantics.get('styleApplication') == 'historical_world_modern_product':
        extra += "\nКонфликт style/product: историческая атмосфера (персонаж/интерьер/свет), но товар должен остаться современным и узнаваемым."
    if product_distribution_enabled:
        extra += "\nPRODUCT REFERENCE RULES:"
        extra += "\npropsRefs represent ONE product photographed from multiple angles."
        extra += "\nDo NOT interpret propsRefs as different objects."
        extra += "\nUse them as:"
        extra += "\n- wide product shot"
        extra += "\n- side angle"
        extra += "\n- close-up detail"
        extra += "\n- interaction with product"
        extra += "\n- hero product shot"
        extra += "\nDistribute product references across scenes to create visual variation."
        extra += "\nUse different angles for different scenes."
        extra += "\nAvoid repeating identical framing unless required by rhythm."
        extra += "\nPRODUCT CONTINUITY RULE:"
        extra += "\nThe product must remain present in most scenes of the clip."
        extra += "\nDo not create scenes that completely ignore the product unless it is a very short atmospheric transition."
        extra += "\nThe product should appear as:"
        extra += "\n- hero product"
        extra += "\n- detail view"
        extra += "\n- interaction with character"
        extra += "\n- product on table"
        extra += "\n- product demonstration"
        extra += "\nAvoid scenes that replace the product with unrelated objects."

    if text:
        extra += "\nТЕКСТ/СМЫСЛ (может быть история или слова песни):\n" + text[:4000]
    if audio_type_hint:
        extra += f"\nПодсказка о типе аудио от UI: {audio_type_hint}"
    if text_type_hint:
        extra += f"\nПодсказка о типе текста от UI: {text_type_hint}"
    extra += f"\nФлаг wantLipSync от UI: {want_lipsync}"
    if text_type_hint == "lyrics" or audio_type_hint == "song" or want_lipsync:
        extra += "\nПРИОРИТЕТ: сначала найди и нарежь вокальные фразы, затем корректируй по ритму/биту."

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
        parsed_http_validation = empty_validation_debug.copy()
        j = _parse_json_from_text(raw_text)
        if isinstance(j, dict) and isinstance(j.get("scenes"), list):
            scenes = _normalize_scenes(duration, j.get("scenes") or [])
            validation = _validate_planner_scenes_quality(duration, scenario_key, scenes)
            parsed_http_validation = validation
            if scenes:
                return {
                    "ok": True,
                    "engine": "gemini_partial",
                    "audioDuration": duration,
                    "scenes": scenes,
                    "plannerDebug": {**planner_debug_base, "validation": validation},
                    "modelUsed": model_used,
                    "fallbackUsed": fallback_used,
                    "hint": "http_error_but_parsed_json" if audio_bytes else "plan_built_without_audio_bytes",
                }
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "plannerDebug": {**planner_debug_base, "validation": parsed_http_validation},
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": "plan_built_without_audio_bytes" if not audio_bytes else (error_hint or raw_text[:1500]),
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
            "plannerDebug": {**planner_debug_base, "validation": empty_validation_debug},
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": "plan_built_without_audio_bytes" if not audio_bytes else hint,
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
    validation = _validate_planner_scenes_quality(duration, scenario_key, scenes)
    if not scenes:
        validation["rejectedReason"] = validation.get("rejectedReason") or "empty_scenes"
        scenes = _fallback_plan(duration, text)
        return {
            "ok": True,
            "engine": "fallback",
            "audioDuration": duration,
            "scenes": scenes,
            "plannerDebug": {**planner_debug_base, "validation": validation},
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": "plan_built_without_audio_bytes" if not audio_bytes else "empty_scenes",
            "error": {
                "code": "GENERATION_FAILED",
                "hint": "empty_scenes",
                "modelUsed": model_used,
                "fallbackUsed": fallback_used,
            },
        }

    is_clip_mode = (scenario_key or "").strip().lower() == "clip"
    should_repair_for_weak_clip = bool(is_clip_mode and validation.get("weakClipPlan"))

    if validation.get("rejectedReason") or should_repair_for_weak_clip:
        if is_clip_mode:
            min_scenes = _minimum_scene_count_for_repair(duration)
            repair_instruction = f"""

REPAIR MODE: предыдущий storyboard требует доработки ({validation.get('rejectedReason') or 'weak_clip_plan'}).
Исправь план и верни новый валидный JSON.
ЖЁСТКИЕ ТРЕБОВАНИЯ:
- Минимум {min_scenes} сцен для этой длительности.
- Не оставляй сцены полностью пустыми по core fields (sceneText/imagePrompt/videoPrompt).
- Старайся сделать sceneText, imagePrompt и videoPrompt содержательными в каждой сцене.
- Каждая сцена должна быть конкретной и полезной для storyboard.
- Верни только валидный JSON на русском языке, без markdown.
"""
            repair_parts = [{"text": rules + extra + repair_instruction}]
            if audio_bytes:
                b64 = base64.b64encode(audio_bytes).decode("ascii")
                repair_parts.append({"inlineData": {"mimeType": audio_mime, "data": b64}})
            repair_body = {
                "contents": [{"role": "user", "parts": repair_parts}],
                "generationConfig": generation_config,
            }
            repair_resp = post_generate_content(settings.GEMINI_API_KEY, model_used, repair_body, timeout=120)
            repair_text = _extract_gemini_text(repair_resp if isinstance(repair_resp, dict) else {})
            repair_json = _parse_json_from_text(repair_text)
            repair_scenes = _normalize_scenes(duration, (repair_json or {}).get("scenes") or []) if isinstance(repair_json, dict) else []
            repair_validation = _validate_planner_scenes_quality(duration, scenario_key, repair_scenes)
            repair_validation["repairRetryUsed"] = True
            if should_repair_for_weak_clip and len(repair_scenes) == 1:
                repair_validation["warnings"] = list(repair_validation.get("warnings") or []) + ["weak_clip_plan_single_scene"]

            if repair_scenes and not repair_validation.get("rejectedReason") and not (should_repair_for_weak_clip and len(repair_scenes) == 1):
                return {
                    "ok": True,
                    "engine": "gemini",
                    "audioDuration": duration,
                    "scenes": repair_scenes,
                    "plannerDebug": {**planner_debug_base, "validation": repair_validation},
                    "modelUsed": model_used,
                    "fallbackUsed": fallback_used,
                    "hint": None if audio_bytes else "plan_built_without_audio_bytes",
                }

            if scenes:
                validation["repairRetryUsed"] = True
                if should_repair_for_weak_clip and len(repair_scenes) == 1:
                    validation["warnings"] = list(validation.get("warnings") or []) + ["weak_clip_plan_single_scene"]
                return {
                    "ok": True,
                    "engine": "gemini",
                    "audioDuration": duration,
                    "scenes": scenes,
                    "plannerDebug": {**planner_debug_base, "validation": validation},
                    "modelUsed": model_used,
                    "fallbackUsed": fallback_used,
                    "hint": None if audio_bytes else "plan_built_without_audio_bytes",
                }

            rejected_reason = repair_validation.get("rejectedReason") or "planner_output_rejected_as_low_quality"
            fallback_scenes = _fallback_plan(duration, text)
            return {
                "ok": True,
                "engine": "fallback",
                "audioDuration": duration,
                "scenes": fallback_scenes,
                "plannerDebug": {**planner_debug_base, "validation": repair_validation},
                "modelUsed": model_used,
                "fallbackUsed": fallback_used,
                "hint": "plan_built_without_audio_bytes" if not audio_bytes else f"planner_output_rejected_as_low_quality:{rejected_reason}",
                "error": {
                    "code": "GENERATION_FAILED",
                    "hint": f"planner_output_rejected_as_low_quality:{rejected_reason}",
                    "modelUsed": model_used,
                    "fallbackUsed": fallback_used,
                },
            }

        # Non-clip scenarios: keep validation in plannerDebug but avoid strict fallback.
        return {
            "ok": True,
            "engine": "gemini",
            "audioDuration": duration,
            "scenes": scenes,
            "plannerDebug": {**planner_debug_base, "validation": validation},
            "modelUsed": model_used,
            "fallbackUsed": fallback_used,
            "hint": None if audio_bytes else "plan_built_without_audio_bytes",
        }

    return {
        "ok": True,
        "engine": "gemini",
        "audioDuration": duration,
        "scenes": scenes,
        "plannerDebug": {**planner_debug_base, "validation": validation},
        "modelUsed": model_used,
        "fallbackUsed": fallback_used,
        "hint": None if audio_bytes else "plan_built_without_audio_bytes",
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
