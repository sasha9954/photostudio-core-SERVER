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
    style: RefUrlItem | list[RefUrlItem] | None = None


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

    def _normalize_ref_list(items, max_items: int = 8):
        out = []
        if not items:
            return out
        for it in items:
            if isinstance(it, dict):
                url = str(it.get("url") or "").strip()
            else:
                url = str(getattr(it, "url", "") or "").strip()
            if url:
                out.append(url)
        return out[:max_items]

    refs_obj = payload.refs
    character_refs = _normalize_ref_list((refs_obj.character if refs_obj else None) or payload.characterRefs)
    location_refs = _normalize_ref_list((refs_obj.location if refs_obj else None) or payload.locationRefs)
    props_refs = _normalize_ref_list((refs_obj.props if refs_obj else None) or payload.propsRefs)

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

    if not character_refs and payload.refCharacter:
        character_refs = [str(payload.refCharacter).strip()]
    if not location_refs and payload.refLocation:
        location_refs = [str(payload.refLocation).strip()]
    if not props_refs and payload.refItems:
        props_refs = [str(payload.refItems).strip()]
    if not style_refs and payload.refStyle:
        style_refs = [str(payload.refStyle).strip()]

    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "code": "GEMINI_API_KEY_MISSING",
                "detail": "Gemini API key is missing for clip planning",
            },
        )

    model_used = (settings.GEMINI_MODEL or "gemini-2.5-flash").strip()

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
}}"""

    user_input = {
        "mode": mode,
        "audioUrl": payload.audioUrl or "",
        "audioDurationHintSec": duration,
        "text": text,
        "refs": {
            "character": character_refs,
            "location": location_refs,
            "style": style_refs,
            "props": props_refs,
        },
    }

    parts = [
        {"text": system_rules},
        {"text": "Input payload:\n" + json.dumps(user_input, ensure_ascii=False)},
    ]
    if audio_bytes:
        parts.append({"inlineData": {"mimeType": audio_mime, "data": base64.b64encode(audio_bytes).decode("ascii")}})

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

    resp, raw_text, parsed = _call_gemini(parts, model_used)
    err_text = _combined_error_text(resp if isinstance(resp, dict) else {})
    if _is_model_unsupported_error(err_text):
        model_used = _pick_fallback_model(model_used)
        resp, raw_text, parsed = _call_gemini(parts, model_used)

    is_valid, reason = _validate_plan(parsed)
    if not is_valid:
        retry_parts = parts + [{"text": f"Previous output invalid ({reason}). Return ONLY one valid JSON object matching required schema."}]
        resp, raw_text, parsed = _call_gemini(retry_parts, model_used)
        is_valid, reason = _validate_plan(parsed)

    if not is_valid:
        err = _combined_error_text(resp if isinstance(resp, dict) else {}) or raw_text or reason or "invalid_gemini_json"
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "code": "CLIP_PLAN_INVALID_JSON",
                "detail": str(err)[:1200],
                "modelUsed": model_used,
                "hint": reason,
            },
        )

    plan = parsed
    track = plan.get("track") or {}
    audio_duration = float(track.get("durationSec") or duration or 30.0)
    scenes = plan.get("scenes") or []

    normalized_scenes = []
    for idx, s in enumerate(scenes):
        start = float(s.get("start"))
        end = float(s.get("end"))
        visual_prompt = str(s.get("visualPrompt") or "").strip()
        visual_desc = str(s.get("visualDescription") or "").strip()
        lip_sync_text = str(s.get("lipSyncText") or "").strip()
        scene_type = str(s.get("sceneType") or "visual_rhythm").strip() or "visual_rhythm"
        normalized_scenes.append({
            **s,
            "id": str(s.get("id") or f"scene_{idx + 1:03d}"),
            "start": start,
            "end": end,
            "prompt": visual_prompt or visual_desc,
            "sceneText": visual_desc,
            "imagePrompt": visual_prompt,
            "why": str(s.get("reason") or "").strip(),
            "sceneType": scene_type,
            "isLipSync": bool(lip_sync_text),
            "lipSyncText": lip_sync_text,
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
        "plannerDebug": {
            "audio": audio_debug,
            "validation": {
                "scenario": mode,
                "sceneCount": len(normalized_scenes),
                "rejectedReason": None,
                "repairRetryUsed": False,
                "warnings": [],
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
