from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)


def load_env_value(name: str) -> str:
    val = os.getenv(name)
    if val:
        return val.strip()

    env_candidates = [
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for env_file in env_candidates:
        if not env_file.exists():
            continue
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != name:
                    continue
                return v.strip().strip('"').strip("'")
        except Exception:
            continue
    return ""



def _try_read_local_static_asset(url: str) -> tuple[Optional[bytes], Optional[str]]:
    """If url points to our own /static/assets/... file, read it from disk to avoid self-HTTP deadlocks."""
    try:
        u = urlparse(url)
        p = (u.path or "").lstrip("/")
        if not p.startswith("static/assets/"):
            return None, None
        filename = p.split("static/assets/", 1)[1]
        if not filename:
            return None, None
        assets_dir = Path(__file__).resolve().parents[1] / "static" / "assets"
        fpath = (assets_dir / filename).resolve()
        # safety: ensure it's under assets_dir
        if assets_dir.resolve() not in fpath.parents and fpath != assets_dir.resolve():
            return None, None
        if not fpath.exists() or not fpath.is_file():
            return None, None
        ext = fpath.suffix.lstrip(".").lower() or None
        return fpath.read_bytes(), ext
    except Exception:
        return None, None

def _download_image_from_source(source_image: str) -> tuple[bytes, str]:
    src = (source_image or "").strip()
    if not src:
        raise ValueError("source_image is required")

    if src.startswith("data:"):
        try:
            header, payload = src.split(",", 1)
            payload = payload.strip()
            header_lower = header.lower()
            ext = "jpg"
            if "image/jpeg" in header_lower or "image/jpg" in header_lower:
                ext = "jpg"
            elif "image/png" in header_lower:
                ext = "png"
            elif "image/webp" in header_lower:
                ext = "webp"
            return base64.b64decode(payload), ext
        except Exception as exc:
            raise ValueError("Invalid source_image dataUrl") from exc

    if src.startswith("http://") or src.startswith("https://"):
        b, ext_local = _try_read_local_static_asset(src)
        if b is not None:
            return b, (ext_local or "jpg")

        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        ext = "jpg"
        if "image/jpeg" in ctype or "image/jpg" in ctype:
            ext = "jpg"
        elif "image/png" in ctype:
            ext = "png"
        elif "image/webp" in ctype:
            ext = "webp"
        return resp.content, ext

    raise ValueError("source_image must be dataUrl or http(s) url")

def _download_reference_images(sources: list[str]) -> list[dict]:
    """Download up to 3 images and convert to Veo referenceImages entries (inlineData)."""
    """Download up to 3 images and convert to Veo referenceImages entries (bytesBase64Encoded)."""
    out: list[dict] = []
    for src in (sources or [])[:3]:
        b, ext = _download_image_from_source(src)
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime = mime_map.get((ext or "").lower(), "image/jpeg")
        out.append(
            {
                "image": {"inlineData": {"mimeType": mime, "data": base64.b64encode(b).decode("utf-8")}},
                "image": {"bytesBase64Encoded": base64.b64encode(b).decode("utf-8"), "mimeType": mime},
                "referenceType": "asset",
            }
        )
    return out

def _download_file(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _extract_video_urls(data: dict) -> tuple[Optional[str], Optional[str]]:
    candidates_video = [
        data.get("videoUrl"),
        data.get("video_url"),
        data.get("url"),
        ((data.get("output") or {}).get("video_url") if isinstance(data.get("output"), dict) else None),
        ((data.get("result") or {}).get("video_url") if isinstance(data.get("result"), dict) else None),
    ]
    candidates_frame = [
        data.get("lastFrameUrl"),
        data.get("last_frame_url"),
        data.get("cover_url"),
        ((data.get("output") or {}).get("last_frame_url") if isinstance(data.get("output"), dict) else None),
        ((data.get("result") or {}).get("last_frame_url") if isinstance(data.get("result"), dict) else None),
    ]
    video_url = next((x for x in candidates_video if isinstance(x, str) and x.startswith("http")), None)
    frame_url = next((x for x in candidates_frame if isinstance(x, str) and x.startswith("http")), None)
    return video_url, frame_url


def _ensure_video_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    videos_dir = root / "static" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir


def _save_video_locally(video_bytes: bytes, job_id: Optional[str] = None) -> tuple[str, Path, str]:
    ts = int(time.time() * 1000)
    resolved_job_id = (job_id or f"job_{ts}").strip()
    videos_dir = _ensure_video_dir()
    file_name = f"{resolved_job_id}.mp4"
    file_path = videos_dir / file_name
    file_path.write_bytes(video_bytes)
    return f"/static/videos/{file_name}", file_path, resolved_job_id


def _save_veo_video_locally(video_bytes: bytes, job_id: Optional[str] = None) -> tuple[str, Path, str]:
    ts = int(time.time() * 1000)
    resolved_job_id = (job_id or f"job_{ts}").strip()
    videos_dir = _ensure_video_dir()
    file_name = f"{resolved_job_id}.mp4"
    file_path = videos_dir / file_name
    file_path.write_bytes(video_bytes)
    return f"/static/videos/{file_name}", file_path, resolved_job_id


def _extract_last_frame(video_path: Path, ts: int) -> tuple[str, str]:
    if shutil.which("ffmpeg") is None:
        return "", "last frame extraction skipped: ffmpeg is not installed"

    videos_dir = _ensure_video_dir()
    frame_name = f"frame_{ts}.png"
    frame_path = videos_dir / frame_name

    cmd = [
        "ffmpeg",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-y",
        str(frame_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return "", "last frame extraction failed: ffmpeg failed to extract last frame"

    return f"/static/videos/{frame_name}", ""


def _kling_request(image_bytes: bytes, image_ext: str, fmt: str, camera: str, prompt: str, seconds: int, api_key: str) -> tuple[bytes, Optional[bytes]]:
    upload_url = os.getenv("KIE_UPLOAD_URL", "https://kieai.redpandaai.co/api/file-base64-upload")
    create_url = os.getenv("KIE_CREATE_TASK_URL", "https://api.kie.ai/api/v1/jobs/createTask")
    details_url = os.getenv("KIE_TASK_DETAILS_URL", "https://api.kie.ai/api/v1/jobs/recordInfo")
    upload_path = os.getenv("KIE_UPLOAD_PATH", "images/photostudio")
    timeout_s = int(os.getenv("KIE_POLL_TIMEOUT_SECONDS", "300"))
    interval_s = int(os.getenv("KIE_POLL_INTERVAL_SECONDS", "4"))
    logger.info("KIE endpoints upload_url=%s create_url=%s details_url=%s", upload_url, create_url, details_url)

    req_prompt = f"{prompt}\nCamera move: {camera}. Duration: {seconds}s. Format: {fmt}."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if image_ext not in ("jpg", "png", "webp"):
        image_ext = "jpg"

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    if image_ext == "jpg":
        mime = "image/jpeg"
    elif image_ext == "png":
        mime = "image/png"
    elif image_ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    # IMPORTANT: use a unique filename per request.
    # Some providers/CDNs may cache content by path+filename; reusing the same
    # name can lead to "old" images being used even when bytes changed.
    uniq = int(time.time() * 1000)
    file_name = f"source_{uniq}.{image_ext}"

    # Debug: hash the image bytes so we can prove what image was sent.
    img_hash = hashlib.sha1(image_bytes).hexdigest()[:12]

    upload_payload = {
        "base64Data": f"data:{mime};base64,{b64}",
        "uploadPath": upload_path,
        "fileName": file_name,
    }
    logger.info(
        "KIE upload start ext=%s bytes_len=%s sha1=%s fileName=%s upload_url=%s",
        image_ext,
        len(image_bytes),
        img_hash,
        file_name,
        upload_url,
    )
    upload_resp = requests.post(upload_url, headers=headers, json=upload_payload, timeout=60)
    logger.debug("KIE file upload status=%s request_id=%s response=%s", upload_resp.status_code, upload_resp.headers.get("x-request-id") or upload_resp.headers.get("request-id"), upload_resp.text[:500])
    _raise_kie_error(upload_resp, "file-base64-upload")
    upload_data = upload_resp.json()
    image_url = None
    if isinstance(upload_data.get("data"), dict):
        image_url = upload_data["data"].get("downloadUrl") or upload_data["data"].get("url")
    if not image_url:
        raise RuntimeError(f"KIE file upload succeeded but image url is missing: {str(upload_data)[:400]}")
    logger.info("KIE upload done ext=%s bytes_len=%s image_url=%s", image_ext, len(image_bytes), image_url[:120])

    task_payload = {
        "model": "kling-2.6/image-to-video",
        "input": {
            "prompt": req_prompt,
            "image_urls": [image_url],
            "sound": False,
            "duration": str(seconds),
        },
    }
    create_resp = requests.post(create_url, headers=headers, json=task_payload, timeout=60)
    logger.debug("KIE createTask status=%s request_id=%s response=%s", create_resp.status_code, create_resp.headers.get("x-request-id") or create_resp.headers.get("request-id"), create_resp.text[:500])
    _raise_kie_error(create_resp, "createTask")

    create_data = create_resp.json()
    task_id = _extract_task_id(create_data)
    if not task_id:
        raise RuntimeError(f"KIE createTask did not return taskId: {str(create_data)[:400]}")

    started = time.time()
    while time.time() - started < timeout_s:
        detail_resp = requests.get(details_url, headers=headers, params={"taskId": task_id}, timeout=60)
        _raise_kie_error(detail_resp, "getTaskDetails")
        detail_data = detail_resp.json()

        status = _extract_task_status(detail_data)
        provider_request_id = detail_resp.headers.get("x-request-id") or detail_resp.headers.get("request-id")
        logger.debug(
            "KIE task poll taskId=%s status=%s request_id=%s response=%s",
            task_id,
            status,
            provider_request_id,
            str(detail_data)[:500],
        )

        if status in {"success", "succeeded", "done", "completed"}:
            video_url, _ = _extract_video_urls(detail_data)
            if not video_url:
                nested_data = detail_data.get("data") if isinstance(detail_data, dict) else None
                if isinstance(nested_data, dict):
                    video_url, _ = _extract_video_urls(nested_data)
                    if not video_url:
                        result_json_str = nested_data.get("resultJson") or nested_data.get("result_json") or nested_data.get("result")
                        if isinstance(result_json_str, str):
                            result_json_trimmed = result_json_str.strip()
                            if result_json_trimmed.startswith("{") or result_json_trimmed.startswith("["):
                                try:
                                    parsed_obj = json.loads(result_json_trimmed)
                                except Exception:
                                    parsed_obj = None
                                if isinstance(parsed_obj, dict):
                                    result_urls = parsed_obj.get("resultUrls") or parsed_obj.get("result_urls")
                                    if isinstance(result_urls, list) and result_urls:
                                        first_url = result_urls[0]
                                        if isinstance(first_url, str) and first_url.startswith(("http://", "https://")):
                                            video_url = first_url
                                if not video_url and isinstance(parsed_obj, dict):
                                    video_url, _ = _extract_video_urls(parsed_obj)
            if not video_url:
                logger.debug(
                    "KIE success but video url missing. detail_keys=%s data_keys=%s",
                    list(detail_data.keys()) if isinstance(detail_data, dict) else [],
                    list((detail_data.get("data") or {}).keys()) if isinstance(detail_data, dict) and isinstance(detail_data.get("data"), dict) else [],
                )
                raise RuntimeError(f"KIE task succeeded but video url is missing: {str(detail_data)[:500]}")
            return _download_file(video_url), None

        if status in {"failed", "error", "canceled", "cancelled"}:
            raise RuntimeError(f"KIE task failed for taskId={task_id}: {str(detail_data)[:500]}")

        time.sleep(interval_s)

    raise RuntimeError(f"KIE task polling timeout for taskId={task_id} after {timeout_s}s")


def _raise_kie_error(resp: requests.Response, endpoint_name: str) -> None:
    if resp.status_code < 400:
        return
    request_id = resp.headers.get("x-request-id") or resp.headers.get("request-id")
    logger.debug(
        "KIE %s error status=%s request_id=%s response=%s",
        endpoint_name,
        resp.status_code,
        request_id,
        resp.text[:500],
    )
    if resp.status_code == 401:
        raise RuntimeError(f"KIE API {endpoint_name} unauthorized (401): check KLING_API_KEY bearer token. {resp.text[:300]}")
    if resp.status_code == 402:
        raise RuntimeError(f"KIE API {endpoint_name} payment required/quota exceeded (402). {resp.text[:300]}")
    if resp.status_code == 422:
        raise RuntimeError(f"KIE API {endpoint_name} invalid request payload (422). {resp.text[:300]}")
    if resp.status_code == 429:
        raise RuntimeError(f"KIE API {endpoint_name} rate limited (429): retry later. {resp.text[:300]}")
    raise RuntimeError(f"KIE API {endpoint_name} error {resp.status_code}: {resp.text[:400]}")


def _extract_task_id(data: dict) -> Optional[str]:
    candidates = [
        data.get("taskId"),
        data.get("task_id"),
        ((data.get("data") or {}).get("taskId") if isinstance(data.get("data"), dict) else None),
        ((data.get("data") or {}).get("task_id") if isinstance(data.get("data"), dict) else None),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_task_status(data: dict) -> str:
    candidates = [
        data.get("status"),
        data.get("state"),
        ((data.get("data") or {}).get("status") if isinstance(data.get("data"), dict) else None),
        ((data.get("data") or {}).get("state") if isinstance(data.get("data"), dict) else None),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "pending"


def _veo_request(
    image_bytes: Optional[bytes],
    image_ext: str,
    fmt: str,
    prompt: str,
    seconds: int,
    api_key: str,
    reference_sources: Optional[list[str]] = None,
) -> tuple[bytes, Optional[bytes]]:
    """
    Gemini Veo predictLongRunning.

    Supports:
    - image-to-video: pass image_bytes
    - reference-to-video (up to 3 refs): pass reference_sources (list of dataUrl/http urls)
      See: https://ai.google.dev/gemini-api/docs/video (referenceImages)
    """
    aspect_ratio = fmt if fmt in {"9:16", "1:1", "16:9"} else "9:16"
    # Current limitation: Veo 3.1 referenceImages support only 16:9 aspect ratio (Gemini API forums).
    if reference_sources and aspect_ratio != "16:9":
        raise RuntimeError(            f"VEO_REF_ASPECT_INVALID: referenceImages currently support only 16:9; got {aspect_ratio}. "            "Switch format to 16:9 or remove extra reference images."        )

    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    predict_url = f"{base_url}/models/veo-3.1-generate-preview:predictLongRunning"

    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    image_mime = mime_map.get((image_ext or "").lower(), "image/jpeg")

    # Build payload in the Vertex-style predictLongRunning format.
    # For reference images, Gemini docs use parameters.referenceImages with inlineData.
    # For predictLongRunning (Vertex-style), image data uses bytesBase64Encoded (NOT inlineData).
    payload: dict = {
        "instances": [
            {
                "prompt": prompt,
            }
        ],
        "parameters": {
            "aspectRatio": aspect_ratio,
            "durationSeconds": seconds,
            "sampleCount": 1,
        },
    }

    reference_images = _download_reference_images(reference_sources or []) if reference_sources else []
    if reference_images:
        # Per docs: durationSeconds must be 8 when using referenceImages.
        if seconds != 8:
            raise RuntimeError(f"VEO_REF_DURATION_INVALID: durationSeconds must be 8 when using referenceImages; got {seconds}")
        payload["parameters"]["referenceImages"] = reference_images
        # NOTE: predictLongRunning uses Vertex-style payload: referenceImages live in instances[0]
        payload["instances"][0]["referenceImages"] = reference_images
    else:
        # Back-compat: image-to-video (single start image) using instances.image
        if image_bytes:
            payload["instances"][0]["image"] = {
                "mimeType": image_mime,
                "bytesBase64Encoded": base64.b64encode(image_bytes).decode("utf-8"),
            }

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    request_start = time.time()
    resp = requests.post(predict_url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini Veo predictLongRunning error {resp.status_code}: {resp.text[:400]}")
    created = resp.json()
    operation_name = created.get("name")
    if not isinstance(operation_name, str) or not operation_name.strip():
        raise RuntimeError(f"Gemini Veo operation name missing in response: {str(created)[:500]}")
    operation_name = operation_name.strip()

    poll_timeout_seconds = int(os.getenv("VEO_POLL_TIMEOUT_SECONDS", "720"))
    poll_interval_seconds = int(os.getenv("VEO_POLL_INTERVAL_SECONDS", "10"))
    logger.info("Gemini Veo operation started name=%s", operation_name)

    poll_url = urljoin(f"{base_url}/", operation_name)
    started = time.time()
    while time.time() - started < poll_timeout_seconds:
        status_resp = requests.get(poll_url, headers={"x-goog-api-key": api_key}, timeout=60)
        if status_resp.status_code >= 400:
            raise RuntimeError(f"Gemini Veo operation poll error {status_resp.status_code}: {status_resp.text[:400]}")
        status_json = status_resp.json()
        if status_json.get("done") is True:
            if isinstance(status_json.get("error"), dict):
                err = status_json.get("error") or {}
                raise RuntimeError(f"Gemini Veo operation failed: {err.get('message') or str(err)[:400]}")
            try:
                response_obj = status_json.get("response") if isinstance(status_json.get("response"), dict) else {}
                gvr = response_obj.get("generateVideoResponse") if isinstance(response_obj.get("generateVideoResponse"), dict) else {}
                generated_samples = gvr.get("generatedSamples") if isinstance(gvr.get("generatedSamples"), list) else []
                if not generated_samples:
                    filtered_count = gvr.get("raiMediaFilteredCount") or 0
                    reasons = gvr.get("raiMediaFilteredReasons") if isinstance(gvr.get("raiMediaFilteredReasons"), list) else []
                    first_reason = next((r for r in reasons if isinstance(r, str) and r.strip()), "")
                    first_reason = (first_reason.strip()[:300] if first_reason else "generatedSamples missing/empty")
                    raise RuntimeError(f"VEO_FILTERED: count={filtered_count} reason={first_reason}")
                video_uri = generated_samples[0]["video"]["uri"]
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Gemini Veo operation completed but video uri missing: {str(status_json)[:500]}") from exc

            video_resp = requests.get(video_uri, headers={"x-goog-api-key": api_key}, timeout=180, allow_redirects=True)
            if video_resp.status_code >= 400:
                raise RuntimeError(f"Gemini Veo video download error {video_resp.status_code}: {video_resp.text[:400]}")
            elapsed = int(time.time() - request_start)
            logger.info("Gemini Veo operation completed name=%s elapsed=%ss", operation_name, elapsed)
            return video_resp.content, None

        logger.debug("Gemini Veo operation pending name=%s elapsed=%ss", operation_name, int(time.time() - started))
        time.sleep(max(8, min(12, poll_interval_seconds)))

    raise TimeoutError("VIDEO_TIMEOUT")
def generate_video(kind: str, source_image: str, fmt: str, model: str, camera: str, prompt: str, seconds: int, lighting: str = "soft") -> dict:
    try:
        job_id = f"job_{int(time.time() * 1000)}"
        if kind != "video_from_image":
            return {"ok": False, "code": "INVALID_KIND", "message": "kind must be 'video_from_image'"}

        # Allow passing multiple reference images for Veo (up to 3).
        # Frontend may send a JSON array string in source_image, e.g. ["url1","url2","url3"].
        sources_list: list[str] | None = None
        if isinstance(source_image, list):
            sources_list = [str(x) for x in source_image if x]
        elif isinstance(source_image, str):
            s = source_image.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        sources_list = [str(x) for x in arr if x]
                except Exception:
                    sources_list = None

        primary_source = (sources_list[0] if sources_list else source_image)
        image_bytes, image_ext = _download_image_from_source(primary_source)

        # Lighting (safe presets)
        lighting_key = (lighting or "soft").strip().lower()
        if lighting_key == "soft":
            lighting_prompt = "Soft environment lighting with natural bounce and physically grounded contact shadows"
        elif lighting_key == "contrast":
            lighting_prompt = "High-contrast environmental lighting with coherent shadow direction and grounded contact shadows"
        elif lighting_key == "warm":
            lighting_prompt = "Warm evening environmental lighting with realistic ambient bounce and contact shadows"
        else:
            lighting_prompt = "Neutral environment-driven lighting consistent with scene sources"

        # Prepend lighting so both Kling and Veo see it.
        world_lock = "Relight all characters and props to match the same scene environment. Ignore source/reference lighting. Keep consistent light direction, color temperature, ambient bounce, reflections, atmospheric diffusion, and contact shadows. No studio or invisible lights."
        prompt = f"{lighting_prompt}. {world_lock}\n{prompt or ''}".strip()

        if model == "classic":
            api_key = load_env_value("KLING_API_KEY")
            if not api_key:
                return {"ok": False, "code": "MISSING_KLING_API_KEY", "message": "KLING_API_KEY is missing in environment/.env"}
            if seconds not in (5, 10):
                return {
                    "ok": False,
                    "code": "INVALID_DURATION",
                    "message": f"Classic (Kling-2.6) supports duration only 5 or 10 seconds; got {seconds}",
                }
            video_bytes, _ = _kling_request(image_bytes, image_ext, fmt, camera, prompt, seconds, api_key)
            video_url, video_path, resolved_job_id = _save_video_locally(video_bytes, job_id=job_id)
            last_frame_url, warning = _extract_last_frame(video_path, int(time.time() * 1000))
            return {
                "ok": True,
                "jobId": resolved_job_id,
                "videoUrl": video_url,
                "lastFrameUrl": last_frame_url,
                "warning": warning,
            }

        if model == "premium":
            api_key = load_env_value("GEMINI_API_KEY")
            if not api_key:
                return {"ok": False, "code": "MISSING_GEMINI_API_KEY", "message": "GEMINI_API_KEY is missing in environment/.env"}

            # If we received multiple sources, use them as Veo referenceImages (up to 3).
            # Veo referenceImages require durationSeconds=8 in Gemini API docs.
            ref_sources = sources_list if (sources_list and len(sources_list) > 1) else None
            effective_seconds = seconds
            if ref_sources and effective_seconds != 8:
                # Auto-fix to 8s to keep UI simple (user can still show 5s/10s presets on UI).
                effective_seconds = 8

            video_bytes, _ = _veo_request(image_bytes if not ref_sources else None, image_ext, fmt, prompt, effective_seconds, api_key, reference_sources=ref_sources)
            video_url, video_path, resolved_job_id = _save_veo_video_locally(video_bytes, job_id=job_id)
            last_frame_url, warning = _extract_last_frame(video_path, int(time.time() * 1000))
            return {
                "ok": True,
                "jobId": resolved_job_id,
                "provider": "veo_gemini",
                "video_url": video_url,
                "videoUrl": video_url,
                "lastFrameUrl": last_frame_url,
                "warning": warning,
                "meta": {
                    "format": fmt,
                    "seconds": effective_seconds,
                    "camera": camera,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "file": f"{resolved_job_id}.mp4",
                },
            }

        return {"ok": False, "code": "INVALID_MODEL", "message": "model must be 'classic' or 'premium'"}

    except TimeoutError:
        return {
            "ok": False,
            "code": "VIDEO_TIMEOUT",
            "hint": "Retry later or inspect provider/backend logs.",
            "message": "Video generation timed out while waiting for Gemini Veo operation.",
        }
    except Exception as exc:
        logger.exception("video generation failed model=%s", model)
        msg = str(exc)
        if msg.startswith("VEO_FILTERED"):
            return {
                "ok": False,
                "code": "VEO_FILTERED",
                "message": "Veo не смог создать видео (фильтр или processing).",
                "hint": "Упростите промт, уберите упоминания музыки/голоса и добавьте 'No audio. Silent video.'",
                "debug": msg,
            }
        return {
            "ok": False,
            "code": "VIDEO_GENERATION_FAILED",
            "message": msg,
        }


def _local_video_path_from_url(video_url: str) -> str | None:
    if not video_url:
        return None
    s = str(video_url)
    # Expect our own /static/videos/... urls
    if "/static/videos/" in s:
        name = s.split("/static/videos/", 1)[1].split("?", 1)[0]
        videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "videos"))
        return os.path.join(videos_dir, name)
    return None


def concat_videos(clip_urls: list[str], fmt: str = "9:16") -> dict:
    """Concat local mp4 clips with ffmpeg (concat demuxer)."""
    try:
        if not clip_urls or len(clip_urls) < 2:
            return {"ok": False, "code": "NEED_2", "message": "Need at least 2 clips"}

        paths = []
        for u in clip_urls:
            p = _local_video_path_from_url(u)
            if not p or not os.path.exists(p):
                return {"ok": False, "code": "MISSING_CLIP", "message": f"Clip not found: {u}"}
            paths.append(p)

        out_job = f"merge_{int(time.time()*1000)}"
        videos_dir = _ensure_video_dir()
        out_path = os.path.join(videos_dir, f"{out_job}.mp4")

        # concat demuxer list file
        list_path = os.path.join(videos_dir, f"{out_job}.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in paths:
                # ffmpeg concat wants file lines with escaped paths
                esc = p.replace("'", "'\\\\''")
                f.write("file '%s'\n" % esc)

        # Run ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return {"ok": False, "code": "FFMPEG_MISSING", "message": "ffmpeg not found in PATH. Install ffmpeg and restart backend."}

        if proc.returncode != 0 or not os.path.exists(out_path):
            err = (proc.stderr or proc.stdout or "").strip()
            return {"ok": False, "code": "FFMPEG_ERROR", "message": err[:400] or "ffmpeg concat failed"}

        base_url = settings.PUBLIC_BASE_URL.rstrip("/")
        video_url = f"{base_url}/static/videos/{os.path.basename(out_path)}"
        last_frame_url, warning = _extract_last_frame(out_path, int(time.time() * 1000))
        return {"ok": True, "videoUrl": video_url, "lastFrameUrl": last_frame_url, "warning": warning}
    finally:
        try:
            pass
        except Exception:
            pass