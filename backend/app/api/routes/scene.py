from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
import json
import threading
import uuid
from datetime import timezone

import os
import re
import base64
import hashlib
from typing import List, Optional, Any

from app.core.config import settings
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url
from app.engine.scene_engine import create_asset
from app.engine.media_io import fetch_url_to_bytes, bytes_to_b64, sniff_mime_from_bytes

from app.core.tokens import verify_token
from app.db.sqlite import db
from app.services.auth_service import add_ledger

COOKIE_NAME = "ps_token"
router = APIRouter()

WORLD_LIGHTING_POLICY = (
    "World lighting consistency: all visible elements (characters, props, tools, held objects, furniture, vehicles, "
    "environment objects) must use the SAME environmental lighting model. "
    "Ignore lighting from reference images and relight all subjects from the current environment only. "
    "Match light direction, color temperature, ambient bounce, shadow softness and direction, atmospheric diffusion, "
    "environmental reflections, and environment color contamination. "
    "All grounded objects must cast visible contact shadows and must not look floating. "
    "Props must look physically integrated: correct reflections, shadows, bounce, atmospheric depth, and surface interaction "
    "(dust/dirt/grounding where appropriate). "
    "Keep prop category behavior and realistic scale consistent across shots. "
    "No studio-style invisible lighting; lighting sources must be explainable by the scene environment."
)

def _current_user_id(req: Request) -> str:
    tok = req.cookies.get(COOKIE_NAME)
    if not tok:
        raise HTTPException(status_code=401, detail="Not authenticated")
    v = verify_token(tok)
    uid = v[0] if v else None
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token")
    return uid

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _default_scene() -> dict:
    return {
        "modelUrl": None,
        "locationUrl": None,
        "modelDetails": {},      # { head|torso|legs|shoes|accessories : url }
        "locationDetails": [None, None, None, None, None],   # 5 slots (urls or null)
    }



def _normalize_format(fmt: str | None) -> str:
    v = (fmt or "").strip()
    if v in ("9:16", "1:1", "16:9"):
        return v
    return "9:16"

def _format_hint(fmt: str, kind: str) -> str:
    # Minimal, prompt-safe hint for aspect ratio.
    # 9:16 = vertical portrait, 1:1 = square, 16:9 = wide landscape.
    if fmt == "1:1":
        return "Square 1:1 aspect ratio."
    if fmt == "16:9":
        return "Wide landscape 16:9 aspect ratio."
    return "Vertical portrait 9:16 aspect ratio."

class ScenePatchIn(BaseModel):
    modelUrl: str | None = None
    locationUrl: str | None = None
    modelDetails: dict | None = None
    locationDetails: dict | list | None = None

@router.get("/scene/current")
def get_or_create_current_scene(req: Request):
    uid = _current_user_id(req)

    with db() as con:
        row = con.execute("SELECT data, updated_at FROM scenes WHERE user_id = ?", (uid,)).fetchone()
        if row:
            data = json.loads(row["data"]) if row["data"] else _default_scene()
            return {"scene": data, "updated_at": row["updated_at"]}

        data = _default_scene()
        con.execute(
            "INSERT INTO scenes(user_id, data, updated_at) VALUES(?,?,?)",
            (uid, json.dumps(data, ensure_ascii=False), _now_iso()),
        )
        return {"scene": data, "updated_at": _now_iso()}

@router.patch("/scene/current")
def patch_current_scene(req: Request, body: ScenePatchIn):
    uid = _current_user_id(req)

    # Важно: нужно уметь "очищать" поля (ставить null).
    # Поэтому проверяем не "is not None", а "было ли поле передано".
    fields_set = set()
    try:
        # Pydantic v2
        fields_set = set(getattr(body, "model_fields_set", set()) or set())
    except Exception:
        fields_set = set()
    if not fields_set:
        try:
            # Pydantic v1
            fields_set = set(getattr(body, "__fields_set__", set()) or set())
        except Exception:
            fields_set = set()

    with db() as con:
        row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
        data = json.loads(row["data"]) if row and row["data"] else _default_scene()

        if "modelUrl" in fields_set:
            data["modelUrl"] = body.modelUrl
        if "locationUrl" in fields_set:
            data["locationUrl"] = body.locationUrl
        if "modelDetails" in fields_set and body.modelDetails is not None:
            data["modelDetails"] = body.modelDetails
        if "locationDetails" in fields_set and body.locationDetails is not None:
            data["locationDetails"] = body.locationDetails

        ts = _now_iso()
        if row:
            con.execute("UPDATE scenes SET data=?, updated_at=? WHERE user_id=?", (json.dumps(data, ensure_ascii=False), ts, uid))
        else:
            con.execute("INSERT INTO scenes(user_id, data, updated_at) VALUES(?,?,?)", (uid, json.dumps(data, ensure_ascii=False), ts))

        return {"scene": data, "updated_at": ts}


# -----------------------
# Scene engine endpoints (model/location generation + apply details)
# -----------------------



# -----------------------
# Scene jobs (server-side) — so generation continues after leaving page
# -----------------------

def _job_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _scene_job_create(uid: str, kind: str, action: str) -> str:
    job_id = f"sc_{uuid.uuid4().hex[:16]}"
    now = _job_now_iso()
    with db() as con:
        con.execute(
            "INSERT INTO scene_jobs(job_id, user_id, kind, action, state, progress, result_json, error, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job_id, uid, kind, action, "queued", 0, None, None, now, now),
        )
    return job_id

def _scene_job_update(job_id: str, **fields):
    if not job_id:
        return
    allowed = {"state", "progress", "result_json", "error"}
    sets = []
    vals = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k}=?")
        vals.append(v)
    sets.append("updated_at=?")
    vals.append(_job_now_iso())
    vals.append(job_id)
    with db() as con:
        con.execute(f"UPDATE scene_jobs SET {', '.join(sets)} WHERE job_id=?", tuple(vals))

def _scene_job_get(uid: str, job_id: str) -> dict | None:
    with db() as con:
        row = con.execute(
            "SELECT job_id, user_id, kind, action, state, progress, result_json, error, created_at, updated_at FROM scene_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        if row["user_id"] != uid:
            return None
        out = dict(row)
        try:
            out["result"] = json.loads(out.get("result_json") or "null")
        except Exception:
            out["result"] = None
        out.pop("result_json", None)
        return out

_DATAURL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)

def _ensure_assets_dir():
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

def _dataurl_to_bytes(data_url: str) -> tuple[bytes, str]:
    m = _DATAURL_RE.match((data_url or "").strip())
    if not m:
        raise ValueError("Invalid dataUrl")
    mime = (m.group("mime") or "image/png").strip()
    b64 = (m.group("data") or "").strip()
    raw = base64.b64decode(b64, validate=True)
    return raw, mime

def _bytes_to_dataurl(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{bytes_to_b64(raw)}"

def _url_to_dataurl(url: str) -> str:
    raw, ct = fetch_url_to_bytes(url)
    mime = (ct or "").split(";", 1)[0].strip().lower() or sniff_mime_from_bytes(raw)
    return _bytes_to_dataurl(raw, mime)

def _save_dataurl_to_asset_url(data_url: str) -> str:
    raw, mime = _dataurl_to_bytes(data_url)
    _ensure_assets_dir()
    h = hashlib.sha256(raw).hexdigest()[:16]
    ext = _guess_ext(mime)
    fn = f"{h}{ext}"
    path = os.path.join(ASSETS_DIR, fn)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    return asset_url(fn)

class SceneGenerateIn(BaseModel):
    kind: str  # "model" | "location"
    prompt: str = ""
    # Optional: reuse existing image as base (URL). If omitted, model creates from scratch.
    baseUrl: Optional[str] = None

class SceneApplyDetailsIn(BaseModel):
    kind: str  # "model" | "location"
    # If omitted, backend will use current scene modelUrl/locationUrl as base
    baseUrl: Optional[str] = None
    # list of URLs (static/assets...) of detail images
    detailUrls: List[str] = []
    # optional extra text hint
    prompt: str = ""

@router.post("/scene/generate")
def scene_generate(req: Request, body: SceneGenerateIn):
    """Generate model/location image via Gemini and return stored asset URL."""
    uid = _current_user_id(req)

    kind = (body.kind or "").strip().lower()
    if kind not in ("model", "location"):
        raise HTTPException(status_code=400, detail="kind must be 'model' or 'location'")

    base_dataurl = None
    if body.baseUrl:
        base_dataurl = _url_to_dataurl(body.baseUrl)

    prompt = (body.prompt or "").strip()
    # minimal safe prompt wrapper
    full_prompt = prompt if prompt else ("Create a photorealistic fashion model" if kind == "model" else "Create a photorealistic fashion location background")

    try:
        out_dataurl = create_asset(kind=kind, prompt=full_prompt, base_image=base_dataurl, details=[])
        asset_url = _save_dataurl_to_asset_url(out_dataurl)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SCENE_GENERATE_FAILED: {e}")

    # persist into current scene
    with db() as con:
        row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
        data = json.loads(row["data"]) if row and row["data"] else _default_scene()
        if kind == "model":
            data["modelUrl"] = asset_url
        else:
            data["locationUrl"] = asset_url
        ts = _now_iso()
        if row:
            con.execute("UPDATE scenes SET data=?, updated_at=? WHERE user_id=?", (json.dumps(data, ensure_ascii=False), ts, uid))
        else:
            con.execute("INSERT INTO scenes(user_id, data, updated_at) VALUES(?,?,?)", (uid, json.dumps(data, ensure_ascii=False), ts))
    return {"ok": True, "url": asset_url}

@router.post("/scene/applyDetails")
def scene_apply_details(req: Request, body: SceneApplyDetailsIn):
    """Apply detail reference images to existing model/location via Gemini and return new stored asset URL."""
    uid = _current_user_id(req)

    kind = (body.kind or "").strip().lower()
    if kind not in ("model", "location"):
        raise HTTPException(status_code=400, detail="kind must be 'model' or 'location'")

    # Load current scene to pick baseUrl if not provided
    with db() as con:
        row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
        sc = json.loads(row["data"]) if row and row["data"] else _default_scene()

    base_url = body.baseUrl or (sc.get("modelUrl") if kind == "model" else sc.get("locationUrl"))
    if not base_url:
        raise HTTPException(status_code=400, detail="No base image. Generate model/location first.")

    base_dataurl = _url_to_dataurl(base_url)

    detail_urls = [u for u in (body.detailUrls or []) if isinstance(u, str) and u.strip()]
    if not detail_urls:
        raise HTTPException(status_code=400, detail="No detailUrls provided")

    detail_dataurls = []
    for u in detail_urls:
        detail_dataurls.append(_url_to_dataurl(u))

    prompt = (body.prompt or "").strip()
    # strict prompt to preserve identity
    if kind == "model":
        base_prompt = "Apply the provided detail reference images to the SAME person and outfit. Keep identity, face, body, pose, and background unchanged. Improve realism and match details only."
    else:
        base_prompt = "Apply the provided detail reference images to the SAME location scene. Keep camera, composition, lighting and all other elements unchanged. Match details only."
    full_prompt = (base_prompt + " " + WORLD_LIGHTING_POLICY + " " + prompt).strip()

    try:
        out_dataurl = create_asset(kind=kind, prompt=full_prompt, base_image=base_dataurl, details=detail_dataurls)
        asset_url = _save_dataurl_to_asset_url(out_dataurl)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SCENE_APPLY_DETAILS_FAILED: {e}")

    # persist: update base image url, and also persist details into scene data (so reload keeps them)
    if kind == "model":
        sc["modelUrl"] = asset_url
    else:
        sc["locationUrl"] = asset_url
    ts = _now_iso()
    with db() as con:
        con.execute("UPDATE scenes SET data=?, updated_at=? WHERE user_id=?", (json.dumps(sc, ensure_ascii=False), ts, uid))
    return {"ok": True, "url": asset_url}



@router.get("/scene/jobs/{job_id}")
def scene_get_job(req: Request, job_id: str):
    uid = _current_user_id(req)
    job = _scene_job_get(uid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}




class SceneGenerateJobIn(BaseModel):
    kind: str
    baseUrl: Optional[str] = None
    prompt: str = ""
    format: Optional[str] = "9:16"

@router.post("/scene/generateJob")
def scene_generate_job(req: Request, body: SceneGenerateJobIn):
    uid = _current_user_id(req)

    kind = (body.kind or "").strip().lower()
    if kind not in ("model", "location"):
        raise HTTPException(status_code=400, detail="kind must be 'model' or 'location'")

    job_id = _scene_job_create(uid, kind, "generate")

    def _run():
        try:
            _scene_job_update(job_id, state="running", progress=5)
            spent = False
            # Spend 1 credit ONCE per job (idempotent via ref=job_id)
            try:
                add_ledger(uid, -1, f"SCENE_CREATE_{kind.upper()}", ref=job_id)
                spent = True
            except ValueError as e:
                _scene_job_update(job_id, state="error", error=str(e), progress=100)
                return

            base_dataurl = None
            if body.baseUrl:
                base_dataurl = _url_to_dataurl(body.baseUrl)

            prompt = (body.prompt or "").strip()
            seed_prompt = prompt if prompt else ("Create a photorealistic fashion model" if kind == "model" else "Create a photorealistic fashion location background")
            full_prompt = (WORLD_LIGHTING_POLICY + " " + seed_prompt).strip()
            fmt = _normalize_format(body.format)
            full_prompt = (full_prompt + " " + _format_hint(fmt, kind)).strip()

            _scene_job_update(job_id, progress=35)
            out_dataurl = create_asset(kind=kind, prompt=full_prompt, base_image=base_dataurl, details=[])
            _scene_job_update(job_id, progress=70)
            asset_url = _save_dataurl_to_asset_url(out_dataurl)

            # persist into current scene
            with db() as con:
                row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
                data = json.loads(row["data"]) if row and row["data"] else _default_scene()
                if kind == "model":
                    data["modelUrl"] = asset_url
                else:
                    data["locationUrl"] = asset_url
                ts = _now_iso()
                if row:
                    con.execute("UPDATE scenes SET data=?, updated_at=? WHERE user_id=?", (json.dumps(data, ensure_ascii=False), ts, uid))
                else:
                    con.execute("INSERT INTO scenes(user_id, data, updated_at) VALUES(?,?,?)", (uid, json.dumps(data, ensure_ascii=False), ts))

            _scene_job_update(job_id, state="done", progress=100, result_json=json.dumps({"url": asset_url}, ensure_ascii=False))
        except Exception as e:
            try:
                if spent:
                    add_ledger(uid, +1, "REFUND", ref=f"SCENE:{job_id}")
            except Exception:
                pass
            _scene_job_update(job_id, state="error", error=str(e), progress=100)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"ok": True, "jobId": job_id}




class SceneApplyDetailsJobIn(BaseModel):
    kind: str
    baseUrl: Optional[str] = None
    detailUrls: List[str] = []
    prompt: str = ""
    format: Optional[str] = "9:16"

@router.post("/scene/applyDetailsJob")
def scene_apply_details_job(req: Request, body: SceneApplyDetailsJobIn):
    uid = _current_user_id(req)

    kind = (body.kind or "").strip().lower()
    if kind not in ("model", "location"):
        raise HTTPException(status_code=400, detail="kind must be 'model' or 'location'")

    job_id = _scene_job_create(uid, kind, "applyDetails")

    def _run():
        try:
            _scene_job_update(job_id, state="running", progress=5)
            spent = False
            # Spend 1 credit ONCE per job (idempotent via ref=job_id)
            try:
                add_ledger(uid, -1, ("SCENE_APPLY_MODEL_DETAILS" if kind=="model" else "SCENE_APPLY_LOCATION_DETAILS"), ref=job_id)
                spent = True
            except ValueError as e:
                _scene_job_update(job_id, state="error", error=str(e), progress=100)
                return


            # Load current scene to pick baseUrl if not provided
            with db() as con:
                row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
                sc = json.loads(row["data"]) if row and row["data"] else _default_scene()

            base_url = body.baseUrl or (sc.get("modelUrl") if kind == "model" else sc.get("locationUrl"))
            if not base_url:
                raise Exception("No base image. Generate model/location first.")

            base_dataurl = _url_to_dataurl(base_url)

            detail_urls = [u for u in (body.detailUrls or []) if isinstance(u, str) and u.strip()]
            if not detail_urls:
                raise Exception("No detailUrls provided")

            _scene_job_update(job_id, progress=25)

            detail_dataurls = []
            for u in detail_urls:
                detail_dataurls.append(_url_to_dataurl(u))

            prompt = (body.prompt or "").strip()
            if kind == "model":
                base_prompt = "Apply the provided detail reference images to the SAME person and outfit. Keep identity, face, body, pose, and background unchanged. Improve realism and match details only."
            else:
                base_prompt = "Apply the provided detail reference images to the SAME location scene. Keep camera, composition, lighting and all other elements unchanged. Match details only."
            full_prompt = (base_prompt + " " + WORLD_LIGHTING_POLICY + " " + prompt).strip()
            fmt = _normalize_format(body.format)
            full_prompt = (full_prompt + " " + _format_hint(fmt, kind)).strip()

            _scene_job_update(job_id, progress=55)
            out_dataurl = create_asset(kind=kind, prompt=full_prompt, base_image=base_dataurl, details=detail_dataurls)
            _scene_job_update(job_id, progress=80)
            asset_url = _save_dataurl_to_asset_url(out_dataurl)

            # persist: update base
            with db() as con:
                row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
                data = json.loads(row["data"]) if row and row["data"] else _default_scene()
                if kind == "model":
                    data["modelUrl"] = asset_url
                else:
                    data["locationUrl"] = asset_url
                ts = _now_iso()
                if row:
                    con.execute("UPDATE scenes SET data=?, updated_at=? WHERE user_id=?", (json.dumps(data, ensure_ascii=False), ts, uid))
                else:
                    con.execute("INSERT INTO scenes(user_id, data, updated_at) VALUES(?,?,?)", (uid, json.dumps(data, ensure_ascii=False), ts))

            _scene_job_update(job_id, state="done", progress=100, result_json=json.dumps({"url": asset_url}, ensure_ascii=False))
        except Exception as e:
            try:
                if spent:
                    add_ledger(uid, +1, "REFUND", ref=f"SCENE:{job_id}")
            except Exception:
                pass
            _scene_job_update(job_id, state="error", error=str(e), progress=100)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"ok": True, "jobId": job_id}
