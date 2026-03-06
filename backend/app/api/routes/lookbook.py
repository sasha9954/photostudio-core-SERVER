from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
import json
import io
import os
import mimetypes
import zipfile
from urllib.parse import urlparse
import os
import re
import base64
import hashlib
import threading
import uuid

from app.core.config import settings
from app.core.static_paths import ASSETS_DIR, ensure_static_dirs, asset_url
from app.services.auth_service import add_ledger
from app.engine.engine_init import load_engine_config
from app.engine.lookbook_engine import photoshoot as engine_photoshoot

from app.core.tokens import verify_token
from app.db.sqlite import db

COOKIE_NAME = "ps_token"
router = APIRouter(prefix="/lookbook")

ALLOWED_MODES = {"TORSO", "LEGS", "FULL"}
TTL_HOURS = 24


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _uid(req: Request) -> str:
    tok = req.cookies.get(COOKIE_NAME)
    if not tok:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # verify_token возвращает либо None, либо tuple(user_id, exp)
    v = verify_token(tok)
    uid = v[0] if v else None
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid session")
    return uid


def _cleanup(con, user_id: str):
    """
    Удаляем старые сессии (TTL) для пользователя, чтобы не раздувать БД.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS)
    rows = con.execute(
        "SELECT mode, updated_at FROM lookbook_sessions WHERE user_id=?",
        (user_id,),
    ).fetchall()
    for mode, updated_at in rows:
        dt = _parse_dt(updated_at)
        if dt < cutoff:
            con.execute(
                "DELETE FROM lookbook_sessions WHERE user_id=? AND mode=?",
                (user_id, mode),
            )


def _acquire_run_lock(user_id: str, mode: str, ttl_seconds: int = 180) -> bool:
    """
    Защита от бесконечных/параллельных запросов:
    - Если фотосессия для (user_id, mode) уже 'running' и не протухла по TTL — не запускаем вторую.
    - Храним флаг прямо в JSON lookbook_sessions.data, чтобы не городить отдельные таблицы.
    """
    now = datetime.now(timezone.utc)
    with db() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT data, updated_at FROM lookbook_sessions WHERE user_id=? AND mode=?",
            (user_id, mode),
        ).fetchone()

        if row and row["data"]:
            data = json.loads(row["data"])
        else:
            data = _default_session(mode)
            con.execute(
                "INSERT OR REPLACE INTO lookbook_sessions(user_id, mode, data, updated_at) VALUES(?,?,?,?)",
                (user_id, mode, json.dumps(data, ensure_ascii=False), _now_iso()),
            )

        run = (data or {}).get("_run") or {}
        if bool(run.get("running")):
            started_at = run.get("startedAt") or run.get("started_at") or ""
            try:
                started_dt = _parse_dt(started_at)
            except Exception:
                started_dt = now
            age = (now - started_dt).total_seconds()
            if age < max(10, int(ttl_seconds or 180)):
                return False  # already running

        # keep jobId if caller already set it (e.g., job resume)
        prev_job_id = (run or {}).get("jobId")
        data["_run"] = {"running": True, "startedAt": now.isoformat(), "jobId": prev_job_id}
        con.execute(
            "UPDATE lookbook_sessions SET data=?, updated_at=? WHERE user_id=? AND mode=?",
            (json.dumps(data, ensure_ascii=False), _now_iso(), user_id, mode),
        )
    return True


def _release_run_lock(user_id: str, mode: str):
    try:
        with db() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT data FROM lookbook_sessions WHERE user_id=? AND mode=?",
                (user_id, mode),
            ).fetchone()
            if not row or not row["data"]:
                return
            data = json.loads(row["data"])
            if not isinstance(data, dict):
                return
            run = (data or {}).get("_run") or {}
            run["running"] = False
            run["finishedAt"] = datetime.now(timezone.utc).isoformat()
            data["_run"] = run
            con.execute(
                "UPDATE lookbook_sessions SET data=?, updated_at=? WHERE user_id=? AND mode=?",
                (json.dumps(data, ensure_ascii=False), _now_iso(), user_id, mode),
            )
    except Exception:
        return


# -----------------------
# Jobs (server-side) — so generation continues after leaving page
# -----------------------

def _job_now_iso() -> str:
    return _now_iso()


def _job_create(uid: str, mode: str) -> str:
    job_id = f"lb_{uuid.uuid4().hex[:16]}"
    now = _job_now_iso()
    with db() as con:
        con.execute(
            "INSERT INTO lookbook_jobs(job_id, user_id, mode, state, progress, result_json, error, spent, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job_id, uid, mode, "queued", 0, None, None, 0, now, now),
        )
    return job_id


def _job_update(job_id: str, **fields):
    if not job_id:
        return
    allowed = {"state", "progress", "result_json", "error", "spent"}
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
        con.execute(f"UPDATE lookbook_jobs SET {', '.join(sets)} WHERE job_id=?", tuple(vals))


def _job_get(uid: str, job_id: str) -> dict | None:
    with db() as con:
        row = con.execute(
            "SELECT job_id, user_id, mode, state, progress, result_json, error, spent, created_at, updated_at FROM lookbook_jobs WHERE job_id=?",
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


def _session_set_job(uid: str, mode: str, job_id: str | None, running: bool):
    """Persist jobId into lookbook_sessions.data._run so UI can recover even without localStorage."""
    with db() as con:
        row = con.execute(
            "SELECT data FROM lookbook_sessions WHERE user_id=? AND mode=?",
            (uid, mode),
        ).fetchone()
        data = json.loads(row["data"]) if row and row["data"] else _default_session(mode)
        run = (data or {}).get("_run") or {}
        run["running"] = bool(running)
        run["jobId"] = job_id
        if running:
            run["startedAt"] = datetime.now(timezone.utc).isoformat()
        else:
            run["finishedAt"] = datetime.now(timezone.utc).isoformat()
        data["_run"] = run
        con.execute(
            "INSERT OR REPLACE INTO lookbook_sessions(user_id, mode, data, created_at, updated_at) VALUES(?,?,?,?,?)",
            (uid, mode, json.dumps(data, ensure_ascii=False), data.get("createdAt") or _now_iso(), _now_iso()),
        )


def _default_session(mode: str) -> dict:
    # 8 карточек: 1..7 вещи/детали, 8 логотип
    cards = []
    # Канонические подписи под макет (можно менять позже в UI)
    labels = [
        "ПЕРЕД", "ПРАВЫЙ БОК", "ЛЕВЫЙ БОК",
        "СПИНА", "ТКАНЬ / МАТЕРИАЛ", "ДЕТАЛИРОВКА 1",
        "ДЕТАЛИРОВКА 2", "ЛОГОТИП"
    ]
    for i in range(8):
        if i == 7:
            cards.append({
                "slot": 8,
                "type": "logo",
                "label": labels[i],
                "refUrl": None,
                "logoKind": "print",  # print | embroidery | patch
            })
        else:
            cards.append({
                "slot": i + 1,
                "type": "shot",
                "label": labels[i],
                "refUrl": None,
                "camera": "front",
                "pose": "classic",
            })
    return {
        "mode": mode,
        "format": "1:1",  # 1:1 | 16:9 | 9:16
        "cards": cards,
        "results": [],  # compact ordered list of urls
        "updatedAt": _now_iso(),
        "createdAt": _now_iso(),
    }



def _ensure_assets_dir():
    ensure_static_dirs()

def _guess_ext(mime: str) -> str:
    m = (mime or "").lower()
    if m == "image/png": return ".png"
    if m in ("image/jpeg", "image/jpg"): return ".jpg"
    if m == "image/webp": return ".webp"
    return ".png"

_DATAURL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)

def _save_dataurl_to_asset_url(data_url: str) -> str:
    m = _DATAURL_RE.match((data_url or "").strip())
    if not m:
        raise ValueError("Invalid dataUrl")
    mime = m.group("mime").strip()
    b64 = m.group("data").strip()
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

def _abs_asset_url(req: Request, url: str | None) -> str | None:
    if not url:
        return None
    u = str(url)
    if u.startswith("http://") or u.startswith("https://"):
        return u
    base = settings.PUBLIC_BASE_URL.rstrip("/")
    if u.startswith("/"):
        return base + u
    return base + "/" + u

def _load_current_scene(uid: str) -> dict:
    with db() as con:
        row = con.execute("SELECT data FROM scenes WHERE user_id = ?", (uid,)).fetchone()
        if not row or not row["data"]:
            return {}
        try:
            return json.loads(row["data"])
        except Exception:
            return {}

class SessionPatch(BaseModel):
    format: str | None = None
    cards: list | None = None
    results: list | None = None


@router.get("/session/{mode}")
def get_session(mode: str, req: Request):
    mode = (mode or "").upper()
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Bad mode")
    uid = _uid(req)
    with db() as con:
        _cleanup(con, uid)
        row = con.execute(
            "SELECT data FROM lookbook_sessions WHERE user_id=? AND mode=?",
            (uid, mode),
        ).fetchone()
        if not row:
            sess = _default_session(mode)
            con.execute(
                "INSERT OR REPLACE INTO lookbook_sessions(user_id, mode, data, created_at, updated_at) VALUES(?,?,?,?,?)",
                (uid, mode, json.dumps(sess, ensure_ascii=False), sess["createdAt"], sess["updatedAt"]),
            )
            return {"session": sess}
        try:
            sess = json.loads(row[0])
        except Exception:
            sess = _default_session(mode)
        # ensure minimal keys
        if sess.get("mode") != mode:
            sess["mode"] = mode
        return {"session": sess}


@router.patch("/session/{mode}")
def patch_session(mode: str, req: Request, body: SessionPatch):
    mode = (mode or "").upper()
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Bad mode")
    uid = _uid(req)
    with db() as con:
        _cleanup(con, uid)
        row = con.execute(
            "SELECT data, created_at FROM lookbook_sessions WHERE user_id=? AND mode=?",
            (uid, mode),
        ).fetchone()
        if row:
            try:
                sess = json.loads(row[0])
            except Exception:
                sess = _default_session(mode)
            created_at = row[1] or sess.get("createdAt") or _now_iso()
        else:
            sess = _default_session(mode)
            created_at = sess["createdAt"]

        # apply patch
        if body.format is not None:
            sess["format"] = body.format
        if body.cards is not None:
            sess["cards"] = body.cards
        if body.results is not None:
            sess["results"] = body.results

        sess["mode"] = mode
        sess["createdAt"] = created_at
        sess["updatedAt"] = _now_iso()

        con.execute(
            "INSERT OR REPLACE INTO lookbook_sessions(user_id, mode, data, created_at, updated_at) VALUES(?,?,?,?,?)",
            (uid, mode, json.dumps(sess, ensure_ascii=False), created_at, sess["updatedAt"]),
        )
        return {"session": sess}




def _asset_file_path_from_url(url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None
    # absolute -> take path
    try:
        parsed = urlparse(s)
        path = parsed.path if parsed.scheme in ("http", "https") else s
    except Exception:
        path = s
    if not path.startswith("/static/assets/"):
        return None
    fname = path.split("/static/assets/", 1)[1]
    # prevent traversal
    fname = os.path.basename(fname)
    if not fname:
        return None
    assets_dir = os.path.normpath(str(ASSETS_DIR))
    full = os.path.normpath(os.path.join(assets_dir, fname))
    if not full.startswith(assets_dir):
        return None
    if not os.path.exists(full):
        return None
    return full


@router.get("/download/{mode}")
def download_results(mode: str, req: Request):
    """Download current session results: 1 image => PNG/JPG; 2+ => ZIP."""
    mode = (mode or "").upper()
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Bad mode")
    uid = _uid(req)

    with db() as con:
        _cleanup(con, uid)
        row = con.execute(
            "SELECT data FROM lookbook_sessions WHERE user_id=? AND mode=?",
            (uid, mode),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No session")

    try:
        sess = json.loads(row[0])
    except Exception:
        raise HTTPException(status_code=500, detail="Bad session data")

    results = sess.get("results") or []
    # Session may store results either as a list of URL strings or objects like
    # {"url": "/static/assets/...", "slotIndex": 1, ...}.
    urls: list[str] = []
    for r in results:
        if isinstance(r, str) and r.strip():
            urls.append(r.strip())
            continue
        if isinstance(r, dict):
            u = r.get("url")
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())

    # de-dup while keeping order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    if not urls:
        raise HTTPException(status_code=404, detail="No results")

    file_paths = []
    for u in urls:
        fp = _asset_file_path_from_url(u)
        if fp:
            file_paths.append(fp)

    if not file_paths:
        raise HTTPException(status_code=404, detail="Assets missing")

    if len(file_paths) == 1:
        fp = file_paths[0]
        mt, _ = mimetypes.guess_type(fp)
        mt = mt or "application/octet-stream"
        ext = os.path.splitext(fp)[1] or ".png"
        filename = f"lookbook_{mode}{ext}"
        return FileResponse(fp, media_type=mt, filename=filename)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, fp in enumerate(file_paths, 1):
            ext = os.path.splitext(fp)[1] or ".png"
            arcname = f"{i:02d}{ext}"
            zf.write(fp, arcname=arcname)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="lookbook_{mode}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)



@router.post("/reset/{mode}")
def reset_session(mode: str, req: Request):
    mode = (mode or "").upper()
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Bad mode")
    uid = _uid(req)
    with db() as con:
        con.execute("DELETE FROM lookbook_sessions WHERE user_id=? AND mode=?", (uid, mode))
    return {"ok": True}

class PhotoshootIn(BaseModel):
    debug: bool = False


@router.get("/jobs/{job_id}")
def get_job(job_id: str, req: Request):
    uid = _uid(req)
    job = _job_get(uid, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job}

@router.post("/photoshoot/{mode}")
def run_photoshoot(req: Request, mode: str, body: PhotoshootIn):
    mode = (mode or "").upper()
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    uid = _uid(req)

    scene = _load_current_scene(uid) or {}
    model_url = _abs_asset_url(req, scene.get("modelUrl"))
    loc_url = _abs_asset_url(req, scene.get("locationUrl"))
    if not model_url or not loc_url:
        raise HTTPException(status_code=400, detail="Scene is incomplete: model and location are required")

    # load session
    session = get_session(mode, req)
    cards = session.get("session", {}).get("cards") if isinstance(session, dict) else None
    # get_session returns {mode, session, updated_at}; session is nested
    sess = session.get("session") if isinstance(session, dict) else {}
    cards = sess.get("cards") or []
    fmt = sess.get("format") or "9:16"

    shots = []
    for c in cards:
        ref = c.get("refUrl")
        if not ref:
            continue
        slot = int(c.get("slot") or 0)
        # В UI у карточек есть label/type, а title/kind могут отсутствовать.
        title = (c.get("title") or c.get("label") or "").lower()
        shot_type = "ITEM"
        if "логотип" in title or c.get("kind") == "logo" or c.get("type") == "logo":
            shot_type = "LOGO"
        elif "ткан" in title or "материал" in title or "детал" in title:
            shot_type = "DETAIL"
        shots.append({
            "id": f"slot_{slot}",
            "refImage": {"source": "url", "imgUrl": _abs_asset_url(req, ref)},
            "shotType": shot_type,
            "cameraAngle": c.get("camera") or "Фронт",
            "poseStyle": c.get("pose") or "Classic",
            "format": fmt,
            "slot": slot,
        })

    if not shots:
        raise HTTPException(status_code=400, detail="No cards with images to shoot")

    credits = len(shots)

    # Create job first (so frontend can resume even if it navigates away immediately)
    job_id = _job_create(uid, mode)

    # Acquire run lock — prevents multiple jobs per (uid, mode)
    lock_acquired = _acquire_run_lock(uid, mode)
    if not lock_acquired:
        # If already running, return 409 with a hint; frontend should continue polling existing job.
        _job_update(job_id, state="error", progress=0, error="already_running")
        raise HTTPException(status_code=409, detail="Фотосессия уже выполняется. Подожди завершения.")

    # Persist jobId into session._run so UI can recover without localStorage
    _session_set_job(uid, mode, job_id, running=True)

    def _runner():
        spent = 0
        try:
            _job_update(job_id, state="running", progress=5)

            # Spend credits once per job
            try:
                add_ledger(uid, -credits, "LOOKBOOK_PHOTOSHOOT", ref=f"{mode}:{credits}:{job_id}")
                spent = credits
                _job_update(job_id, spent=spent)
            except ValueError as e:
                _job_update(job_id, state="error", progress=0, error=str(e))
                return

            _job_update(job_id, progress=15)
            cfg = load_engine_config()
            prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "engine", "prompts")
            prompts_dir = os.path.abspath(prompts_dir)
            payload_scene = {
                "model": {"source": "url", "imgUrl": model_url},
                "location": {"source": "url", "imgUrl": loc_url},
            }
            eng = engine_photoshoot(cfg, prompts_dir, mode, payload_scene, shots, debug=bool(body.debug))
            if not eng.get("ok"):
                raise ValueError(eng.get("message") or eng.get("code") or "Engine error")

            _job_update(job_id, progress=80)

            out_results = []
            for r in eng.get("results") or []:
                data_url = r.get("image")
                if not data_url:
                    continue
                url = _save_dataurl_to_asset_url(data_url)
                slot = None
                rid = r.get("id") or ""
                m = re.match(r"slot_(\d+)", rid)
                if m:
                    slot = int(m.group(1))
                out_results.append({"slotIndex": slot, "url": url})

            # persist session results
            with db() as con:
                row = con.execute(
                    "SELECT data FROM lookbook_sessions WHERE user_id=? AND mode=?",
                    (uid, mode),
                ).fetchone()
                data = json.loads(row["data"]) if row and row["data"] else _default_session(mode)
                data["results"] = out_results
                # keep run info with jobId
                run = (data or {}).get("_run") or {}
                run["running"] = False
                run["jobId"] = job_id
                run["finishedAt"] = datetime.now(timezone.utc).isoformat()
                data["_run"] = run
                con.execute(
                    "UPDATE lookbook_sessions SET data=?, updated_at=? WHERE user_id=? AND mode=?",
                    (json.dumps(data, ensure_ascii=False), _now_iso(), uid, mode),
                )

            _job_update(job_id, state="done", progress=100, result_json=json.dumps({"results": out_results, "spent": spent}, ensure_ascii=False))
        except Exception as e:
            # refund spent credits
            if spent:
                try:
                    add_ledger(uid, spent, "REFUND", ref=f"LOOKBOOK:{mode}:{job_id}")
                except Exception:
                    pass
            _job_update(job_id, state="error", progress=0, error=str(e))
        finally:
            _release_run_lock(uid, mode)
            _session_set_job(uid, mode, job_id, running=False)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return {"ok": True, "mode": mode, "jobId": job_id, "state": "queued", "spent": credits}