from app.engine.engine_init import load_engine_config
import os
import re
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.router import api_router
from app.db.sqlite import init_db
from app.core.static_paths import STATIC_DIR, ASSETS_DIR, ensure_static_dirs

app = FastAPI(title="PhotoStudio Core API", version="0.2.0")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    # Превращаем в 400, чтобы фронт видел русское сообщение
    return JSONResponse(status_code=400, content={"ok": False, "message": str(exc)})

app.add_middleware(
    CORSMiddleware,
    # Dev: accept any localhost/127.0.0.1 port, keep cookies (allow_credentials)
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Чтобы фронт мог прочитать имя файла при скачивании (Content-Disposition)
    expose_headers=["Content-Disposition"],
)

# Extra CORS headers for StaticFiles / 304 responses (some browsers are picky)
@app.middleware("http")
async def _force_cors_headers(request: Request, call_next):
    resp = await call_next(request)
    # NOTE:
    # - <img> requests (and especially canvas usage) can be picky about CORS for /static/*
    # - some browsers omit Origin on certain asset requests; in that case we still want assets to load
    path = request.url.path or ""
    origin = request.headers.get("origin")

    is_dev_origin = bool(origin and re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", origin))
    is_static = path.startswith("/static/")

    if is_static:
        # For static assets we can be permissive in dev.
        # If Origin is present and is localhost/127.* -> reflect it.
        # If Origin is missing -> set '*'.
        if is_dev_origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers.setdefault("Vary", "Origin")
        else:
            resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    else:
        if is_dev_origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers.setdefault("Vary", "Origin")
    return resp

class CORSStaticFiles(StaticFiles):
    """StaticFiles that always adds CORS headers for dev so the frontend can load images safely."""
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        try:
            headers = dict((k.decode() if isinstance(k, (bytes, bytearray)) else k,
                            v.decode() if isinstance(v, (bytes, bytearray)) else v)
                           for k, v in resp.headers.raw)
        except Exception:
            headers = {}

        origin = None
        for (k, v) in scope.get("headers", []):
            if k == b"origin":
                origin = v.decode("utf-8", "ignore")
                break

        is_dev_origin = bool(origin and re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", origin))
        # For static assets in dev we can reflect localhost origins (works with fetch/canvas).
        if is_dev_origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers.setdefault("Vary", "Origin")
        else:
            # No Origin header (e.g. direct navigation) or non-dev origin: be permissive for static.
            resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp

@app.on_event("startup")
def _startup():
    ensure_static_dirs()
    print("STATIC_DIR =", str(STATIC_DIR))
    print("ASSETS_DIR =", str(ASSETS_DIR))
    init_db()


@app.get("/engine/status")
def engine_status():
    from datetime import datetime, timezone

    cfg = load_engine_config()

    # Read both sources: cfg (engine_init) and ENV, because some keys may be wired only one way.
    gemini_env = (os.getenv("GEMINI_API_KEY") or "").strip()
    veo_env = (os.getenv("VEO_API_KEY") or "").strip()
    kling_env = (os.getenv("KLING_API_KEY") or "").strip()

    gemini_cfg = (getattr(cfg, "api_key", "") or "").strip()
    veo_cfg = (getattr(cfg, "veo_api_key", "") or "").strip()
    kling_cfg = (getattr(cfg, "kling_api_key", "") or "").strip()

    # Models (may live in cfg or ENV)
    gemini_image_model_cfg = getattr(cfg, "image_model", None)
    gemini_vision_model_cfg = getattr(cfg, "vision_model", None)
    gemini_text_model_cfg = getattr(cfg, "text_model", None)

    gemini_image_model_env = (os.getenv("GEMINI_IMAGE_MODEL") or "").strip() or None
    gemini_vision_model_env = (os.getenv("GEMINI_VISION_MODEL") or "").strip() or None
    gemini_text_model_env = (os.getenv("GEMINI_TEXT_MODEL") or "").strip() or None

    # Veo defaults (optional)
    veo_model = getattr(cfg, "veo_model", None)
    veo_aspect_default = getattr(cfg, "veo_aspect_ratio", None)
    veo_duration_default = getattr(cfg, "veo_duration_sec", None)

    return {
        "ok": True,
        "engine": "stub",

        # configured flags (effective + breakdown)
        "gemini_configured": bool(gemini_env or gemini_cfg),
        "gemini_configured_env": bool(gemini_env),
        "gemini_configured_cfg": bool(gemini_cfg),

        "veo_configured": bool(veo_env or veo_cfg),
        "veo_configured_env": bool(veo_env),
        "veo_configured_cfg": bool(veo_cfg),

        "kling_configured": bool(kling_env or kling_cfg),
        "kling_configured_env": bool(kling_env),
        "kling_configured_cfg": bool(kling_cfg),

        # models (cfg/env + effective)
        "gemini_image_model": gemini_image_model_cfg or gemini_image_model_env,
        "gemini_vision_model": gemini_vision_model_cfg or gemini_vision_model_env,
        "gemini_text_model": gemini_text_model_cfg or gemini_text_model_env,

        # veo details
        "veo_model": veo_model,
        "veo_aspect_default": veo_aspect_default,
        "veo_duration_default": veo_duration_default,

        "time": datetime.now(timezone.utc).isoformat(),
    }
app.include_router(api_router, prefix="/api")
app.mount("/static", CORSStaticFiles(directory=str(STATIC_DIR)), name="static")
