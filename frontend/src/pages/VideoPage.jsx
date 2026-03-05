import React from "react";
import { fetchJson, API_BASE } from "../services/api.js";
import { useAuth } from "../app/AuthContext.jsx";
import "./VideoPage.css";
import { useLocation } from "react-router-dom";

function GlassSelect({ value, options, onChange, ariaLabel }){
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef(null);
  const btnRef = React.useRef(null);

  React.useEffect(() => {
    if(!open) return;
    const onDocDown = (e) => {
      const el = rootRef.current;
      if(!el) return;
      if(el.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => {
      if(e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = (opt) => {
    onChange?.(opt);
    setOpen(false);
    btnRef.current?.focus?.();
  };

  return (
    <div ref={rootRef} className={`glassSelect ${open ? "open" : ""}`} aria-label={ariaLabel || "Выбор"}>
      <button
        ref={btnRef}
        type="button"
        className="glassSelectBtn"
        aria-haspopup="listbox"
        aria-expanded={open ? "true" : "false"}
        onClick={() => setOpen(v => !v)}
      >
        <span className="glassSelectValue">{value}</span>
        <span className="glassSelectChevron" aria-hidden="true" />
      </button>

      {open && (
        <>
          <div className="glassSelectBackdrop" aria-hidden="true" onClick={() => setOpen(false)} />
          <div className="glassSelectMenu" role="listbox">
            {options.map((opt) => {
              const active = opt === value;
              return (
                <button
                  key={opt}
                  type="button"
                  role="option"
                  aria-selected={active ? "true" : "false"}
                  className={`glassSelectItem ${active ? "active" : ""}`}
                  onClick={() => pick(opt)}
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function getAccountKey(user){
  const id = (user && user.id) ? String(user.id) : "";
  const email = (user && user.email) ? String(user.email).toLowerCase() : "";
  return (id || email || "guest").trim();
}

function safeParse(json, fallback){
  try{ return JSON.parse(json); }catch{ return fallback; }
}

// Debug: enable with ?debug=1 or localStorage "ps_debug_video" = "1"
function isVideoDebugEnabled(){
  try{
    const sp = new URLSearchParams(window.location.search || "");
    if(sp.get("debug") === "1") return true;
    return String(localStorage.getItem("ps_debug_video") || "") === "1";
  }catch{ return false; }
}
function vlog(...args){
  if(!isVideoDebugEnabled()) return;
  try{ console.log("[VIDEO]", ...args); }catch{}
}

// Accept legacy values: string or array length 1..9; always return array length 9
function normalizeSlots(input){
  if(!input) return Array(9).fill("");
  if(typeof input === "string"){
    const arr = Array(9).fill("");
    arr[0] = input;
    return arr;
  }
  if(Array.isArray(input)){
    const arr = Array(9).fill("");
    for(let i=0;i<Math.min(9, input.length); i++){
      arr[i] = input[i] || "";
    }
    return arr;
  }
  return Array(9).fill("");
}

function sanitizePersistentUrl(url){
  let s = String(url || "").trim();
  if(!s) return "";

  // never persist blob/data
  if(s.startsWith("blob:")) return "";
  if(s.startsWith("data:")) return "";

  // normalize windows backslashes if any
  if (s.includes("\\")) s = s.replace(/\\/g, "/");

  // keep absolute http(s)
  if(/^https?:\/\//i.test(s)) return s;

  // If backend returned bare filename for a video, normalize to /static/videos/<name>
  // (Observed in some builds where upload returns "clip_...." without prefix)
  if(!s.includes("/") && /^clip_/i.test(s)){
    return `${API_BASE}/static/videos/${s}`;
  }

  // normalize backend-served paths (assets + videos) to API_BASE
  if(s.startsWith("/static/")) return `${API_BASE}${s}`;
  if(s.startsWith("static/")) return `${API_BASE}/${s}`;

  // sometimes we may get "app/static/videos/..." or "videos/..."
  if(s.startsWith("videos/")) return `${API_BASE}/static/${s}`;
  if(s.startsWith("/videos/")) return `${API_BASE}/static${s}`;

  // normalize API paths
  if(s.startsWith("/api/")) return `${API_BASE}${s}`;

  // unknown relative path: keep as-is (may be already correct)
  return s;
}

function resolveAssetUrl(url){
  return sanitizePersistentUrl(url);
}

// Backend merge endpoint expects STRICT relative paths: /static/videos/...
// Our persistent urls may be absolute (http://127.0.0.1:8000/static/videos/...) or other variants.
function toStaticVideosPath(url){
  let s = String(url || "").trim();
  if(!s) return "";

  // normalize windows backslashes if any
  if(s.includes("\\") ) s = s.replace(/\\/g, "/");

  // If absolute URL, extract pathname
  try{
    if(/^https?:\/\//i.test(s)){
      const u = new URL(s);
      s = u.pathname || "";
    }
  }catch{
    // ignore
  }

  // Ensure leading slash
  if(s && !s.startsWith("/")) s = "/" + s;

  // If it contains /static/videos/ anywhere, slice from there
  const i = s.indexOf("/static/videos/");
  if(i >= 0) s = s.slice(i);

  // Accept only /static/videos/
  if(!s.startsWith("/static/videos/")) return "";
  return s;
}



function readModeFromSearch(search){
  const sp = new URLSearchParams(search || "");
  const raw = (sp.get("mode") || sp.get("variant") || sp.get("v") || "").trim().toUpperCase();
  if(raw === "TORSO" || raw === "LEGS" || raw === "FULL") return raw;
  return "";
}

async function fileToDataUrl(file){
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

const ENGINE_LIST = [
  { key: "STANDARD", label: "STANDARD", sub: "2.6", locked: false },
  { key: "CINEMA", label: "CINEMA", sub: "3.0", locked: true, disabled: true, disabledHint: "Нет API для Kling 3.0" },
  { key: "ULTRA", label: "ULTRA", sub: "3.1", locked: true },
  { key: "STUDIO", label: "STUDIO", sub: "2.0", locked: true },
];

const CAMERA_LIST = [
  // Product-safe camera moves (no orbit/crane/panorama that can cause hallucinated redraws)
  "Статично",
  "Микро-движение",
  "Наезд",
  "Отъезд",
  "Сдвиг ←",
  "Сдвиг →",
  "Сдвиг ↑",
  "Сдвиг ↓",
  "Наклон ↑",
  "Наклон ↓",
];

export default function VideoPage(){
  const { user } = useAuth();
  const accountKey = React.useMemo(() => getAccountKey(user), [user]);

  const location = useLocation();
  const KEY_LAST_MODE = React.useMemo(() => `ps_video_last_mode_v1:${accountKey}`, [accountKey]);
  const urlMode = React.useMemo(() => readModeFromSearch(location.search), [location.search]);
  const mode = React.useMemo(() => {
    let stored = "";
    try { stored = String(localStorage.getItem(KEY_LAST_MODE) || "").toUpperCase(); } catch {}
    if(urlMode) return urlMode;
    if(stored === "TORSO" || stored === "LEGS" || stored === "FULL") return stored;
    return "FULL";
  }, [urlMode, KEY_LAST_MODE]);

  React.useEffect(() => {
    if(!urlMode) return;
    try { localStorage.setItem(KEY_LAST_MODE, urlMode); } catch {}
  }, [urlMode, KEY_LAST_MODE]);

  const KEY = React.useMemo(() => ({
    photo: `ps_video_photoSlots_v1:${accountKey}`,
    clips: `ps_video_clipSlots_v1:${accountKey}`,
    state: `ps_video_state_v1:${accountKey}`,
    unlock: `ps_video_unlock_v1:${accountKey}`,
    meta: `ps_video_clipMeta_v1:${accountKey}`,
    merged: `ps_video_mergedResult_v1:${accountKey}`,
    activeJob: `ps_video_activeJob_v1:${accountKey}`,
  }), [accountKey]);

  // Unlocks must be session-scoped (reset on tab close) AND reset on account logout/switch.
  const prevAccountKeyRef = React.useRef(null);
  React.useEffect(() => {
    const prev = prevAccountKeyRef.current;
    if(prev && prev !== accountKey){
      try { sessionStorage.removeItem(`ps_video_unlock_v1:${prev}`); } catch {}
    }
    prevAccountKeyRef.current = accountKey;
    // Always require re-unlock after account switch/logout.
    setUnlocks({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountKey]);

  
  // When account changes, force re-hydration BEFORE any persist writes,
  // otherwise the initial "guest" render may wipe the real user's storage.
  React.useEffect(() => {
    didHydrateRef.current = false;
    hydratedForAccountRef.current = null;
    // Optional: clear UI quickly (prevents showing previous account for a frame)
    setPhotoSlots(Array(9).fill(""));
    setActivePhotoIdx(0);
    setClipSlots(Array(9).fill(""));
    vlog("accountKey changed", { accountKey });
    didHydrateRef.current = false;
    hydratedForAccountRef.current = null;
    // Clear UI quickly to avoid showing previous account's frames for a moment.
    // Persist effects are guarded by didHydrateRef, so this should NOT wipe storage.
    setPhotoSlots(Array(9).fill(""));
    setActivePhotoIdx(0);
    setClipSlots(Array(9).fill(""));
    setClipMeta(Array(9).fill(""));
    setMergedResultUrl("");
    setActiveClipIdx(0);
    setStatus("");
  }, [accountKey]);
const [photoSlots, setPhotoSlots] = React.useState(() => Array(9).fill(""));
  const [clipSlots, setClipSlots] = React.useState(() => Array(9).fill(""));
  const [clipMeta, setClipMeta] = React.useState(() => Array(9).fill(""));
  const [mergedResultUrl, setMergedResultUrl] = React.useState("");

  // Server-backed job (so generation continues after leaving / refresh)
  const [videoJobId, setVideoJobId] = React.useState("");
  const [videoJobState, setVideoJobState] = React.useState(""); // queued|running|done|error
  const [isStarting, setIsStarting] = React.useState(false);
  const [isMerging, setIsMerging] = React.useState(false);
  const isGenerating = isStarting || videoJobState === "running" || videoJobState === "queued";
  const [videoJobProgress, setVideoJobProgress] = React.useState(0);
  const [overlayText, setOverlayText] = React.useState("");
  const overlayTimerRef = React.useRef(null);
  const pollTimerRef = React.useRef(null);
  const [lastAddedClipIdx, setLastAddedClipIdx] = React.useState(null);
  const [activePhotoIdx, setActivePhotoIdx] = React.useState(0);
  const [ultraRefs, setUltraRefs] = React.useState([]); // выбранные рефы для ULTRA(Veo): индексы 0..8
const [photoZoomOpen, setPhotoZoomOpen] = React.useState(false);
const [videoZoomOpen, setVideoZoomOpen] = React.useState(false);
React.useEffect(() => {
  if(!photoZoomOpen && !videoZoomOpen) return;
  const onKey = (e) => {
    if(e.key === "Escape"){
      setPhotoZoomOpen(false);
      setVideoZoomOpen(false);
    }
  };
  document.addEventListener("keydown", onKey);
  return () => document.removeEventListener("keydown", onKey);
}, [photoZoomOpen, videoZoomOpen]);
const zoomSrc = sanitizePersistentUrl(photoSlots[activePhotoIdx]) ? resolveAssetUrl(photoSlots[activePhotoIdx]) : "";
  const [activeClipIdx, setActiveClipIdx] = React.useState(0);

  const [format, setFormat] = React.useState("9:16");
  const [duration, setDuration] = React.useState(5);
  const [camera, setCamera] = React.useState(CAMERA_LIST[0]);
  const [light, setLight] = React.useState("Мягкий");
  const [engine, setEngine] = React.useState("STANDARD");
  const [enginePulse, setEnginePulse] = React.useState(false);
  const [status, setStatus] = React.useState("Здесь будут действия и ошибки (почему генерация не запустилась и т.д.).");
  const [statusLog, setStatusLog] = React.useState([]); // {text, level, t}
  const statusLogRef = React.useRef("");

  React.useEffect(() => {
    // Visual pulse when switching engine modes
    setEnginePulse(true);
    const t = setTimeout(() => setEnginePulse(false), 520);
    return () => clearTimeout(t);
  }, [engine]);


  // ULTRA(Veo): если выбрано 2–3 рефа — формат фиксируется 16:9 (API ограничение)
  React.useEffect(() => {
    if (engine !== "ULTRA") return;
    const n = Array.isArray(ultraRefs) ? ultraRefs.length : 0;
    if (n > 1 && format !== "16:9") setFormat("16:9");
  }, [engine, ultraRefs, format]);



  React.useEffect(() => {
    // Append status updates into a scrollable log (no refactor of existing setStatus calls).
    const msg = (status ?? "").toString();
    if (!msg.trim()) return;
    if (msg === statusLogRef.current) return;
    statusLogRef.current = msg;

    const lower = msg.toLowerCase();
    let level = "info";
    if (lower.includes("ошибка") || lower.includes("error") || lower.includes("failed") || lower.includes("invalid")) level = "error";
    else if (lower.includes("предуп") || lower.includes("warn") || lower.includes("warning")) level = "warn";
    else if (lower.includes("готово") || lower.includes("успеш") || lower.includes("done")) level = "ok";

    const item = { text: msg, level, t: Date.now() };

    setStatusLog((prev) => {
      const next = Array.isArray(prev) ? [...prev] : [];
      // Avoid repeating the same line twice in a row (now newest is at top)
      if (next.length && next[0]?.text === msg) return next;
      next.unshift(item); // newest first
      return next.length > 200 ? next.slice(0, 200) : next;
    });
  }, [status]);
  const formatHint = React.useMemo(() => {
    const fmt = format;
    const sec = duration;
    const eng = engine;
    const lines = [];
    if (fmt === "9:16") lines.push("9:16 — вертикально (TikTok / Reels / Shorts).");
    if (fmt === "1:1") lines.push("1:1 — квадрат (лента Instagram / маркетплейсы).");
    if (fmt === "16:9") lines.push("16:9 — горизонтально (YouTube / сайт).");
    lines.push("");
    lines.push(`Длительность: ${sec}s. Чем длиннее — тем больше движения и шанс артефактов.`);
    if (eng === "STANDARD") {
      lines.push("STANDARD: 1 кадр → 1 клип (быстрее, стабильнее).");
    } else {
      lines.push("CINEMA/ULTRA/STUDIO: можно до 3 кадров (плавнее, но требовательнее).");
    }
    lines.push("");
    lines.push("Подсказка: загрузи 2–3 клипа и нажми «Объединить» — итог будет в блоке «Результат» справа.");
    return lines.join("\n");
  }, [format, duration, engine]);



  const [unlocks, setUnlocks] = React.useState(() => ({ }));
  const [unlockOpen, setUnlockOpen] = React.useState(false);
  const [unlockTarget, setUnlockTarget] = React.useState(null);
  const [unlockCode, setUnlockCode] = React.useState("");

  const didHydrateRef = React.useRef(false);
  
  const hydratedForAccountRef = React.useRef(null);
const didAutoImportRef = React.useRef(false);
  const uploadingRef = React.useRef(false);
  const photoFileRef = React.useRef(null);
  const clipFileRef = React.useRef(null);
  const clipStripRef = React.useRef(null);
  const clipBtnRefs = React.useRef([]);
  const photoBtnRefs = React.useRef([]);


React.useEffect(() => {
  if(!photoZoomOpen) return;
  const onKey = (e) => { if(e.key === "Escape") setPhotoZoomOpen(false); };
  document.addEventListener("keydown", onKey);
  return () => document.removeEventListener("keydown", onKey);
}, [photoZoomOpen]);


React.useEffect(() => {
  if(lastAddedClipIdx === null || lastAddedClipIdx === undefined) return;
  const el = clipBtnRefs.current?.[lastAddedClipIdx];
  if(el && typeof el.scrollIntoView === "function"){
    try{
      el.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      if(typeof el.focus === "function") el.focus();
    }catch{}
  }
  const t = setTimeout(() => setLastAddedClipIdx(null), 1600);
  return () => clearTimeout(t);
}, [lastAddedClipIdx]);

  // hydrate
  React.useEffect(() => {
    const p = safeParse(localStorage.getItem(KEY.photo) || "", null);
    const c = safeParse(localStorage.getItem(KEY.clips) || "", null);
    const st = safeParse(localStorage.getItem(KEY.state) || "", null);
    const mt = safeParse(localStorage.getItem(KEY.meta) || "", null);
    const mg = String(localStorage.getItem(KEY.merged) || "");

    const p9 = normalizeSlots(p).map(resolveAssetUrl);
    const c9 = normalizeSlots(c).map(resolveAssetUrl);
    const m9 = (Array.isArray(mt) ? mt : Array(9).fill("")).slice(0, 9);
    while(m9.length < 9) m9.push("");

    // Legacy migration: older UI stored merged result inside clipSlots/meta.
    // Move it into dedicated mergedResultUrl and remove from the clip list.
    let merged = resolveAssetUrl(mg || "");
    if(!merged){
      for(let i = m9.length - 1; i >= 0; i--){
        if(m9[i] === "merged" && sanitizePersistentUrl(c9[i])){
          merged = resolveAssetUrl(c9[i]);
          c9[i] = "";
          m9[i] = "";
          break;
        }
      }
    }

    setPhotoSlots(p9);
    setClipSlots(c9);
    setClipMeta(m9);
    setMergedResultUrl(merged);

    vlog("hydrate", {
      accountKey,
      KEY,
      rawPhotoType: (p === null) ? "null" : Array.isArray(p) ? `array:${p.length}` : typeof p,
      rawClipType: (c === null) ? "null" : Array.isArray(c) ? `array:${c.length}` : typeof c,
      p9, c9, merged
    });

    if(st && typeof st === "object"){
      if(["9:16","1:1","16:9"].includes(st.format)) setFormat(st.format);
      if([5,10].includes(st.duration)) setDuration(st.duration);
      if(typeof st.camera === "string" && CAMERA_LIST.includes(st.camera)) setCamera(st.camera);
      if(typeof st.light === "string" && ["Мягкий","Контраст","Тёплый"].includes(st.light)) setLight(st.light);
      if(typeof st.engine === "string") setEngine(st.engine);
      if(Number.isFinite(st.activePhotoIdx)) setActivePhotoIdx(Math.min(8, Math.max(0, st.activePhotoIdx)));
      if(Number.isFinite(st.activeClipIdx)) setActiveClipIdx(Math.min(8, Math.max(0, st.activeClipIdx)));
    }
    const un = safeParse((() => { try { return sessionStorage.getItem(KEY.unlock) || ""; } catch { return ""; } })(), null);
    if(un && typeof un === "object"){
      setUnlocks(un);
    }
    didHydrateRef.current = true;
    hydratedForAccountRef.current = accountKey;
  }, [KEY.photo, KEY.clips, KEY.state, KEY.unlock, KEY.meta, KEY.merged, accountKey]);

  // auto-import from Lookbook on entry (if coming from Lookbook /video?mode=...)
  React.useEffect(() => {
    if(!didHydrateRef.current) return;
    if(didAutoImportRef.current) return;
    didAutoImportRef.current = true;

    // если уже есть фото — не трогаем (пользователь мог загрузить вручную)
    const hasAny = photoSlots.some((s) => !!sanitizePersistentUrl(s));
    if(hasAny) return;

    importFromLookbook();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, accountKey]);


  // -----------------------------
  // Video generation jobs (server)
  // -----------------------------

  const funnyPhrases = React.useMemo(() => ([
    "Снимаем дубль… только без дубля 😄",
    "Камера, мотор… ой, мотор тут нейросеть 🤖",
    "Пожалуйста, не моргай… хотя ты уже фото 👀",
    "Стабилизация: держим кадр как держим кофе ☕",
    "Подкручиваем магию пикселей…",
    "Договариваемся с физикой движения…",
    "Собираем клип по молекулам…",
    "Не уходи далеко — мы почти в кино 🎬",
    "Рендерим так, чтобы было «ВАУ», а не «ОЙ»",
    "Если видишь это — значит процесс живой 🙂",
    "Терпение… нейросети тоже думают",
    "Проверяем реализм: 3… 2… 1…",
  ]), []);

  const startOverlay = React.useCallback(() => {
    if(overlayTimerRef.current) clearInterval(overlayTimerRef.current);
    let idx = 0;
    setOverlayText(funnyPhrases[0] || "Генерация…");
    overlayTimerRef.current = setInterval(() => {
      idx = (idx + 1) % Math.max(1, funnyPhrases.length);
      setOverlayText(funnyPhrases[idx] || "Генерация…");
    }, 4500);
  }, [funnyPhrases]);

  const stopOverlay = React.useCallback(() => {
    if(overlayTimerRef.current){
      clearInterval(overlayTimerRef.current);
      overlayTimerRef.current = null;
    }
    setOverlayText("");
  }, []);

  const clearPoll = React.useCallback(() => {
    if(pollTimerRef.current){
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const saveActiveJob = React.useCallback((jobId) => {
    try{
      if(!jobId){
        localStorage.removeItem(KEY.activeJob);
        return;
      }
      localStorage.setItem(KEY.activeJob, JSON.stringify({ jobId, t: Date.now() }));
    }catch(_e){}
  }, [KEY.activeJob]);

  const pollJobOnce = React.useCallback(async (jobId) => {
    if(!jobId) return;
    try{
      const res = await fetchJson(`/api/video/jobs/${jobId}`, { method: "GET" });
      const job = res?.job;
      if(!job) return;

      setVideoJobState(job.state || "");
      setVideoJobProgress(Number(job.progress || 0));

      if(job.state === "done"){
        clearPoll();
        stopOverlay();
        saveActiveJob("");
        setVideoJobId("");

        const urlsRaw = job?.result?.videos || [];
        const urls = (Array.isArray(urlsRaw) ? urlsRaw : [urlsRaw])
          .map((u) => sanitizePersistentUrl(u))
          .filter(Boolean);

        if(urls.length){
          // Newest clip goes to slot #1, shifting older ones to the right.
          // If the same URL already exists, we move it to the front (no duplicates).
          const newest = urls[0];

          setClipSlots((prev) => {
            const prevSan = prev.map((x) => sanitizePersistentUrl(x) || "");
            const kept = prevSan.filter((x) => x && x !== newest);
            const next = [newest, ...kept].slice(0, 9);
            while(next.length < 9) next.push("");
            return next;
          });

          setClipMeta((prev) => {
            const next = ["generated", ...prev].slice(0, 9);
            while(next.length < 9) next.push("");
            return next;
          });

          setActiveClipIdx(0);
          setStatus(`Готово: ${urls.length} клип(а). Добавлено в слот #1. Можно объединять справа.`);
        }else{
          setStatus("Готово, но videos пустой/некорректный (job.result.videos). Проверь backend.");
        }
      }

      if(job.state === "error"){
        clearPoll();
        stopOverlay();
        saveActiveJob("");
        setVideoJobId("");
        const err = job.error || "Ошибка";
        setStatus(`Ошибка генерации: ${err}`);
      }
    }catch(_e){
      // keep polling
    }
  }, [activeClipIdx, clearPoll, saveActiveJob, stopOverlay]);

  const startPolling = React.useCallback((jobId) => {
    if(!jobId) return;
    clearPoll();
    pollJobOnce(jobId);
    pollTimerRef.current = setInterval(() => pollJobOnce(jobId), 1600);
  }, [clearPoll, pollJobOnce]);

  // Resume active job after refresh / navigation
  React.useEffect(() => {
    if(!didHydrateRef.current) return;
    try{
      const raw = localStorage.getItem(KEY.activeJob) || "";
      if(!raw) return;
      const j = JSON.parse(raw);
      const jobId = j?.jobId;
      if(!jobId) return;
      setVideoJobId(jobId);
      setVideoJobState("running");
      setStatus("Продолжаю генерацию (job)…");
      startOverlay();
      startPolling(jobId);
    }catch(_e){}
  }, [KEY.activeJob, startOverlay, startPolling]);

  React.useEffect(() => {
    return () => {
      try{ clearPoll(); }catch(_e){}
      try{ stopOverlay(); }catch(_e){}
    };
  }, [clearPoll, stopOverlay]);


  // persist
  React.useEffect(() => {
    if(!didHydrateRef.current) { vlog("persist:photo skipped (not hydrated)", { accountKey, key: KEY.photo }); return; }
    // Keep guard for photos because they can briefly be data: URLs during import.
    if(uploadingRef.current) { vlog("persist:photo skipped (uploading)", { accountKey }); return; }
    try{
      const payload = photoSlots.map(resolveAssetUrl);
      localStorage.setItem(KEY.photo, JSON.stringify(payload));
      vlog("persist:photo", { accountKey, key: KEY.photo, payload });
    }catch(e){
      vlog("persist:photo error", e);
    }
  }, [photoSlots, KEY.photo, accountKey]);

  React.useEffect(() => {
    if(!didHydrateRef.current) { vlog("persist:clips skipped (not hydrated)", { accountKey, key: KEY.clips }); return; }
    // NOTE: do NOT block clip persist during uploading. Otherwise video URLs never get written.
    try{
      const payload = clipSlots.map(resolveAssetUrl);

      // ANTI-WIPE: never overwrite a non-empty stored set with an all-empty payload
      const allEmpty = payload.every(v => !v);
      const prevRaw = localStorage.getItem(KEY.clips) || "";
      let prevHadAny = false;
      if(prevRaw){
        try{
          const prev = JSON.parse(prevRaw);
          if(Array.isArray(prev)) prevHadAny = prev.some(v => !!v);
        }catch(_e){}
      }
      if(allEmpty && prevHadAny){
        vlog("persist:clips skipped (anti-wipe)", { accountKey, key: KEY.clips });
        return;
      }

      localStorage.setItem(KEY.clips, JSON.stringify(payload));
      vlog("persist:clips", { accountKey, key: KEY.clips, payload });
    }catch(e){
      vlog("persist:clips error", e);
    }
  }, [clipSlots, KEY.clips, accountKey]);


  React.useEffect(() => {
    if(!didHydrateRef.current) { vlog("persist:meta skipped (not hydrated)", { accountKey, key: KEY.meta }); return; }
    try{
      const payload = (Array.isArray(clipMeta) ? clipMeta : Array(9).fill("")).slice(0,9);
      while(payload.length < 9) payload.push("");

      // ANTI-WIPE (meta): never overwrite a stored non-empty meta with all-empty payload
      // while we still have clips. This prevents losing the markers after reload/logout
      // in edge cases where UI state briefly resets.
      const allEmpty = payload.every(v => !v);
      const prevRaw = localStorage.getItem(KEY.meta) || "";
      let prevHadAny = false;
      if(prevRaw){
        try{
          const prev = JSON.parse(prevRaw);
          if(Array.isArray(prev)) prevHadAny = prev.some(v => !!v);
        }catch(_e){}
      }
      const hasAnyClips = (Array.isArray(clipSlots) ? clipSlots : []).some(s => !!sanitizePersistentUrl(s));
      if(allEmpty && prevHadAny && hasAnyClips){
        vlog("persist:meta skipped (anti-wipe)", { accountKey, key: KEY.meta });
        return;
      }

      localStorage.setItem(KEY.meta, JSON.stringify(payload));
      vlog("persist:meta", { accountKey, key: KEY.meta, payload });
    }catch(e){
      vlog("persist:meta error", e);
    }
  }, [clipMeta, KEY.meta, accountKey, clipSlots]);

  React.useEffect(() => {
    if(!didHydrateRef.current) { vlog("persist:merged skipped (not hydrated)", { accountKey, key: KEY.merged }); return; }
    try{
      const url = resolveAssetUrl(mergedResultUrl || "");
      // ANTI-WIPE: do not overwrite stored merged result with empty unless it was already empty.
      const prevRaw = localStorage.getItem(KEY.merged) || "";
      const prevUrl = (typeof prevRaw === "string" ? prevRaw : "");
      if(!url && prevUrl){
        vlog("persist:merged skipped (anti-wipe)", { accountKey, key: KEY.merged });
        return;
      }
      localStorage.setItem(KEY.merged, url || "");
      vlog("persist:merged", { accountKey, key: KEY.merged, url });
    }catch(e){
      vlog("persist:merged error", e);
    }
  }, [mergedResultUrl, KEY.merged, accountKey]);


  React.useEffect(() => {
    if(!didHydrateRef.current) return;
    if(uploadingRef.current) return;
    try{
      localStorage.setItem(KEY.state, JSON.stringify({
        format, duration, camera, light, engine, activePhotoIdx, activeClipIdx,
      }));
    }catch{}
  }, [KEY.state, format, duration, camera, light, engine, activePhotoIdx, activeClipIdx]);

  React.useEffect(() => {
    if(!didHydrateRef.current) return;
    try{ sessionStorage.setItem(KEY.unlock, JSON.stringify(unlocks || {})); }catch{}
  }, [KEY.unlock, unlocks]);

  const isEngineLocked = React.useMemo(() => {
    const def = ENGINE_LIST.find(e => e.key === engine);
    if(!def) return false;
    if(!def.locked) return false;
    return !unlocks?.[def.key];
  }, [engine, unlocks]);

  const cost = React.useMemo(() => {
    if(engine === "ULTRA") return 2; // Veo: фикс 2 кредита за генерацию (1–3 фото)
    if(engine === "STANDARD") return duration === 10 ? 2 : 1;
    return 0;
  }, [engine, duration]);

  async function importFromLookbook(){
    setStatus(`Импортируем кадры из Lookbook (${mode})…`);
    try{
      // SERVER SOURCE OF TRUTH: берём результаты из backend session
      const res = await fetch(`${API_BASE}/api/lookbook/session/${mode}`, {
        method: "GET",
        credentials: "include",
      });
      const data = await res.json().catch(() => null);
      if(!res.ok){
        const msg = data?.detail || data?.message || `HTTP ${res.status}`;
        throw new Error(msg);
      }

      const arr =
        (Array.isArray(data?.results) ? data.results : null) ||
        (Array.isArray(data?.session?.results) ? data.session.results : null) ||
        (Array.isArray(data?.data?.results) ? data.data.results : null) ||
        [];

      const urls = arr
        .map((r) => sanitizePersistentUrl(r?.url || r?.assetUrl || r?.src || r))
        .filter(Boolean)
        .slice(0, 9);

      if(!urls.length){
        setStatus("В Lookbook пока нет результатов для импорта.");
        return;
      }

      setPhotoSlots(prev => {
        const next = [...prev];
        let inserted = 0;
        for(let i=0; i<9 && inserted<urls.length; i++){
          if(!sanitizePersistentUrl(next[i])){
            next[i] = urls[inserted++];
          }
        }
        // если все слоты были заняты, то просто перезапишем первые N (ожидаемое поведение при автоимпорте)
        if(inserted === 0){
          urls.forEach((u, i) => { next[i] = u; });
          inserted = urls.length;
        }
        setStatus(`Импортировано кадров: ${inserted}`);
        return next;
      });
      setActivePhotoIdx(0);
    }catch(e){
      setStatus(`Импорт не удался: ${e?.message || e}`);
      console.warn("Lookbook import failed", e);
    }
  }

  async function onPickPhotoFile(file){
    if(!file) return;
    uploadingRef.current = true;
    setStatus("Загружаем фото…");
    try{
      const dataUrl = await fileToDataUrl(file);
      const out = await fetchJson("/api/assets/fromDataUrl", { method: "POST", body: { dataUrl } });
      const url = sanitizePersistentUrl(out?.url);
      if(!url) throw new Error("Не удалось сохранить фото");

      setPhotoSlots(prev => {
        const next = [...prev];
        next[activePhotoIdx] = url;
        return next;
      });
      setStatus("Фото добавлено.");
    }catch(e){
      setStatus(`Ошибка загрузки фото: ${e?.message || e}`);
    }finally{
      uploadingRef.current = false;
    }
  }

  async function onPickClipFile(file){
    if(!file) return;
    uploadingRef.current = true;
    setStatus("Загружаем клип…");
    try{
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/api/video/upload`, {
        method: "POST",
        credentials: "include",
        body: fd,
      });
      const data = await res.json().catch(() => null);
      if(!res.ok){
        const msg = data?.detail || data?.message || `HTTP ${res.status}`;
        throw new Error(msg);
      }
      const url = sanitizePersistentUrl(data?.url);
      if(!url) throw new Error("Не удалось сохранить клип");
      setClipSlots(prev => {
        const next = [...prev];
        next[activeClipIdx] = url;
        return next;
      });
      setClipMeta(prev => {
        const next = Array.isArray(prev) ? [...prev] : Array(9).fill("");
        next[activeClipIdx] = "uploaded";
        return next;
      });
      setLastAddedClipIdx(activeClipIdx);
      setStatus("Клип добавлен.");
    }catch(e){
      setStatus(`Ошибка загрузки клипа: ${e?.message || e}`);
    }finally{
      uploadingRef.current = false;
    }
  }

  async function saveActivePhoto(){
    const raw = sanitizePersistentUrl(photoSlots[activePhotoIdx]);
    if(!raw){
      setStatus("Нечего сохранять: выберите кадр с фото.");
      return;
    }
    setStatus("Сохраняем кадр…");
    try{
      const url = resolveAssetUrl(raw);
      const res = await fetch(url, { method: "GET", credentials: "include" });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);

      const ext = (() => {
        const mime = (blob.type || "").toLowerCase();
        if(mime.includes("png")) return "png";
        if(mime.includes("webp")) return "webp";
        if(mime.includes("jpeg") || mime.includes("jpg")) return "jpg";
        const m = url.match(/\.([a-z0-9]{2,5})(?:\?|#|$)/i);
        return (m ? m[1].toLowerCase() : "png");
      })();

      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `frame_${activePhotoIdx + 1}.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
      setStatus("Кадр сохранён.");
    }catch(e){
      setStatus(`Ошибка сохранения: ${e?.message || e}`);
    }
  }


  
  async function saveClipUrl(raw, filenameBase="clip"){
    const clean = sanitizePersistentUrl(raw);
    if(!clean){
      setStatus("Нечего сохранять: нет видео.");
      return;
    }
    setStatus("Сохраняем видео…");
    try{
      const url = resolveAssetUrl(clean);
      const res = await fetch(url, { method: "GET", credentials: "include" });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);

      const ext = (() => {
        const mime = (blob.type || "").toLowerCase();
        if(mime.includes("mp4")) return "mp4";
        if(mime.includes("webm")) return "webm";
        const m = url.match(/\.([a-z0-9]{2,5})(?:\?|#|$)/i);
        return (m ? m[1].toLowerCase() : "mp4");
      })();

      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `${filenameBase}.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
      setStatus("Видео сохранено.");
    }catch(e){
      setStatus(`Ошибка сохранения: ${e?.message || e}`);
    }
  }
function deletePhotoSlot(idx) {
    setPhotoSlots((prev) => {
      const next = Array.isArray(prev) ? prev.slice() : Array(9).fill("");
      next[idx] = "";
      return next;
    });
    setActivePhotoIdx((prevIdx) => (prevIdx === idx ? 0 : prevIdx));
  }

  function deleteClipSlot(idx) {
    setClipSlots((prev) => {
      const next = Array.isArray(prev) ? prev.slice() : Array(9).fill("");
      next[idx] = "";
      return next;
    });
    setClipMeta((prev) => {
      const next = Array.isArray(prev) ? prev.slice() : Array(9).fill("");
      next[idx] = "";
      return next;
    });
    setActiveClipIdx((prevIdx) => (prevIdx === idx ? 0 : prevIdx));
  }

  async function mergeClips(){
    if(isMerging) return;
    setIsMerging(true);
    const urls = clipSlots
      .map(sanitizePersistentUrl)
      .map(toStaticVideosPath)
      .filter(Boolean);
    if(urls.length < 2){
      setStatus("Нужно минимум 2 клипа для объединения.");
      setIsMerging(false);
      return;
    }
    setStatus("Объединяем клипы (ffmpeg)…");
    setOverlayText("Объединяем клипы…");
    try{
      const out = await fetchJson("/api/video/merge", { method: "POST", body: { clipUrls: urls } });
      const url = sanitizePersistentUrl(out?.url);
      if(!url) throw new Error("Не удалось собрать итог");
      // Итог объединения хранится отдельно и НЕ попадает в ленту клипов
      setMergedResultUrl(url);
      setStatus("Готово. Итог объединён.");
    }catch(e){
      setStatus(`Ошибка объединения: ${e?.message || e}`);
    }finally{
      setIsMerging(false);
      setOverlayText("");
    }
  }

  function handleEngineClick(key){
    const def = ENGINE_LIST.find(e => e.key === key);
    if(!def) return;
    if(def.disabled){
      setStatus(def.disabledHint || "Этот движок пока недоступен.");
      return;
    }
    if(def.locked && !unlocks?.[def.key]){
      setUnlockTarget(def.key);
      setUnlockCode("");
      setUnlockOpen(true);
      return;
    }
    setEngine(def.key);
  }

  function tryUnlock(){
    // приватный код (только для владельца). Сбрасывается при выходе из аккаунта и закрытии вкладки.
    const ok = (unlockCode || "").trim() === "99541984 Aa";
    if(!ok){
      setStatus("Неверный код доступа.");
      return;
    }
    setUnlocks(prev => ({ ...(prev || {}), [unlockTarget]: true }));
    setEngine(unlockTarget);
    setUnlockOpen(false);
    setStatus(`Движок ${unlockTarget} разблокирован (только для вас).`);
  }



function toggleUltraRef(idx){
  setUltraRefs(prev => {
    const has = prev.includes(idx);
    if(has){
      return prev.filter(x => x !== idx);
    }
    if(prev.length >= 3){
      setStatus("ULTRA (Veo): можно выбрать максимум 3 фото.");
      return prev;
    }
    return [...prev, idx];
  });
}

  const activeClipUrl = resolveAssetUrl(clipSlots[activeClipIdx]);

  const mergedUrl = resolveAssetUrl(mergedResultUrl || "");

  return (
    <div className={`videoPage engine-${engine} ${enginePulse?"enginePulse":""}`}>
      {isVideoDebugEnabled() ? (
        <div style={{
          position: "fixed",
          right: 12,
          bottom: 12,
          zIndex: 9999,
          background: "rgba(0,0,0,0.75)",
          color: "#fff",
          padding: "10px 12px",
          borderRadius: 10,
          maxWidth: 420,
          fontSize: 12,
          lineHeight: 1.35,
          boxShadow: "0 10px 30px rgba(0,0,0,0.35)"
        }}>
          <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", gap: 10}}>
            <b>VIDEO DEBUG</b>
            <button
              type="button"
              onClick={() => { try{ localStorage.setItem("ps_debug_video","0"); }catch{} window.location.search = window.location.search.replace(/([?&])debug=1(&?)/, "$1").replace(/\?&/, "?").replace(/\?$/, ""); }}
              style={{background:"transparent", color:"#fff", border:"1px solid rgba(255,255,255,0.35)", borderRadius: 8, padding:"3px 8px", cursor:"pointer"}}
              title="Выключить debug"
            >
              off
            </button>
          </div>
          <div style={{marginTop:8}}>
            <div><b>accountKey:</b> {accountKey}</div>
            <div><b>hydrated:</b> {String(didHydrateRef.current)} / <b>uploading:</b> {String(uploadingRef.current)}</div>
            <div><b>keys:</b></div>
            <div style={{opacity:0.9, wordBreak:"break-all"}}>photo: {KEY.photo}</div>
            <div style={{opacity:0.9, wordBreak:"break-all"}}>clips: {KEY.clips}</div>
            <div style={{opacity:0.9, wordBreak:"break-all"}}>state: {KEY.state}</div>
            <div style={{marginTop:6}}>
              <b>filled:</b> photos {photoSlots.filter(sanitizePersistentUrl).length}/9, clips {clipSlots.filter(sanitizePersistentUrl).length}/9
            </div>
            <div style={{marginTop:6, opacity:0.9}}>
              <div><b>clip0:</b> {(clipSlots[0]||"").slice(0,90)}</div>
              <div><b>clip1:</b> {(clipSlots[1]||"").slice(0,90)}</div>
            </div>
            <div style={{marginTop:8, display:"flex", gap:8, flexWrap:"wrap"}}>
              <button
                type="button"
                onClick={() => {
                  try{
                    const dump = {
                      accountKey,
                      KEY,
                      didHydrate: didHydrateRef.current,
                      uploading: uploadingRef.current,
                      raw: {
                        photo: localStorage.getItem(KEY.photo),
                        clips: localStorage.getItem(KEY.clips),
                        state: localStorage.getItem(KEY.state),
                        unlock: localStorage.getItem(KEY.unlock),
                      }
                    };
                    console.log("[VIDEO][DUMP]", dump);
                    alert("DUMP отправлен в Console: [VIDEO][DUMP]");
                  }catch(e){
                    console.log("[VIDEO][DUMP] error", e);
                    alert("Не смог сделать DUMP. См. Console.");
                  }
                }}
                style={{background:"rgba(255,255,255,0.12)", color:"#fff", border:"1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding:"5px 10px", cursor:"pointer"}}
              >
                Dump storage
              </button>
              <button
                type="button"
                onClick={() => { try{ localStorage.removeItem(KEY.clips); }catch{} try{ localStorage.removeItem(KEY.photo); }catch{} try{ localStorage.removeItem(KEY.meta); }catch{} try{ localStorage.removeItem(KEY.state); }catch{} window.location.reload(); }}
                style={{background:"rgba(255,120,120,0.15)", color:"#fff", border:"1px solid rgba(255,120,120,0.35)", borderRadius: 8, padding:"5px 10px", cursor:"pointer"}}
                title="Очистить только видео-ключи этого аккаунта и перезагрузить"
              >
                Clear video keys
              </button>
            </div>
            <div style={{marginTop:6, opacity:0.75}}>
              Включение: добавь <b>?debug=1</b> к URL или localStorage <b>ps_debug_video=1</b>
            </div>
          </div>
        </div>
      ) : null}

      <div className="videoHero">
        <h1 className="videoTitle">Создай видео</h1>
        <div className="videoSubtitle">Видео из фото. Верх — исходники (1–9), низ — клипы для объединения.</div>
      </div>

      <div className="videoGrid">
        {/* LEFT */}
        <div className="videoCol">
          <div className="videoCard">
            <div className="videoCardHeader">
              <div>
                <div className="videoCardTitle">Фото</div>
                <div className="videoCardHint">Кадры (1–9). Активный кадр подсвечен. При ручной загрузке — добавляем по индексу.</div>
              </div>
            </div>

            <div className="slotPreview" onClick={() => { if (zoomSrc) setPhotoZoomOpen(true); }}>
              {sanitizePersistentUrl(photoSlots[activePhotoIdx])
                ? <img src={resolveAssetUrl(photoSlots[activePhotoIdx])} alt="frame" />
                : <div className="slotEmpty">Фото не выбрано</div>}
            </div>

            <div className="slotStrip" aria-label="Кадры (1–9)">
              {photoSlots.map((u, idx) => {
                const ok = !!sanitizePersistentUrl(u);
                const isActive = idx === activePhotoIdx;
                return (
                  <button
                    key={idx}
                    ref={(el) => { photoBtnRefs.current[idx] = el; }}
                    className={`slotCard ${isActive ? "active" : ""} ${ok ? "filled" : ""}`}
                    onClick={() => setActivePhotoIdx(idx)}
                    title={`Кадр ${idx + 1}`}
                    type="button"
                  >
                    {ok ? (
                      <img className="slotThumb" src={resolveAssetUrl(u)} alt={`Кадр ${idx + 1}`} />
                    ) : null}
                    <div className="slotNum">{idx + 1}</div>
                    {engine === "ULTRA" && ok ? (
                      <div
                        className={`refCheck ${ultraRefs.includes(idx) ? "on" : ""}`}
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); toggleUltraRef(idx); }}
                        title="Реф для ULTRA (Veo): до 3 фото"
                      />
                    ) : null}
                    {ok ? (
                      <div
                role="button"
                tabIndex={0}
                className="slotX"
                title="Удалить кадр"
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); deletePhotoSlot(idx); } }
                onKeyDown={(e) => { if(e.key==="Enter"||e.key===" "){ e.preventDefault(); e.stopPropagation(); deletePhotoSlot(idx); } } }
                aria-label="Удалить кадр"
              >
                ×
              </div>
                    ) : null}
                  </button>
                );
              })}
            </div>

            <input
              ref={photoFileRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              style={{display:"none"}}
              onChange={(e) => onPickPhotoFile(e.target.files?.[0])}
            />

            <div className="videoLeftBtns">
              <button className="videoBtn" onClick={() => photoFileRef.current?.click()}>Загрузить фото</button>
              <button className="videoBtn secondary" onClick={() => setPhotoSlots(Array(9).fill(""))}>Очистить</button>
              <button className="videoBtn secondary" onClick={saveActivePhoto}>Сохранить</button>
            </div>

            <div className="formatRow">
              <div className="formatLabel">Формат</div>
              {["9:16","1:1","16:9"].map(f => {
                const lockAspect = engine === "ULTRA" && Array.isArray(ultraRefs) && ultraRefs.length > 1;
                const disabled = lockAspect && f !== "16:9";
                return (
                  <button
                    key={f}
                    className={`pill ${format===f?"active":""} ${disabled?"disabled":""}`}
                    disabled={disabled}
                    onClick={() => !disabled && setFormat(f)}
                    title={disabled ? "ULTRA(Veo): при 2–3 фото доступен только формат 16:9" : undefined}
                  >{f}</button>
                );
              })}
            </div>

            <div className="hintBox grow">
              <div className="hintTitle">Подсказка</div>
              <div className="hintText">{formatHint}</div>
            </div>
          </div>
        </div>

        {/* CENTER */}
        <div className="videoCol">
          <div className="videoCard">
            <div className="videoCardTitle">Создание видео</div>

            <div className="sectionTitle">Движение камеры</div>
            <div className="cameraSelectRow">
              <GlassSelect
                ariaLabel="Движение камеры"
                value={camera}
                options={CAMERA_LIST}
                onChange={(v) => setCamera(v)}
              />
            </div>

            <div className="sectionTitle">Освещение</div>
            <div className="row">
              {["Мягкий","Контраст","Тёплый"].map(x => (
                <button key={x} className={`pill ${light===x?"active":""}`} onClick={() => setLight(x)}>{x}</button>
              ))}
            </div>

            <div className="sectionTitle">Модель видео</div>
            <div className="engineRow">
              {ENGINE_LIST.map((e) => {
                const locked = e.locked && !unlocks?.[e.key];
                const active = engine === e.key;
                return (
                  <button
                    key={e.key}
                    className={`engineCard ${active?"active":""} ${locked?"locked":""}`}
                    onClick={() => handleEngineClick(e.key)}
                  >
                    <div className="engineTop">
                      <div className="engineName">{e.label}</div>
                      <div className="engineSub">{e.sub}</div>
                    </div>
                    <div className="engineBottom">
                      {e.disabled ? <span>⛔ Недоступно</span> : (locked ? <span>🔒 Доступ по коду</span> : <span>Доступно</span>)}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="sectionTitle">Длительность</div>
            <div className="row">
              {engine === "ULTRA" ? (
                <>
                  <button className={`pill active disabled` } disabled>8s</button>
                  <div className="hintInline">Veo: длительность фиксированная (8s).</div>
                </>
              ) : (
                [5,10].map(x => (
                  <button
                    key={x}
                    className={`pill ${duration===x?"active":""}`}
                    onClick={() => setDuration(x)}
                  >
                    {x}s
                  </button>
                ))
              )}
            </div>
            <button
              className={`videoBtn primary ${isEngineLocked?"disabled":""}`}
              disabled={isEngineLocked || isGenerating}
              onClick={async () => {
                  if(isGenerating) return;
                  setIsStarting(true);
                  try {
                    if (isEngineLocked) return;
                    if (videoJobId) {
                      setStatus("Генерация уже идёт. Можно выйти/обновить — я подхвачу по job.");
                      return;
                    }
                    setVideoJobState("queued");
                    setStatus("Запуск генерации…");
                    const srcs = (engine === "ULTRA")
                      ? (
                          (ultraRefs && ultraRefs.length)
                            ? ultraRefs.map(i => sanitizePersistentUrl(photoSlots[i])).filter(Boolean)
                            : [sanitizePersistentUrl(photoSlots[activePhotoIdx])].filter(Boolean)
                        )
                      : [sanitizePersistentUrl(photoSlots[activePhotoIdx])].filter(Boolean);

                    if (!srcs.length) {
                      setStatus("Нет исходного фото. Добавь хотя бы 1 кадр (для CINEMA можно до 3).");
                      return;
                    }

                    const lighting = (light === "Контраст" ? "contrast" : (light === "Тёплый" ? "warm" : "soft"));

                    const payload = {
                      engine,
                      sourceImages: srcs,
                      format,
                      seconds: duration,
                      camera: camera,
                      lighting,
                      prompt: "Realistic sportswear studio shoot, natural movement, soft light, no text.",
                      count: 1
                    };

                    const res = await fetchJson("/api/video/generateJob", {
                      method: "POST",
                      body: payload,
                    });

                    const urls = (res && res.videos) ? res.videos : [];
                    if (!urls.length) {
                      setStatus("Генерация завершилась без результата (videos пустой).");
                      return;
                    }

                    setClipSlots(prev => {
                      const next = [...prev];
                      let putAt = next.findIndex(x => !sanitizePersistentUrl(x));
                      if (putAt < 0) putAt = activeClipIdx;
                      for (const u of urls) {
                        if (putAt >= next.length) break;
                        next[putAt] = u;
                        putAt += 1;
                      }
                      return next;
                    });

                    setStatus(`Готово: ${urls.length} клип(а). Можно объединять справа.`);
                    const jobId = res?.jobId;
                    if (!jobId) {
                      setStatus("Job не вернул jobId. Проверь backend /api/video/generateJob");
                      return;
                    }

                    setVideoJobId(jobId);
                    setVideoJobState("running");
                    setVideoJobProgress(0);
                    saveActiveJob(jobId);
                    startOverlay();
                    startPolling(jobId);
                    setStatus("Генерация запущена. Можно выйти/обновить страницу — генерация не пропадёт." );
                  } catch (e) {
                    // backend may return: {detail}, {data:{detail}}, Error(message), etc.
                    const detail =
                      (e && (e.detail ?? e.data?.detail ?? e.response?.detail)) ??
                      (e && (e.message ?? e.data?.message)) ??
                      e;
                    let msg = "";
                    try {
                      msg = typeof detail === "string" ? detail : JSON.stringify(detail);
                    } catch {
                      msg = String(detail);
                    }
                    setStatus(`Ошибка генерации: ${msg}`);
                  } finally {
                    setIsStarting(false);
                    // if we failed before getting a job id, clear queued flag
                    if (!videoJobId) {
                      setVideoJobState((st) => (st === "queued" ? "" : st));
                    }
                  }
                }}
            >
              {isEngineLocked ? "НЕДОСТУПНО (🔒)" : (isGenerating ? "ГЕНЕРАЦИЯ…" : `СДЕЛАТЬ ВИДЕО • ${cost} кр`)}
            </button>

            <div className="statusBox grow">
              <div className="statusTitle">Статус</div>
              <div className="statusText">{statusLog.length ? statusLog.map((m, idx) => (
                <div key={(m?.t||0)+":"+idx} style={{
                  padding: "2px 0",
                  color: m?.level==="error" ? "rgba(255,90,90,0.95)" : (m?.level==="warn" ? "rgba(255,190,60,0.95)" : (m?.level==="ok" ? "rgba(120,255,160,0.95)" : "inherit")),
                  fontWeight: (m?.level==="error" || m?.level==="warn") ? 800 : 600
                }}>{m?.text || ""}</div>
              )) : status}</div>
              {videoJobId ? (
                <div className="statusSmall">Job: {videoJobState || "…"} • {Number(videoJobProgress || 0)}%</div>
              ) : null}
            </div>
          </div>
        </div>

        {/* RIGHT */}
        <div className="videoCol">
          <div className="videoCard">
            <div className="videoCardHeader">
              <div className="videoCardTitle">Видео</div>
            </div>

            <div className="videoPlayer" onDoubleClick={() => { if (sanitizePersistentUrl(clipSlots[activeClipIdx])) setVideoZoomOpen(true); }}>
              {activeClipUrl
                ? <video src={activeClipUrl} controls />
                : <div className="slotEmpty">Видео появится здесь</div>}

              {overlayText && (videoJobId || videoJobState === "running" || videoJobState === "queued") ? (
                <div className="genOverlay" aria-live="polite">
                  <div className="genOverlayCard">
                    <div className="genOverlayTop">Генерация видео</div>
                    <div className="genOverlayText">{overlayText}</div>
                    <div className="genOverlayBottom">
                      {Number(videoJobProgress || 0)}% • можно выйти/обновить — продолжится
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            <input
              ref={clipFileRef}
              type="file"
              accept="video/mp4,video/webm"
              style={{display:"none"}}
              onChange={(e) => onPickClipFile(e.target.files?.[0])}
            />

            <div className="clipStrip" aria-label="Клипы (1–9)">
              {clipSlots.map((u, idx) => {
                const ok = !!sanitizePersistentUrl(u);
                const isActive = idx === activeClipIdx;
                return (
                  <button
                    key={idx}
                    className={`slotCard ${isActive ? "active" : ""} ${ok ? "filled" : ""} ${clipMeta?.[idx] ? "clipKind-" + clipMeta[idx] : ""} ${lastAddedClipIdx===idx ? "clipPulse" : ""}`}
                    onClick={() => setActiveClipIdx(idx)}
                    title={`Клип ${idx + 1}`}
                    type="button"
                  >
                    {ok ? (
                      <video className="slotThumb" src={resolveAssetUrl(u)} muted playsInline preload="metadata" />
                    ) : null}
                    <div className="slotNum">{idx + 1}</div>
                    {ok ? (
                      <div
                role="button"
                tabIndex={0}
                className="slotX"
                title="Удалить клип"
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); deleteClipSlot(idx); } }
                onKeyDown={(e) => { if(e.key==="Enter"||e.key===" "){ e.preventDefault(); e.stopPropagation(); deleteClipSlot(idx); } } }
                aria-label="Удалить клип"
              >
                ×
              </div>
                    ) : null}
                  </button>
                );
              })}
            </div>

            <div className="videoRightBtns">
              <button className="videoBtn" onClick={() => clipFileRef.current?.click()}>Загрузить клип</button>
              <button className="videoBtn secondary" onClick={mergeClips} disabled={isMerging}>{isMerging ? "ОБЪЕДИНЯЕМ…" : "Объединить"}</button>
            </div>

            <div className="mergeResultBox">
              {mergedResultUrl ? (
                <button
                  type="button"
                  className="mergeClearBtn"
                  title="Убрать результат"
                  onClick={() => {
                    setMergedResultUrl("");
                    try { localStorage.removeItem(keys.merged); } catch {}
                    setStatus("Результат объединения очищен.");
                  }}
                >×</button>
              ) : null}
              <div className="mergeResultTitle">Результат</div>
              <div className="mergeResultMini">
                {mergedUrl ? (
                  <video src={mergedUrl} controls />
                ) : (
                  <div className="slotEmpty">Итог объединения появится здесь</div>
                )}
              </div>
              <button
                className="videoBtn secondary"
                disabled={!mergedUrl}
                onClick={() => saveClipUrl(mergedUrl, `merged_${Date.now()}`)}
              >
                Сохранить
              </button>
            </div>
          </div>
        </div>
      </div>

{photoZoomOpen && zoomSrc && (
  <div className="photoZoomOverlay" onMouseDown={() => setPhotoZoomOpen(false)}>
    <img className="photoZoomImg" src={zoomSrc} alt="Просмотр" onMouseDown={(e) => e.stopPropagation()} />
  </div>
)}


{videoZoomOpen && sanitizePersistentUrl(clipSlots[activeClipIdx]) && (
  <div className="photoZoomOverlay" onMouseDown={() => setVideoZoomOpen(false)}>
    <video
      className="photoZoomVideo"
      src={resolveAssetUrl(clipSlots[activeClipIdx])}
      controls
      autoPlay
      onMouseDown={(e) => e.stopPropagation()}
    />
  </div>
)}


      {unlockOpen && (
        <div className="unlockOverlay" onMouseDown={() => setUnlockOpen(false)}>
          <div className="unlockModal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="unlockTitle">Доступ к {unlockTarget}</div>
            <div className="unlockText">Введите код доступа (только для владельца).</div>
            <input className="unlockInput" value={unlockCode} onChange={(e) => setUnlockCode(e.target.value)} placeholder="Код" />
            <div className="unlockBtns">
              <button className="videoBtn" onClick={tryUnlock}>ОК</button>
              <button className="videoBtn secondary" onClick={() => setUnlockOpen(false)}>Отмена</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}