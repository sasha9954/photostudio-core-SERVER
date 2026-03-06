import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  addEdge,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./ClipStoryboardPage.css";

import { API_BASE, fetchJson } from "../services/api";
import { useAuth } from "../app/AuthContext";
import { useNavigate } from "react-router-dom";



// -------------------------
// typed ports + colors (for clear wiring)
// -------------------------
const PORT_COLORS = {
  audio: "#ff4d4f",        // red
  text: "#40a9ff",         // blue
  ref_character: "#73d13d",// green
  ref_location: "#9254de", // purple
  ref_style: "#faad14",    // amber
  plan: "#36cfc9",         // teal
};

function portColor(key) {
  return PORT_COLORS[key] || "#8c8c8c";
}

function isBrainInput(handleId) {
  return handleId === "audio" || handleId === "text" || handleId === "ref_character" || handleId === "ref_location" || handleId === "ref_style";
}

// -------------------------
// helpers
// -------------------------

function getAccountKey(user) {
  // Canonical accountKey:
  // - if user.id exists => "u_<id>" (always prefix to avoid format jumps)
  // - else if email exists => lowercased email
  // - else => "guest"
  const uidRaw = user?.id != null ? String(user.id) : "";
  const emailRaw = user?.email ? String(user.email).toLowerCase() : "";

  let key = "guest";
  if (uidRaw) key = uidRaw.startsWith("u_") ? uidRaw : `u_${uidRaw}`;
  else if (emailRaw) key = emailRaw;

  // Remember last known identity for post-refresh stability
  try {
    if (uidRaw) localStorage.setItem("ps:lastUserId", key);
    if (emailRaw) localStorage.setItem("ps:lastEmail", emailRaw);
  } catch {
    // ignore
  }

  // Fallback: if we're guest but we have a remembered user id/email, use it.
  if (key === "guest") {
    try {
      const lastUserId = localStorage.getItem("ps:lastUserId") || "";
      const lastEmail = (localStorage.getItem("ps:lastEmail") || "").toLowerCase();
      if (lastUserId) return String(lastUserId);
      if (lastEmail) return String(lastEmail);
    } catch {
      // ignore
    }
  }

  return key;
}

function safeGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key, value) {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

function safeDel(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

function fmtTime(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s)) return "—";
  const total = Math.max(0, Math.floor(s));
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function fmtTimeWithMs(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s)) return "—";
  const safe = Math.max(0, s);
  const mm = Math.floor(safe / 60);
  const ss = Math.floor(safe % 60);
  const ms = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function fmtSecAndMs(seconds) {
  if (seconds == null || seconds === "") return "—";
  const s = Number(seconds);
  if (!Number.isFinite(s)) return "—";
  const ms = Math.round(s * 1000);
  return `${s.toFixed(3)}s (${ms}ms)`;
}

function normalizeDurationSec(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function resolveAssetUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return "";
  if (/^https?:\/\//i.test(raw) || raw.startsWith("data:")) return raw;
  if (raw.startsWith("/static/assets/") || raw.startsWith("/assets/")) return `${API_BASE}${raw}`;
  if (raw.startsWith("static/assets/")) return `${API_BASE}/${raw}`;
  return raw;
}

const SCENE_IMAGE_FORMATS = ["9:16", "1:1", "16:9"];
const DEFAULT_SCENE_IMAGE_FORMAT = "9:16";

function normalizeSceneImageFormat(format) {
  return SCENE_IMAGE_FORMATS.includes(format) ? format : DEFAULT_SCENE_IMAGE_FORMAT;
}

function getSceneImageSize(format) {
  const normalized = normalizeSceneImageFormat(format);
  if (normalized === "1:1") return { width: 1024, height: 1024 };
  if (normalized === "16:9") return { width: 1344, height: 768 };
  return { width: 768, height: 1344 };
}

async function uploadAsset(file) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/api/assets/upload`, {
    method: "POST",
    body: fd,
    credentials: "include",
  });
  if (!res.ok) {
    let txt = "";
    try {
      txt = await res.text();
    } catch {
      // ignore
    }
    throw new Error(`upload_failed:${res.status}:${txt || res.statusText}`);
  }
  return await res.json();
}


async function getAudioDurationSec(url) {
  // Browser-side duration probe (metadata only)
  return await new Promise((resolve) => {
    try {
      const a = new Audio();
      a.preload = "metadata";
      a.onloadedmetadata = () => {
        const d = Number.isFinite(a.duration) ? a.duration : null;
        resolve(d);
      };
      a.onerror = () => resolve(null);
      a.src = url;
    } catch {
      resolve(null);
    }
  });
}


// -------------------------
// node UIs
// -------------------------

function NodeShell({ title, icon, children, className = "", onClose }) {
  return (
    <div className={`clipSB_node ${className}`}>
      <div className="clipSB_nodeHeader">
        <div className="clipSB_nodeIcon">{icon}</div>
        <div className="clipSB_nodeTitle">{title}</div>
        {onClose ? (
          <button className="clipSB_nodeClose" onClick={onClose} title="Удалить ноду">×</button>
        ) : null}
      </div>
      <div className="clipSB_nodeBody">{children}</div>
    </div>
  );
}

function AudioNode({ id, data }) {
  const inputRef = useRef(null);

  const onPick = () => inputRef.current?.click();
  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    data?.onUpload?.(id, f);
    // allow picking same file again
    e.target.value = "";
  };

  return (
    <>
      <Handle type="source" position={Position.Right} id="audio" className="clipSB_handle" style={{ background: portColor("audio"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <NodeShell title="AUDIO" onClose={() => data?.onRemoveNode?.(id)} icon={<span aria-hidden>🎧</span>} className="clipSB_nodeAudio">
        <div className="clipSB_row" style={{ marginBottom: 8 }}>
          <div className="clipSB_rowLabel" style={{ fontSize: 12, opacity: 0.85 }}>Тип</div>
          <select
            className="clipSB_select"
            style={{ width: "100%", marginTop: 6 }}
            value={data?.audioType || "bg"}
            onChange={(e) => data?.onAudioType?.(id, e.target.value)}
          >
            <option value="bg">бит / фон</option>
            <option value="song">песня (вокал + музыка)</option>
          </select>
        </div>

          {data?.uploading ? (
            <div className="clipSB_btn clipSB_btnMuted">Загрузка…</div>
          ) : data?.audioName ? (
            <div className="clipSB_fileRow">
              <div className="clipSB_fileName" title={data.audioName}>
                Файл: {data.audioName}
              </div>
              <button className="clipSB_x" onClick={() => data?.onClear?.(id)} title="Удалить">
                ×
              </button>
            </div>
          ) : (
            <button className="clipSB_btn" onClick={onPick}>
              Загрузить файл
            </button>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,.mp3,.wav,.ogg,.m4a"
            style={{ display: "none" }}
            onChange={onFile}
          />
        <div className="clipSB_hint" style={{ marginTop: 10 }}>
          mp3 / wav / ogg{data?.audioDurationSec ? ` • ${Math.round(data.audioDurationSec)}s` : ""}
        </div>
      </NodeShell>
    </>
  );
}

function TextNode({ id, data }) {
  return (
    <>
      <Handle type="source" position={Position.Right} id="text" className="clipSB_handle" style={{ background: portColor("text"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <NodeShell title="TEXT" onClose={() => data?.onRemoveNode?.(id)} icon={<span aria-hidden>📄</span>} className="clipSB_nodeText">
        <div className="clipSB_row" style={{ marginBottom: 8 }}>
          <div className="clipSB_rowLabel" style={{ fontSize: 12, opacity: 0.85 }}>Тип</div>
          <select
            className="clipSB_select"
            style={{ width: "100%", marginTop: 6 }}
            value={data?.textType || "story"}
            onChange={(e) => data?.onTextType?.(id, e.target.value)}
          >
            <option value="story">история / сюжет</option>
            <option value="lyrics">текст песни (lyrics)</option>
            <option value="notes">заметки</option>
          </select>
        </div>

        <div className="clipSB_textWrap">
          <textarea
            className="clipSB_textarea"
            placeholder="Текст для истории / слова песни / идея сюжета…"
            value={data?.textValue || ""}
            onChange={(e) => data?.onChange?.(id, e.target.value)}
          />
          <button className="clipSB_clear" title="Очистить" onClick={() => data?.onChange?.(id, "")}
          >
            ⟲
          </button>
        </div>
        <div className="clipSB_hint" style={{ marginTop: 10 }}>
          выход: текст → BRAIN
        </div>
      </NodeShell>
    </>
  );
}

function BrainNode({ id, data }) {
  const scenario = data?.scenarioKey || "song_meaning";
  const shoot = data?.shootKey || "cinema";
  const style = data?.styleKey || "realism";
  const freezeStyle = !!data?.freezeStyle;

  // clip length mode: auto from AUDIO duration or manual value
  const clipMode = data?.clipMode || "auto"; // "auto" | "manual"
  const manualSec = Math.max(5, Math.min(3600, Number(data?.clipSec || 30)));
  const audioSecRaw = Number(data?.audioDurationSec || 0);
  const audioSec = Number.isFinite(audioSecRaw) && audioSecRaw > 0 ? audioSecRaw : 0;
  const clipSec = clipMode === "auto" ? (audioSec > 0 ? Math.round(audioSec) : 30) : manualSec;

  return (
    <>
      {/* typed inputs */}
      <div style={{ position: "absolute", top: 110, left: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>AUDIO</div>
      <div style={{ position: "absolute", top: 150, left: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>TEXT</div>
      <div style={{ position: "absolute", top: 190, left: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>REF Персонаж</div>
      <div style={{ position: "absolute", top: 230, left: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>REF Локация</div>
      <div style={{ position: "absolute", top: 270, left: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>REF Стиль</div>

      {/* typed output */}
      <div style={{ position: "absolute", top: 310, right: 18, fontSize: 11, opacity: 0.75, pointerEvents: "none" }}>PLAN</div>

      {/* explicit ports (позже удобно валидировать связи) */}
      <Handle type="target" position={Position.Left} id="audio" style={{ top: 116, background: portColor("audio"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="target" position={Position.Left} id="text" style={{ top: 156, background: portColor("text"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="target" position={Position.Left} id="ref_character" style={{ top: 196, background: portColor("ref_character"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="target" position={Position.Left} id="ref_location" style={{ top: 236, background: portColor("ref_location"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="target" position={Position.Left} id="ref_style" style={{ top: 276, background: portColor("ref_style"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="source" position={Position.Right} id="plan" style={{ top: 176, background: portColor("plan"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />

      <NodeShell
        title="BRAIN"
        onClose={() => data?.onRemoveNode?.(id)}
        icon={<span aria-hidden>🧠</span>}
        className="clipSB_nodeBrain"
      >
        <div className="clipSB_small" style={{ marginBottom: 10 }}>
          Настройки планирования сцен (пока без генерации).
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <div className="clipSB_hint" style={{ marginBottom: 6 }}>
              Сценарий
            </div>
            <select
              className="clipSB_select"
              value={scenario}
              onChange={(e) => data?.onScenario?.(id, e.target.value)}
            >
              <option value="song_meaning">по смыслу</option>
              <option value="beats">по биту/ритму</option>
              <option value="verses">по куплетам/припевам</option>
              <option value="mini_story">мини‑история</option>
              <option value="dynamic">динамичный монтаж</option>
              <option value="clip">клип (performance + монтаж)</option>
            </select>
          </div>

          <div>
            <div className="clipSB_hint" style={{ marginBottom: 6 }}>
              Съёмка
            </div>
            <select
              className="clipSB_select"
              value={shoot}
              onChange={(e) => data?.onShoot?.(id, e.target.value)}
            >
              <option value="cinema">кино</option>
              <option value="clip">клиповая</option>
              <option value="doc">док/реализм</option>
              <option value="pov">POV</option>
              <option value="static">статичная камера</option>
            </select>
          </div>

          <div>
            <div className="clipSB_hint" style={{ marginBottom: 6 }}>
              Стиль
            </div>
            <select
              className="clipSB_select"
              value={style}
              onChange={(e) => data?.onStyle?.(id, e.target.value)}
            >
              <option value="realism">реализм</option>
              <option value="neon">неон/кибер</option>
              <option value="film">плёнка/грейн</option>
              <option value="glossy">глянец/реклама</option>
              <option value="soft">мягкий свет</option>
            </select>
          </div>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 12px",
              borderRadius: 12,
              background: "rgba(255,255,255,0.05)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)",
              color: "rgba(255,255,255,0.86)",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={freezeStyle}
              onChange={(e) => data?.onFreezeStyle?.(id, e.target.checked)}
              style={{ width: 16, height: 16 }}
            />
            <span>Freeze Style — фиксировать стиль на все сцены</span>
          </label>
        </div>

        <div style={{ marginTop: 10 }}>
          <label
            className="clipSB_check"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 12px",
              borderRadius: 12,
              background: "rgba(255,255,255,0.04)",
              boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)",
              color: "rgba(255,255,255,0.86)",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={!!data?.wantLipSync}
              onChange={(e) => data?.onWantLipSync?.(id, e.target.checked)}
              style={{ width: 16, height: 16 }}
            />
            <span>Липсинк — добавить сцены с ртом (1–3)</span>
          </label>
        </div>

        <div className="clipSB_small" style={{ marginTop: 10 }}>
          Следующий шаг: кнопка “Разобрать” → ScenePlan (таймкоды/описания).
        </div>

        <div style={{ marginTop: 10 }}>
          <div className="clipSB_hint" style={{ marginBottom: 6 }}>
            Длина клипа
          </div>

          <select
            className="clipSB_select"
            value={clipMode}
            onChange={(e) => data?.onClipMode?.(id, e.target.value)}
          >
            <option value="auto">авто — по AUDIO</option>
            <option value="manual">вручную</option>
          </select>

          {clipMode === "manual" ? (
            <input
              className="clipSB_input"
              type="number"
              min={5}
              max={3600}
              step={1}
              value={clipSec}
              onChange={(e) => data?.onClipSec?.(id, e.target.value)}
              style={{ marginTop: 8 }}
            />
          ) : (
            <div className="clipSB_small" style={{ marginTop: 8, opacity: 0.9 }}>
              Берём длительность из подключённой AUDIO-ноды (если есть). Если AUDIO не подключена — используем 30 сек.
            </div>
          )}
        </div>

        <button className="clipSB_btn" onClick={() => data?.onParse?.(id)} disabled={!!data?.isParsing} style={{ marginTop: 10 }}>
          Разобрать (бесплатно)
        </button>

        {data?.isParsing ? (
          <div className="clipSB_small" style={{ marginTop: 8, opacity: 0.9 }}>Думаю над сценами…</div>
        ) : data?.lastParseError ? (
          <div className="clipSB_small" style={{ marginTop: 8, color: "#ff7875" }}>Ошибка: {String(data.lastParseError)}</div>
        ) : null}
        {(!data?.isParsing && !data?.lastParseError && data?.scenePlan?.engine) ? (
          <div className="clipSB_small" style={{ marginTop: 8, opacity: 0.95 }}>
            <div>
              engine: <b>{String(data.scenePlan.engine)}</b>
              {data.scenePlan.modelUsed ? <> · model: <span style={{ opacity: 0.9 }}>{String(data.scenePlan.modelUsed)}</span></> : null}
              {typeof data.scenePlan.sceneCount === "number" ? <> · scenes: <b>{String(data.scenePlan.sceneCount)}</b></> : null}
            </div>
            {Array.isArray(data.scenePlan.warnings) && data.scenePlan.warnings.length ? (
              <div style={{ marginTop: 4, color: "#ffb86c" }}>warnings: {data.scenePlan.warnings.join(", ")}</div>
            ) : null}
            {data.scenePlan.rejectedReason ? (
              <div style={{ marginTop: 4, color: "#ff7875" }}>rejectedReason: {String(data.scenePlan.rejectedReason)}</div>
            ) : null}
            <div style={{ marginTop: 4 }}>
              repairRetryUsed: <b>{data.scenePlan.repairRetryUsed ? "true" : "false"}</b>
              {data.scenePlan.audioHint ? <> · audio.hint: <span style={{ opacity: 0.9 }}>{String(data.scenePlan.audioHint)}</span></> : null}
            </div>
            {data.scenePlan.hint ? (
              <div style={{ marginTop: 4, color: "#ffb86c" }}>hint: {String(data.scenePlan.hint).slice(0, 180)}</div>
            ) : null}
          </div>
        ) : null}

        <div className="clipSB_hint" style={{ marginTop: 10 }}>
          входы: AUDIO/TEXT/REF • выход: PLAN → Storyboard
        </div>
      </NodeShell>
    </>
  );
}

function RefNode({ id, data }) {
  const title = data?.title || "REFERENCE";
  const icon = data?.icon || "📷";
  const kind = data?.kind || "ref";
  const url = data?.url || "";
  const name = data?.name || "";

  return (
    <>
      <Handle type="source" position={Position.Right} id={kind} className="clipSB_handle" style={{ background: portColor(kind), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <NodeShell
        title={title}
        onClose={() => data?.onRemoveNode?.(id)}
        icon={<span aria-hidden>{icon}</span>}
        className="clipSB_nodeRef"
      >
        <div className="clipSB_fileRow" style={{ marginBottom: 10 }}>
          <div className="clipSB_fileName" title={name || url || ""}>
            {name || (url ? url : "нет изображения")}
          </div>
        </div>

        <button className="clipSB_btn" onClick={() => data?.onPickImage?.(id)}>
          Загрузить фото
        </button>

        <div className="clipSB_hint" style={{ marginTop: 10 }}>
          подключай к BRAIN (персонаж/локация/стиль)
        </div>
      </NodeShell>
    </>
  );
}

function StoryboardPlanNode({ id, data }) {
  const scenes = data?.scenes || [];
  return (
    <>
      <Handle type="target" position={Position.Left} id="plan_in" className="clipSB_handle" style={{ background: portColor("plan"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <Handle type="source" position={Position.Right} id="plan_out" className="clipSB_handle" style={{ background: portColor("plan"), width: 12, height: 12, border: "2px solid rgba(255,255,255,0.35)" }} />
      <NodeShell
        title="STORYBOARD"
        onClose={() => data?.onRemoveNode?.(id)}
        icon={<span aria-hidden>🎞️</span>}
        className="clipSB_nodeStoryboard"
      >
        <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
          <button className="clipSB_btn clipSB_btnSecondary" onClick={() => data?.onOpenScenario?.(id)} disabled={scenes.length === 0}>
            Сценарий
          </button>
        </div>

        {scenes.length === 0 ? (
          <div className="clipSB_small">Пусто. Нажми «Разобрать» в BRAIN (бесплатно).</div>
        ) : (
          <div className="clipSB_planList">
            {scenes.map((s, idx) => (
              <div key={idx} className="clipSB_planRow">
                <div className="clipSB_planTime">{s.t0}s → {s.t1}s</div>
                <div className="clipSB_planText">{s.sceneText || s.imagePrompt || s.prompt}</div>
              </div>
            ))}
          </div>
        )}
      </NodeShell>
    </>
  );
}

function AssemblyNode({ id, data }) {
  return (
    <>
      <Handle type="target" position={Position.Left} />
      <NodeShell
        title="ASSEMBLY"
        onClose={() => data?.onRemoveNode?.(id)}
        icon={<span aria-hidden>🎬</span>}
        className="clipSB_nodeAssembly"
      >
        <button
          className="clipSB_btn"
          onClick={() => window.dispatchEvent(new CustomEvent("ps:clipOpenScenario", { detail: { nodeId: data?.scenarioNodeId || null } }))}
          title="Открыть сценарий"
        >
          Сценарий
        </button>

        <button className="clipSB_btn clipSB_btnMuted" disabled style={{ marginTop: 8 }}>
          Собрать (скоро)
        </button>

        <div className="clipSB_hint" style={{ marginTop: 10 }}>
          скоро: склейка сцен, музыка, lip-sync, экспорт mp4
        </div>
      </NodeShell>
    </>
  );
}

// -------------------------
// page
// -------------------------

export default function ClipStoryboardPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const accountKey = useMemo(() => getAccountKey(user) || "guest", [user]);
  const STORE_KEY = useMemo(() => `ps:clipStoryboard:v1:${accountKey}`, [accountKey]);

  const didHydrateRef = useRef(false);
  const isHydratingRef = useRef(true);
  const selectedEdgeRef = useRef(null);
  const parseTokenRef = useRef(0);

  const [lastSavedAt, setLastSavedAt] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);

  
const [scenarioEditor, setScenarioEditor] = useState({
  open: false,
  nodeId: null,
  selected: 0,
});

// Open scenario overlay from node button (custom event)
useEffect(() => {
  const handler = (e) => {
    const nodeId = e?.detail?.nodeId || null;
    setScenarioEditor((s) => ({
      ...s,
      open: true,
      nodeId,
      selected: 0,
    }));
  };

  window.addEventListener("ps:clipOpenScenario", handler);
  return () => window.removeEventListener("ps:clipOpenScenario", handler);
}, []);

// Fullscreen canvas on this page (hide global sidebar)
  useEffect(() => {
    document.body.classList.add("ps_clip_fullscreen");
    return () => document.body.classList.remove("ps_clip_fullscreen");
  }, []);

  const defaultNodes = useMemo(
    () => [
      {
        id: "audio",
        type: "audioNode",
        position: { x: 120, y: 120 },
        data: { audioUrl: "", audioName: "", uploading: false, audioType: "bg" },
      },
      {
        id: "text",
        type: "textNode",
        position: { x: 120, y: 300 },
        data: { textValue: "", textType: "story" },
      },
      {
        id: "brain",
        type: "brainNode",
        position: { x: 520, y: 200 },
        data: { mode: "song_meaning" },
      },
      {
        id: "assembly",
        type: "assemblyNode",
        position: { x: 860, y: 200 },
        data: {},
      },
    ],
    []
  );

  const defaultEdges = useMemo(
    () => [
      { id: "e-audio-brain", source: "audio", target: "brain" },
      { id: "e-text-brain", source: "text", target: "brain" },
      { id: "e-brain-assembly", source: "brain", target: "assembly" },
    ],
    []
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes);

const scenarioNode = useMemo(() => {
  if (scenarioEditor.nodeId) return nodes.find((n) => n.id === scenarioEditor.nodeId) || null;
  return nodes.find((n) => n.type === "storyboardNode") || null;
}, [nodes, scenarioEditor.nodeId]);

const scenarioScenes = useMemo(() => {
  const arr = scenarioNode?.data?.scenes;
  return Array.isArray(arr) ? arr : [];
}, [scenarioNode]);

const scenarioSelected = scenarioScenes[scenarioEditor.selected] || null;
const scenarioSelectedImageFormat = normalizeSceneImageFormat(scenarioSelected?.imageFormat);
const scenarioSelectedIndexLabel = Number.isFinite(scenarioEditor.selected) ? scenarioEditor.selected + 1 : 0;
const scenarioSelectedT0 = Number(scenarioSelected?.t0 ?? scenarioSelected?.start ?? 0);
const scenarioSelectedT1 = Number(scenarioSelected?.t1 ?? scenarioSelected?.end ?? 0);
const scenarioSelectedExpectedSliceSec = Number(
  scenarioSelected?.audioSliceExpectedDurationSec ?? Math.max(0, scenarioSelectedT1 - scenarioSelectedT0)
);
const globalAudioUrlRaw = useMemo(() => {
  const audioNodeWithUrl = nodes.find((n) => n.type === "audioNode" && n?.data?.audioUrl);
  return audioNodeWithUrl?.data?.audioUrl ? String(audioNodeWithUrl.data.audioUrl) : "";
}, [nodes]);
const globalAudioUrlResolved = useMemo(() => resolveAssetUrl(globalAudioUrlRaw), [globalAudioUrlRaw]);
const scenarioSelectedAudioSliceUrl = useMemo(() => resolveAssetUrl(scenarioSelected?.audioSliceUrl), [scenarioSelected?.audioSliceUrl]);
  const [scenarioImageLoading, setScenarioImageLoading] = useState(false);
  const [scenarioImageError, setScenarioImageError] = useState("");
  const [scenarioVideoLoading, setScenarioVideoLoading] = useState(false);
  const [scenarioVideoError, setScenarioVideoError] = useState("");
  const [scenarioVideoOpen, setScenarioVideoOpen] = useState(false);
  const [assemblyHint, setAssemblyHint] = useState("");
  const [lightboxUrl, setLightboxUrl] = useState("");

  useEffect(() => {
    if (!lightboxUrl) return;
    const onKeyDown = (e) => {
      if (e.key === "Escape") setLightboxUrl("");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [lightboxUrl]);

  useEffect(() => {
    setScenarioVideoError("");
    setScenarioVideoLoading(false);
    setScenarioVideoOpen(false);
  }, [scenarioEditor.selected]);

  const updateScenarioScene = useCallback((idx, patch) => {
    if (!scenarioNode?.id || idx < 0) return;
    setNodes((prev) => prev.map((n) => {
      if (n.id !== scenarioNode.id) return n;
      const scenes = Array.isArray(n?.data?.scenes) ? n.data.scenes : [];
      if (!scenes[idx]) return n;
      const nextScenes = scenes.map((s, i) => (i === idx ? { ...s, ...patch } : s));
      return { ...n, data: { ...n.data, scenes: nextScenes } };
    }));
  }, [scenarioNode?.id, setNodes]);

  const handleGenerateScenarioImage = useCallback(async () => {
    if (!scenarioSelected) return;
    const sceneId = String(scenarioSelected.id || `s${scenarioEditor.selected + 1}`);
    const prompt = String(scenarioSelected.imagePrompt || scenarioSelected.sceneText || "").trim();
    const imageFormat = normalizeSceneImageFormat(scenarioSelected.imageFormat);
    const { width, height } = getSceneImageSize(imageFormat);
    if (!prompt) {
      setScenarioImageError("Добавьте Prompt (Image)");
      return;
    }

    setScenarioImageLoading(true);
    setScenarioImageError("");
    try {
      const out = await fetchJson(`/api/clip/image`, {
        method: "POST",
        body: {
          sceneId,
          prompt: `${prompt}\nAspect ratio: ${imageFormat}`,
          width,
          height,
        },
      });
      if (!out?.ok || !out?.imageUrl) throw new Error(out?.hint || out?.code || "image_generation_failed");
      updateScenarioScene(scenarioEditor.selected, { imageUrl: String(out.imageUrl || ""), imageFormat });
    } catch (e) {
      console.error(e);
      setScenarioImageError(String(e?.message || e));
    } finally {
      setScenarioImageLoading(false);
    }
  }, [scenarioSelected, scenarioEditor.selected, updateScenarioScene]);

  const handleClearScenarioImage = useCallback(() => {
    setScenarioImageError("");
    updateScenarioScene(scenarioEditor.selected, { imageUrl: "" });
  }, [scenarioEditor.selected, updateScenarioScene]);

  const handleScenarioVideoTakeAudio = useCallback(async () => {
    if (!scenarioSelected) return;
    if (!globalAudioUrlRaw) {
      setScenarioVideoError("Не найден общий audioUrl в Audio node");
      return;
    }
    const sceneId = String(scenarioSelected.id || `s${scenarioEditor.selected + 1}`);
    const t0 = Number(scenarioSelected.t0 ?? scenarioSelected.start ?? 0);
    const t1 = Number(scenarioSelected.t1 ?? scenarioSelected.end ?? 0);

    setScenarioVideoLoading(true);
    setScenarioVideoError("");
    try {
      const out = await fetchJson("/api/clip/audio-slice", {
        method: "POST",
        body: { sceneId, t0, t1, audioUrl: globalAudioUrlRaw },
      });
      if (!out?.ok || !out?.audioSliceUrl) throw new Error(out?.hint || out?.code || "audio_slice_failed");
      const outT0 = Number(out?.t0 ?? t0);
      const outT1 = Number(out?.t1 ?? t1);
      const expectedDuration = Math.max(0, outT1 - outT0);
      updateScenarioScene(scenarioEditor.selected, {
        audioSliceUrl: String(out.audioSliceUrl || ""),
        audioSliceT0: outT0,
        audioSliceT1: outT1,
        audioSliceExpectedDurationSec: expectedDuration,
        audioSliceBackendDurationSec: normalizeDurationSec(out?.audioSliceBackendDurationSec ?? out?.duration),
        audioSliceActualDurationSec: null,
        audioSliceLoadError: "",
      });
    } catch (e) {
      console.error(e);
      setScenarioVideoError(String(e?.message || e));
    } finally {
      setScenarioVideoLoading(false);
    }
  }, [globalAudioUrlRaw, scenarioEditor.selected, scenarioSelected, updateScenarioScene]);

  const handleScenarioSliceLoadedMetadata = useCallback((event) => {
    if (!scenarioSelected) return;
    const duration = Number(event?.currentTarget?.duration);
    updateScenarioScene(scenarioEditor.selected, {
      audioSliceActualDurationSec: Number.isFinite(duration) ? duration : null,
      audioSliceLoadError: "",
    });
  }, [scenarioEditor.selected, scenarioSelected, updateScenarioScene]);

  const handleScenarioSliceAudioError = useCallback(() => {
    if (!scenarioSelected) return;
    const msg = "Не удалось загрузить вырезанный mp3-срез. Проверьте URL и наличие файла.";
    updateScenarioScene(scenarioEditor.selected, {
      audioSliceActualDurationSec: null,
      audioSliceLoadError: msg,
    });
    setScenarioVideoError(msg);
  }, [scenarioEditor.selected, scenarioSelected, updateScenarioScene]);

  const handleScenarioGenerateVideo = useCallback(async () => {
    if (!scenarioSelected?.imageUrl) return;
    if (scenarioSelected?.lipSync && !scenarioSelected?.audioSliceUrl) {
      setScenarioVideoError("Для lipSync сначала возьмите аудио");
      return;
    }

    const sceneId = String(scenarioSelected.id || `s${scenarioEditor.selected + 1}`);
    const t0 = Number(scenarioSelected.t0 ?? scenarioSelected.start ?? 0);
    const t1 = Number(scenarioSelected.t1 ?? scenarioSelected.end ?? 0);
    const dur = Math.max(0, t1 - t0);
    const requestedDurationSec = dur <= 5 ? 5 : 10;

    setScenarioVideoLoading(true);
    setScenarioVideoError("");
    try {
      const endpoint = scenarioSelected?.lipSync ? "/api/clip/lipsync" : "/api/clip/video";
      const out = await fetchJson(endpoint, {
        method: "POST",
        body: {
          sceneId,
          imageUrl: scenarioSelected.imageUrl,
          audioSliceUrl: scenarioSelected.audioSliceUrl || "",
          videoPrompt: scenarioSelected.videoPrompt || "",
          requestedDurationSec,
          lipSync: !!scenarioSelected.lipSync,
          format: normalizeSceneImageFormat(scenarioSelected.imageFormat),
        },
      });
      if (!out?.ok || !out?.videoUrl) throw new Error(out?.hint || out?.code || "video_generation_failed");
      updateScenarioScene(scenarioEditor.selected, { videoUrl: String(out.videoUrl || "") });
    } catch (e) {
      console.error(e);
      setScenarioVideoError(String(e?.message || e));
    } finally {
      setScenarioVideoLoading(false);
    }
  }, [scenarioEditor.selected, scenarioSelected, updateScenarioScene]);

  const handleScenarioClearVideo = useCallback(() => {
    setScenarioVideoError("");
    updateScenarioScene(scenarioEditor.selected, { videoUrl: "" });
  }, [scenarioEditor.selected, updateScenarioScene]);

  const handleScenarioAddToVideo = useCallback(() => {
    if (!scenarioSelected?.imageUrl) return;
    setScenarioVideoOpen(true);
    setScenarioVideoError("");
  }, [scenarioSelected]);

  const handleScenarioAssembly = useCallback(() => {
    const segments = scenarioScenes
      .filter((s) => s?.videoUrl)
      .map((s, idx) => ({
        sceneId: String(s.id || `s${idx + 1}`),
        t0: Number(s.t0 ?? s.start ?? 0),
        t1: Number(s.t1 ?? s.end ?? 0),
        videoUrl: String(s.videoUrl || ""),
      }));
    if (!segments.length) return;
    const payload = { segments, audioUrl: globalAudioUrlRaw || "" };
    console.info("assembly_todo_payload", payload);
    setAssemblyHint("TODO: страница монтажа пока не подключена");
  }, [globalAudioUrlRaw, scenarioScenes]);

  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges);
  const edgesRef = useRef([]);

  useEffect(() => {
    edgesRef.current = edges || [];
  }, [edges]);


  const removeNode = useCallback((nodeId) => {
    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
    setEdges((prev) => prev.filter((e) => e.source !== nodeId && e.target !== nodeId));
  }, [setNodes, setEdges]);

  
  // wire handlers into node.data (keeps render simple)
  const bindHandlers = useCallback(
    (ns) =>
      ns.map((n) => {
        const base = { ...n, data: { ...n.data, onRemoveNode: (nodeId) => removeNode(nodeId) } };
        if (n.type === "audioNode") {
          return {
            ...base,
            data: {
              ...base.data,
              onAudioType: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, audioType: value } } : x)));
              },
              onUpload: async (nodeId, file) => {
                // optimistic ui
                setNodes((prev) =>
                  prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, uploading: true } } : x))
                );
                try {
                  const out = await uploadAsset(file);
                  const dur = out?.url ? await getAudioDurationSec(out.url) : null;
                  setNodes((prev) =>
                    prev.map((x) =>
                      x.id === nodeId
                        ? {
                            ...x,
                            data: {
                              ...x.data,
                              uploading: false,
                              audioUrl: out?.url || "",
                              audioName: out?.name || file.name,
                              audioDurationSec: (out?.durationSec ?? dur ?? x.data?.audioDurationSec ?? null),
                            },
                          }
                        : x
                    )
                  );
                } catch (e) {
                  setNodes((prev) =>
                    prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, uploading: false } } : x))
                  );
                  alert(
                    "Не удалось загрузить аудио на сервер. Файл выбран, но не сохранён. Проверь backend /api/assets/upload."
                  );
                  console.error(e);
                }
              },
              onClear: (nodeId) => {
                setNodes((prev) =>
                  prev.map((x) =>
                    x.id === nodeId
                      ? { ...x, data: { ...x.data, audioUrl: "", audioName: "", audioDurationSec: null, uploading: false } }
                      : x
                  )
                );
              },
            },
          };
        }
        if (n.type === "textNode") {
          return {
            ...base,
            data: {
              ...base.data,
              onTextType: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, textType: value } } : x)));
              },
              onChange: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, textValue: value } } : x)));
              },
            },
          };
        }
        if (n.type === "brainNode") {
          return {
            ...base,
            data: {
              ...base.data,
              onMode: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, mode: value } } : x)));
              },
              onScenario: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, scenarioKey: value } } : x)));
              },
              onShoot: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, shootKey: value } } : x)));
              },
              onStyle: (nodeId, value) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, styleKey: value } } : x)));
              },
              onFreezeStyle: (nodeId, checked) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, freezeStyle: !!checked } } : x)));
              },
              onWantLipSync: (nodeId, checked) => {
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, wantLipSync: !!checked } } : x)));
              },
                            onClipMode: (nodeId, mode) => {
                const m = mode === "manual" ? "manual" : "auto";
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, clipMode: m } } : x)));
              },
onClipSec: (nodeId, value) => {
                const num = Number(value);
                const safe = Number.isFinite(num) ? Math.max(5, Math.min(3600, Math.floor(num))) : 30;
                setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, clipSec: safe } } : x)));
              },
onParse: async (nodeId) => {
  const brainCurrent = nodes.find((x) => x.id === nodeId);
  if (brainCurrent?.data?.isParsing) return;

  const parseToken = parseTokenRef.current + 1;
  parseTokenRef.current = parseToken;

  setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, isParsing: true, activeParseToken: parseToken } } : x)));

  try {
    const brainNow = nodes.find((x) => x.id === nodeId);
    if (!brainNow) return;

    // --- inputs by explicit handles (no conflicts) ---
    const inEdges = edges.filter((e) => e.target === nodeId);

    const pickSourceNode = (handleId) => {
      // 1) Prefer explicit handle wiring (ReactFlow targetHandle)
      const edgeExplicit = [...inEdges].reverse().find((e) => (e.targetHandle || "") === handleId);
      if (edgeExplicit) return nodes.find((x) => x.id === edgeExplicit.source) || null;

      // 2) Backward-compat / default edges: if targetHandle is missing,
      // map by well-known default node ids ("audio","text") so BRAIN can actually see inputs.
      if (handleId === "audio") {
        const e0 = [...inEdges].reverse().find((e) => e.source === "audio");
        if (e0) return nodes.find((x) => x.id === "audio") || null;
      }
      if (handleId === "text") {
        const e0 = [...inEdges].reverse().find((e) => e.source === "text");
        if (e0) return nodes.find((x) => x.id === "text") || null;
      }

      return null;
    };

    const audioNode = pickSourceNode("audio");
    const textNode = pickSourceNode("text");
    const refCharNode = pickSourceNode("ref_character");
    const refLocNode = pickSourceNode("ref_location");
    const refStyleNode = pickSourceNode("ref_style");

    const refCharacter = refCharNode?.type === "refNode" && refCharNode?.data?.kind === "ref_character" ? (refCharNode.data?.url || "") : "";
    const refLocation = refLocNode?.type === "refNode" && refLocNode?.data?.kind === "ref_location" ? (refLocNode.data?.url || "") : "";
    const refStyle = refStyleNode?.type === "refNode" && refStyleNode?.data?.kind === "ref_style" ? (refStyleNode.data?.url || "") : "";

    const scenarioKey = brainNow.data?.scenarioKey || "beat_rhythm";
    const shootKey = brainNow.data?.shootKey || "cinema";
    const styleKey = brainNow.data?.styleKey || "realism";
    const freezeStyle = !!brainNow.data?.freezeStyle;
    const wantLipSync = !!brainNow.data?.wantLipSync;

    const audioUrl = audioNode?.type === "audioNode" ? (audioNode.data?.audioUrl || "") : "";
    const audioType = audioNode?.type === "audioNode" ? (audioNode.data?.audioType || "bg") : null; // bg | song
    const textType = textNode?.type === "textNode" ? (textNode.data?.textType || "story") : null; // story | lyrics | notes
    const textValue = textNode?.type === "textNode" ? String(textNode.data?.textValue || "") : "";

    const payload = {
      audioUrl: audioUrl || null,
      text: textValue || null,
      scenarioKey,
      shootKey,
      styleKey,
      freezeStyle,
      wantLipSync,
      refCharacter: refCharacter || null,
      refLocation: refLocation || null,
      refStyle: refStyle || null,
      audioType,
      textType,
    };

    const res = await fetch(`${API_BASE}/api/clip/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const out = await res.json().catch(() => ({}));
    if (!res.ok || !out?.ok) throw new Error(out?.detail || out?.hint || "clip_plan_failed");

    if (parseTokenRef.current !== parseToken) return;

    const audioDuration = Number(out?.audioDuration || 30);
    const scenesRaw = Array.isArray(out?.scenes) ? out.scenes : [];
    const validation = out?.plannerDebug?.validation || {};

    // Map to storyboard format (t0/t1/prompt)
    const scenes = scenesRaw
      .map((s, idx) => {
        const t0 = Number(s.start ?? s.t0 ?? 0);
        const t1 = Number(s.end ?? s.t1 ?? 0);
        const prompt = String(s.imagePrompt || s.prompt || s.sceneText || `Scene ${idx + 1}`);
        return {
          id: s.id || `s${String(idx + 1).padStart(2, "0")}`,
          start: t0,
          end: t1,
          t0,
          t1,
          prompt,
          sceneText: s.sceneText || "",
          imagePrompt: s.imagePrompt || "",
          videoPrompt: s.videoPrompt || "",
          imageUrl: s.imageUrl || "",
          imageFormat: normalizeSceneImageFormat(s.imageFormat),
          audioSliceUrl: s.audioSliceUrl || "",
          audioSliceT0: Number(s.audioSliceT0 ?? t0),
          audioSliceT1: Number(s.audioSliceT1 ?? t1),
          audioSliceExpectedDurationSec: Number(s.audioSliceExpectedDurationSec ?? Math.max(0, t1 - t0)),
          audioSliceBackendDurationSec: normalizeDurationSec(s.audioSliceBackendDurationSec),
          audioSliceActualDurationSec: normalizeDurationSec(s.audioSliceActualDurationSec),
          audioSliceLoadError: s.audioSliceLoadError || "",
          videoUrl: s.videoUrl || "",
          why: s.why || "",
          audioType: s.audioType || "mixed",
          sceneType: s.sceneType || "visual_rhythm",
          hasVocals: !!s.hasVocals,
          isLipSync: !!(s.isLipSync ?? s.lipSync),
          lyricFragment: s.lyricFragment || "",
          timingReason: s.timingReason || s.why || "",
          beatAnchor: s.beatAnchor || "",
          performanceType: s.performanceType || "cinematic_visual",
          shotType: s.shotType || "",
        };
      })
      .filter((s) => Number.isFinite(s.t0) && Number.isFinite(s.t1) && s.t1 > s.t0);

    setNodes((prev) => {
      const updated = prev.map((x) =>
        x.id === nodeId
          ? {
              ...x,
              data: {
                ...x.data,
                scenes,
                isParsing: false,
                activeParseToken: parseToken,
                lastParseError: null,
                lastPlanMeta: {
                  engine: out.engine || "gemini",
                  modelUsed: out.modelUsed || null,
                  fallbackUsed: !!out.fallbackUsed,
                  hint: out.hint || null,
                  error: out.error || null,
                  sceneCount: Number(validation.sceneCount ?? scenes.length),
                  warnings: Array.isArray(validation.warnings) ? validation.warnings : [],
                  rejectedReason: validation.rejectedReason || null,
                  repairRetryUsed: !!validation.repairRetryUsed,
                  audioHint: out?.plannerDebug?.audio?.hint || null,
                },
                scenePlan: {
                  engine: out.engine || "gemini",
                  modelUsed: out.modelUsed || null,
                  hint: out.hint || null,
                  audioDuration,
                  scenes,
                  sceneCount: Number(validation.sceneCount ?? scenes.length),
                  warnings: Array.isArray(validation.warnings) ? validation.warnings : [],
                  rejectedReason: validation.rejectedReason || null,
                  repairRetryUsed: !!validation.repairRetryUsed,
                  audioHint: out?.plannerDebug?.audio?.hint || null,
                  refs: { refCharacter, refLocation, refStyle },
                  settings: { scenarioKey, shootKey, styleKey, freezeStyle },
                },
              },
            }
          : x
      );

      // flood to storyboard/results node
      const targets = (edgesRef.current || []).filter((e) => e.source === nodeId).map((e) => e.target);
      return updated.map((x) =>
        targets.includes(x.id) && (x.type === "storyboardNode" || x.type === "resultsNode")
          ? { ...x, data: { ...x.data, scenes } }
          : x
      );
    });
  } catch (err) {
    if (parseTokenRef.current !== parseToken) return;
    console.error(err);
    setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, isParsing: false, activeParseToken: parseToken, lastParseError: String(err?.message || err) } } : x)));
  }
},
            },
          };
        }

        if (n.type === "refNode") {
          return {
            ...base,
            data: {
              ...base.data,
              onPickImage: (nodeId) => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.onchange = async () => {
                  const f = input.files && input.files[0];
                  if (!f) return;
                  try {
                    const out = await uploadAsset(f);
                    const url = out?.url || "";
                    setNodes((prev) => prev.map((x) => (x.id === nodeId ? { ...x, data: { ...x.data, url, name: out?.name || f.name } } : x)));
                  } catch (err) {
                    console.error(err);
                  }
                };
                input.click();
              },
            },
          };
        }
        
        if (n.type === "storyboardNode") {
          return {
            ...base,
            data: {
              ...base.data,
              onOpenScenario: (nodeId) => {
                try {
                  window.dispatchEvent(new CustomEvent("ps:clipOpenScenario", { detail: { nodeId } }));
                } catch (e) {}
              },
            },
          };
        }
return base;
      }),
    [setNodes, removeNode, edges]
  );

const hydrate = useCallback(() => {
    // IMPORTANT: we always have an accountKey (fallback to "guest"), so persistence works even before auth init.
    didHydrateRef.current = false;
    isHydratingRef.current = true;

    // Try current key first; if empty, try a few compatible legacy keys (to survive format changes)
    let raw = safeGet(STORE_KEY);
    if (!raw) {
      const candidates = [];
      const add = (k) => {
        if (k && !candidates.includes(k)) candidates.push(k);
      };

      add(STORE_KEY);

      // legacy: without/with u_ prefix
      if (accountKey.startsWith("u_")) add(`ps:clipStoryboard:v1:${accountKey.slice(2)}`);
      else add(`ps:clipStoryboard:v1:u_${accountKey}`);

      // legacy: email key
      const emailRaw = user?.email ? String(user.email).toLowerCase() : "";
      if (emailRaw) add(`ps:clipStoryboard:v1:${emailRaw}`);

      // guest key (if user worked before login)
      add("ps:clipStoryboard:v1:guest");

      for (const k of candidates) {
        const v = safeGet(k);
        if (v) {
          raw = v;
          // migrate into current key so next loads are stable
          if (k !== STORE_KEY) safeSet(STORE_KEY, v);
          break;
        }
      }
    }
    if (!raw) {
      setNodes(bindHandlers(defaultNodes));
      setEdges(defaultEdges);
      // mark hydrated on next tick to avoid wiping storage with default state
      setTimeout(() => {
        didHydrateRef.current = true;
        isHydratingRef.current = false;
      }, 0);
      return;
    }

    try {
      const parsed = JSON.parse(raw);
      const savedNodes = Array.isArray(parsed?.nodes) ? parsed.nodes : null;
      const savedEdges = Array.isArray(parsed?.edges) ? parsed.edges : null;

      if (!savedNodes || !savedEdges) throw new Error("bad_format");

      // sanitize
      const cleanNodes = savedNodes
        .filter((n) => n && typeof n.id === "string" && typeof n.type === "string" && n.position)
        .map((n) => ({
          id: n.id,
          type: n.type,
          position: n.position,
          data: n.data || {},
        }));
      const cleanEdges = savedEdges
        .filter((e) => e && typeof e.id === "string" && e.source && e.target)
        .map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle || null, target: e.target, targetHandle: e.targetHandle || null, style: e.style, animated: e.animated, data: e.data }));

      setNodes(bindHandlers(cleanNodes.length ? cleanNodes : defaultNodes));
      setEdges(cleanEdges.length ? cleanEdges : defaultEdges);
    } catch {
      setNodes(bindHandlers(defaultNodes));
      setEdges(defaultEdges);
    } finally {
      // mark hydrated on next tick so persist effect can't overwrite storage
      setTimeout(() => {
        didHydrateRef.current = true;
        isHydratingRef.current = false;
      }, 0);
    }
  }, [STORE_KEY, setNodes, setEdges, defaultNodes, defaultEdges, accountKey, user]);

  const addNodeFromDrawer = useCallback((type) => {
    const id = `${type}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    const centerX = 360;
    const centerY = 220;
    const jitterX = Math.round((Math.random() - 0.5) * 120);
    const jitterY = Math.round((Math.random() - 0.5) * 120);

    let node;
    if (type === "audio") {
      node = { id, type: "audioNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { audioUrl: "", audioName: "", uploading: false, audioType: "bg" } };
    } else if (type === "text") {
      node = { id, type: "textNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { textValue: "", textType: "story" } };
    } else if (type === "brain") {
      node = { id, type: "brainNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { mode: "song_meaning", scenarioKey: "song_meaning", shootKey: "cinema", styleKey: "realism", freezeStyle: false, clipSec: 30 } };
    } else if (type === "ref_character") {
      node = { id, type: "refNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { title: "REF — ПЕРСОНАЖ", icon: "🧍", kind: "ref_character", url: "", name: "" } };
    } else if (type === "ref_location") {
      node = { id, type: "refNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { title: "REF — ЛОКАЦИЯ", icon: "📍", kind: "ref_location", url: "", name: "" } };
    } else if (type === "ref_style") {
      node = { id, type: "refNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { title: "REF — СТИЛЬ", icon: "🎨", kind: "ref_style", url: "", name: "" } };
    } else if (type === "storyboard") {
      node = { id, type: "storyboardNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: { scenes: [] } };
    } else if (type === "assembly") {
      node = { id, type: "assemblyNode", position: { x: centerX + jitterX, y: centerY + jitterY }, data: {} };
    } else {
      return;
    }

    setNodes((prev) => bindHandlers(prev.concat(node)));
    setDrawerOpen(false);
  }, [setNodes, bindHandlers]);


  // hydrate on account change
  useEffect(() => {
    hydrate();
  }, [hydrate]);

  // re-hydrate when session changes (logout/login without full reload)
  useEffect(() => {
    const onSessionChanged = () => {
      // wait a tick so AuthContext can update ps:lastUserId/ps:lastEmail
      setTimeout(() => {
        hydrate();
      }, 0);
    };
    window.addEventListener("ps:sessionChanged", onSessionChanged);
    return () => window.removeEventListener("ps:sessionChanged", onSessionChanged);
  }, [hydrate]);




  // persist
  useEffect(() => {
    if (!didHydrateRef.current) return;
    if (isHydratingRef.current) return;

    // strip handlers from data
    const serialNodes = nodes.map((n) => {
      const d = { ...(n.data || {}) };
      delete d.onUpload;
      delete d.onClear;
      delete d.onChange;
      delete d.onMode;
      delete d.onFreezeStyle;
      delete d.onStyle;
      delete d.onShoot;
      delete d.onScenario;
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        data: d,
      };
    });
    const serialEdges = edges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle || null, target: e.target, targetHandle: e.targetHandle || null }));

    const ok = safeSet(STORE_KEY, JSON.stringify({ nodes: serialNodes, edges: serialEdges }));
    if (ok) setLastSavedAt(Date.now());
  }, [nodes, edges, STORE_KEY, accountKey]);

  // extra safety: flush to storage on page unload (helps when navigating away quickly)
  useEffect(() => {
    const onBeforeUnload = () => {
      if (!didHydrateRef.current) return;
      if (isHydratingRef.current) return;

      const serialNodes = nodes.map((n) => {
        const d = { ...(n.data || {}) };
        delete d.onUpload;
        delete d.onClear;
        delete d.onChange;
        delete d.onMode;
      delete d.onFreezeStyle;
      delete d.onStyle;
      delete d.onShoot;
      delete d.onScenario;
        return { id: n.id, type: n.type, position: n.position, data: d };
      });
      const serialEdges = edges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle || null, target: e.target, targetHandle: e.targetHandle || null }));
      const ok = safeSet(STORE_KEY, JSON.stringify({ nodes: serialNodes, edges: serialEdges }));
    if (ok) setLastSavedAt(Date.now());
    };

    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [nodes, edges, STORE_KEY, accountKey]);


  const nodeTypes = useMemo(
    () => ({
      audioNode: AudioNode,
      textNode: TextNode,
      brainNode: BrainNode,
      refNode: RefNode,
      storyboardNode: StoryboardPlanNode,
      assemblyNode: AssemblyNode,
    }),
    []
  );

  const onConnect = useCallback(
    (params) => {
      // simple connection validation to avoid conflicts:
      // BRAIN reads ONLY by targetHandle (audio/text/ref_*). One connection per handle.
      setEdges((eds) => {
        const src = nodes.find((n) => n.id === params.source);
        const dst = nodes.find((n) => n.id === params.target);
        if (!src || !dst) return eds;

        // --- validate connections into BRAIN ---
        if (dst.type === "brainNode") {
          const h = params.targetHandle || "";
          const ok =
            (h === "audio" && src.type === "audioNode" && (params.sourceHandle || "") === "audio") ||
            (h === "text" && src.type === "textNode" && (params.sourceHandle || "") === "text") ||
            (h === "ref_character" && src.type === "refNode" && (params.sourceHandle || "") === "ref_character") ||
            (h === "ref_location" && src.type === "refNode" && (params.sourceHandle || "") === "ref_location") ||
            (h === "ref_style" && src.type === "refNode" && (params.sourceHandle || "") === "ref_style");

          if (!ok) return eds;

          // remove old edge(s) that already occupy the same handle
          const cleaned = eds.filter((e) => !(e.target === dst.id && (e.targetHandle || "") === h));
          return addEdge({ ...params, animated: false, style: { stroke: portColor(h), strokeWidth: 2 }, data: { kind: h } }, cleaned);
        }

        // --- validate PLAN route: BRAIN(plan) -> STORYBOARD(plan_in) ---
        if (src.type === "brainNode" && (params.sourceHandle || "") === "plan") {
          if (dst.type === "storyboardNode" && (params.targetHandle || "") === "plan_in") {
            // only one plan edge into storyboard
            const cleaned = eds.filter((e) => !(e.target === dst.id && (e.targetHandle || "") === "plan_in"));
            return addEdge({ ...params, animated: false, style: { stroke: portColor("plan"), strokeWidth: 2 }, data: { kind: "plan" } }, cleaned);
          }
        }

        // default: allow
        return addEdge({ ...params, animated: false }, eds);
      });
    },
    [setEdges, nodes]
  );

  const onEdgeClick = useCallback((evt, edge) => {
    evt?.stopPropagation?.();
    selectedEdgeRef.current = edge?.id || null;
  }, []);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const id = selectedEdgeRef.current;
      if (!id) return;
      setEdges((eds) => eds.filter((x) => x.id !== id));
      selectedEdgeRef.current = null;
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setEdges]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") setDrawerOpen(false);
      if (e.key === "Tab") {
        e.preventDefault();
        setDrawerOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const stats = useMemo(
    () => ({
      nodes: nodes.length,
      edges: edges.length,
    }),
    [nodes.length, edges.length]
  );

  return (
    <div className="clipSB_root">
      <div className="clipSB_hud">
        <button className="clipSB_hudBtn" onClick={() => setDrawerOpen(true)}>Ноды</button>
        <button className="clipSB_hudBtn clipSB_hudBtnSecondary" onClick={() => navigate("/studio")}>К студиям</button>
      </div>

      
      {scenarioEditor.open ? (
        <div className="clipSB_scenarioOverlay" onClick={() => {
          setScenarioEditor((s) => ({ ...s, open: false }));
          setLightboxUrl("");
        }}>
          <div className="clipSB_scenarioPanel" onClick={(e) => e.stopPropagation()}>
            <div className="clipSB_scenarioHeader">
              <div className="clipSB_scenarioTitle">SCENARIO</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  className="clipSB_btn clipSB_btnSecondary"
                  onClick={handleScenarioAssembly}
                  disabled={!scenarioScenes.some((s) => s?.videoUrl)}
                >
                  Монтаж
                </button>
                <div className="clipSB_scenarioMeta">
                  {scenarioScenes.length} сцен
                </div>
                <button className="clipSB_iconBtn" onClick={() => {
                  setScenarioEditor((s) => ({ ...s, open: false }));
                  setLightboxUrl("");
                }} aria-label="Закрыть">
                  ✕
                </button>
              </div>
            </div>

            <div className="clipSB_scenarioBody">
              <div className="clipSB_scenarioList">
                {scenarioScenes.map((s, i) => (
                  <button
                    key={s.id || i}
                    className={"clipSB_scenarioItem" + (i === scenarioEditor.selected ? " isActive" : "")}
                    onClick={() => setScenarioEditor((x) => ({ ...x, selected: i }))}
                  >
                    <div className="clipSB_scenarioItemInner">
                      {s.imageUrl ? (
                        <img
                          src={s.imageUrl}
                          alt="scene"
                          className="clipSB_scenarioThumb"
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            setLightboxUrl(s.imageUrl);
                          }}
                        />
                      ) : (
                        <div className="clipSB_scenarioThumb clipSB_scenarioThumbPlaceholder" />
                      )}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="clipSB_scenarioItemTop">
                          <div className="clipSB_scenarioItemTime">
                            {fmtTime(s.start)} → {fmtTime(s.end)}
                          </div>
                          <div className="clipSB_scenarioTags">
                            {s.imageUrl ? <div className="clipSB_tag">IMG</div> : null}
                            {s.videoUrl ? <div className="clipSB_tag clipSB_tagDone">VIDEO ✓</div> : null}
                            {s.lipSync ? <div className="clipSB_tag">LIP</div> : null}
                          </div>
                        </div>
                        <div className="clipSB_scenarioItemText">{(s.sceneText || "").slice(0, 90) || "—"}</div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              <div className="clipSB_scenarioEdit">
                {scenarioSelected ? (
                  <>
                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint">Тайм</div>
                      <div className="clipSB_kv">
                        <span>{fmtTime(scenarioSelected.start)}</span>
                        <span>—</span>
                        <span>{fmtTime(scenarioSelected.end)}</span>
                        <span style={{ opacity: 0.7 }}>({Math.max(0, (scenarioSelected.end || 0) - (scenarioSelected.start || 0)).toFixed(1)}s)</span>
                      </div>
                    </div>

                    <div className="clipSB_scenarioEditRow">
                      <label className="clipSB_check" style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <input
                          type="checkbox"
                          checked={!!scenarioSelected.lipSync}
                          onChange={(e) => updateScenarioScene(scenarioEditor.selected, { lipSync: !!e.target.checked })}
                        />
                        <span>Этот кадр под липсинк (рот/лицо видно)</span>
                      </label>
                    </div>

                    {scenarioSelected.lipSync ? (
                      <div className="clipSB_scenarioEditRow">
                        <div className="clipSB_hint">Текст для губ (коротко)</div>
                        <input
                          className="clipSB_input"
                          value={scenarioSelected.lipSyncText || ""}
                          onChange={(e) => updateScenarioScene(scenarioEditor.selected, { lipSyncText: e.target.value })}
                          placeholder="фраза/строка припева…"
                        />
                      </div>
                    ) : null}

                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint">Описание сцены</div>
                      <textarea
                        className="clipSB_textarea"
                        rows={6}
                        value={scenarioSelected.sceneText || ""}
                        onChange={(e) => updateScenarioScene(scenarioEditor.selected, { sceneText: e.target.value })}
                      />
                    </div>

                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint">Prompt (Image)</div>
                      <textarea
                        className="clipSB_textarea"
                        rows={5}
                        value={scenarioSelected.imagePrompt || ""}
                        onChange={(e) => updateScenarioScene(scenarioEditor.selected, { imagePrompt: e.target.value })}
                      />
                    </div>

                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint" style={{ marginBottom: 6 }}>Изображение сцены</div>
                      <div className="clipSB_aspectPills" style={{ marginBottom: 8 }}>
                        {SCENE_IMAGE_FORMATS.map((format) => (
                          <button
                            key={format}
                            type="button"
                            className={`clipSB_aspectPill${scenarioSelectedImageFormat === format ? " isActive" : ""}`}
                            onClick={() => updateScenarioScene(scenarioEditor.selected, { imageFormat: format })}
                          >
                            {format}
                          </button>
                        ))}
                      </div>
                      <div className="clipSB_scenarioPreviewWrap">
                        {scenarioSelected.imageUrl ? (
                          <img
                            src={scenarioSelected.imageUrl}
                            alt="scene preview"
                            className="clipSB_scenarioPreview"
                            onClick={() => setLightboxUrl(scenarioSelected.imageUrl)}
                          />
                        ) : (
                          <div className="clipSB_scenarioPreview clipSB_scenarioPreviewPlaceholder">Превью отсутствует</div>
                        )}
                      </div>
                      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                        <button className="clipSB_btn clipSB_btnSecondary" onClick={handleGenerateScenarioImage} disabled={scenarioImageLoading}>
                          {scenarioImageLoading ? "Генерация..." : "Сгенерировать изображение"}
                        </button>
                        <button className="clipSB_btn clipSB_btnSecondary" onClick={handleClearScenarioImage}>
                          Очистить изображение
                        </button>
                        {scenarioSelected.imageUrl ? (
                          <button className="clipSB_btn clipSB_btnSecondary" onClick={handleScenarioAddToVideo}>
                            Добавить в Видео
                          </button>
                        ) : null}
                      </div>
                      {scenarioImageError ? <div className="clipSB_hint" style={{ color: "#ff8a8a", marginTop: 6 }}>{scenarioImageError}</div> : null}
                    </div>

                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint">Prompt (Video)</div>
                      <textarea
                        className="clipSB_textarea"
                        rows={4}
                        value={scenarioSelected.videoPrompt || ""}
                        onChange={(e) => updateScenarioScene(scenarioEditor.selected, { videoPrompt: e.target.value })}
                      />
                    </div>

                    <div className="clipSB_scenarioEditRow">
                      <div className="clipSB_hint">Scene decision debug</div>
                      <div className="clipSB_kv" style={{ display: "grid", gridTemplateColumns: "160px 1fr", rowGap: 6, columnGap: 8, alignItems: "start" }}>
                        <span>audioType</span><span>{String(scenarioSelected.audioType || "") || "—"}</span>
                        <span>sceneType</span><span>{String(scenarioSelected.sceneType || "") || "—"}</span>
                        <span>hasVocals</span><span>{String(!!scenarioSelected.hasVocals)}</span>
                        <span>isLipSync</span><span>{String(!!scenarioSelected.isLipSync || !!scenarioSelected.lipSync)}</span>
                        <span>lyricFragment</span><span>{String(scenarioSelected.lyricFragment || "") || "—"}</span>
                        <span>timingReason</span><span>{String(scenarioSelected.timingReason || "") || "—"}</span>
                        <span>beatAnchor</span><span>{String(scenarioSelected.beatAnchor || "") || "—"}</span>
                        <span>performanceType</span><span>{String(scenarioSelected.performanceType || "") || "—"}</span>
                        <span>shotType</span><span>{String(scenarioSelected.shotType || "") || "—"}</span>
                      </div>
                    </div>

                    <div className="clipSB_scenarioEditRow clipSB_audioDebugBlock">
                      <div className="clipSB_audioDebugTitle">AUDIO SLICE DEBUG</div>
                      <div className="clipSB_audioDebugSceneLabel">
                        Scene {scenarioSelectedIndexLabel} · {fmtTimeWithMs(scenarioSelectedT0)} → {fmtTimeWithMs(scenarioSelectedT1)}
                      </div>

                      <div className="clipSB_audioDebugSection">
                        <div className="clipSB_hint" style={{ marginBottom: 6 }}>Полный трек (original audioUrl)</div>
                        {globalAudioUrlResolved ? (
                          <audio
                            key={`full-track-${globalAudioUrlResolved}`}
                            className="clipSB_audioPlayer"
                            controls
                            preload="metadata"
                            src={globalAudioUrlResolved}
                          />
                        ) : (
                          <div className="clipSB_hint" style={{ color: "#ffb4b4" }}>Полный трек не найден: загрузите аудио в Audio node.</div>
                        )}
                        <div className="clipSB_audioDebugUrl">{globalAudioUrlResolved || "—"}</div>
                      </div>

                      <div className="clipSB_audioDebugSection">
                        <div className="clipSB_hint" style={{ marginBottom: 6 }}>Срез по сцене (audioSliceUrl)</div>
                        {scenarioSelectedAudioSliceUrl ? (
                          <audio
                            key={`slice-${String(scenarioSelected.id || scenarioEditor.selected)}-${String(scenarioSelectedAudioSliceUrl || "")}`}
                            className="clipSB_audioPlayer"
                            controls
                            preload="metadata"
                            src={scenarioSelectedAudioSliceUrl}
                            onLoadedMetadata={handleScenarioSliceLoadedMetadata}
                            onError={handleScenarioSliceAudioError}
                          />
                        ) : (
                          <div className="clipSB_hint">Срез ещё не создан. Нажмите «Взять аудио».</div>
                        )}
                        <div className="clipSB_audioDebugUrl">{scenarioSelectedAudioSliceUrl || "—"}</div>
                      </div>

                      <div className="clipSB_audioDebugGrid">
                        <div className="clipSB_videoKv"><span>t0</span><span>{fmtSecAndMs(Number(scenarioSelected.audioSliceT0 ?? scenarioSelectedT0))}</span></div>
                        <div className="clipSB_videoKv"><span>t1</span><span>{fmtSecAndMs(Number(scenarioSelected.audioSliceT1 ?? scenarioSelectedT1))}</span></div>
                        <div className="clipSB_videoKv"><span>expected duration</span><span>{fmtSecAndMs(scenarioSelectedExpectedSliceSec)}</span></div>
                        <div className="clipSB_videoKv"><span>backend duration</span><span>{fmtSecAndMs(scenarioSelected.audioSliceBackendDurationSec)}</span></div>
                        <div className="clipSB_videoKv"><span>actual duration</span><span>{fmtSecAndMs(scenarioSelected.audioSliceActualDurationSec)}</span></div>
                      </div>

                      {scenarioSelected.audioSliceLoadError ? (
                        <div className="clipSB_hint" style={{ color: "#ff8a8a", marginTop: 6 }}>{scenarioSelected.audioSliceLoadError}</div>
                      ) : null}
                    </div>

                    {scenarioVideoOpen ? (
                      <div className="clipSB_scenarioEditRow clipSB_videoBlock">
                        <div className="clipSB_hint" style={{ marginBottom: 8 }}>Видео сцены</div>
                        <div className="clipSB_videoPreviewWrap">
                          {scenarioSelected.videoUrl ? (
                            <video
                              key={String(scenarioSelected.id || scenarioEditor.selected) + ":" + String(scenarioSelected.videoUrl || "")}
                              className="clipSB_videoPlayer"
                              controls
                              playsInline
                              preload="metadata"
                              src={scenarioSelected.videoUrl}
                              poster={scenarioSelected.imageUrl || ""}
                            />
                          ) : scenarioSelected.imageUrl ? (
                            <img className="clipSB_videoPoster" src={scenarioSelected.imageUrl} alt="poster" />
                          ) : null}

                          {scenarioVideoLoading ? (
                            <div className="clipSB_videoOverlay">
                              <span className="clipSB_videoLoadingPulse">Генерация видео...</span>
                            </div>
                          ) : null}
                        </div>
                        <details className="clipSB_videoDetails">
                          <summary>Детали</summary>
                          <div className="clipSB_videoKv"><span>imageUrl</span><span>{scenarioSelected.imageUrl || "—"}</span></div>
                          <div className="clipSB_videoKv"><span>audioSliceUrl</span><span>{scenarioSelected.audioSliceUrl || "—"}</span></div>
                          <div className="clipSB_videoKv"><span>videoUrl</span><span>{scenarioSelected.videoUrl || "—"}</span></div>
                        </details>
                        <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
                          <button className="clipSB_btn clipSB_btnSecondary" onClick={handleScenarioVideoTakeAudio} disabled={scenarioVideoLoading || !globalAudioUrlRaw}>
                            Взять аудио
                          </button>
                          <button
                            className="clipSB_btn clipSB_btnSecondary"
                            onClick={handleScenarioGenerateVideo}
                            disabled={scenarioVideoLoading || !scenarioSelected.imageUrl || (scenarioSelected.lipSync && !scenarioSelected.audioSliceUrl)}
                          >
                            Сгенерировать видео
                          </button>
                          <button className="clipSB_btn clipSB_btnSecondary" onClick={handleScenarioClearVideo} disabled={scenarioVideoLoading}>
                            Очистить видео
                          </button>
                        </div>
                        {scenarioVideoError ? <div className="clipSB_hint" style={{ color: "#ff8a8a", marginTop: 6 }}>{scenarioVideoError}</div> : null}
                      </div>
                    ) : null}
                    {assemblyHint ? <div className="clipSB_hint" style={{ marginTop: 6 }}>{assemblyHint}</div> : null}
                  </>
                ) : (
                  <div className="clipSB_empty">Нет выбранной сцены</div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {lightboxUrl ? (
        <div className="clipSB_lightbox" onClick={() => setLightboxUrl("")}>
          <img
            className="clipSB_lightboxImg"
            src={lightboxUrl}
            alt="Full preview"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      ) : null}

{drawerOpen ? (
        <div className="clipSB_drawerOverlay" onClick={() => setDrawerOpen(false)}>
          <div className="clipSB_drawer" onClick={(e) => e.stopPropagation()}>
            <div className="clipSB_drawerHeader">
              <div className="clipSB_drawerTitle">Ноды</div>
              <button className="clipSB_drawerClose" onClick={() => setDrawerOpen(false)} title="Закрыть">×</button>
            </div>
            <div className="clipSB_drawerList">
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("audio")}>🎧 Аудио</button>
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("text")}>📄 Текст</button>
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("brain")}>🧠 Мозг</button>
              <div className="clipSB_drawerSep" />
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("ref_character")}>🧍 REF — Персонаж</button>
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("ref_location")}>📍 REF — Локация</button>
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("ref_style")}>🎨 REF — Стиль</button>
              <div className="clipSB_drawerSep" />
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("storyboard")}>🎞️ Storyboard</button>
              <button className="clipSB_drawerItem" onClick={() => addNodeFromDrawer("assembly")}>🎬 Сборка</button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="clipSB_canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onEdgeClick={onEdgeClick}
          defaultEdgeOptions={{ style: { strokeDasharray: "6 6" } }}
          connectionLineStyle={{ strokeDasharray: "6 6" }}
          fitView
          minZoom={0.2}
          maxZoom={2}
        >
          <Background gap={24} size={1} color="rgba(255,255,255,0.07)" />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}
