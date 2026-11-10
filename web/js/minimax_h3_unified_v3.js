// MiniMaxH3 LK long-video studio frontend.
//
// The segment editor is rendered entirely with ComfyUI's native widget
// language: standard litegraph widgets (combo / number / toggle / button /
// multiline text) for every scalar control, plus one canvas-drawn widget per
// segment for the reference chips row. No in-node DOM overlay is installed,
// so other extensions see and hit-test plain widgets like on any native node.
//
// Segment data still flows to the backend through the hidden `segments_json`
// STRING widget (schema unchanged), so existing workflows keep loading.
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE = "LKH3Studio";
const RH_NODE_IDS = [
    NODE,
    "LKH3Settings",
    "LKH3Shot",
    "LKH3Shots",
];
const WIDTH = 500;
const ASPECTS = {
    adaptive: null,
    "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 3 / 2, "2:3": 2 / 3, "4:3": 4 / 3,
    "3:4": 3 / 4, "1:1": 1, "21:9": 21 / 9,
};
const imageSlots = ["first_frame", "last_frame", ...Array.from({ length: 9 }, (_, i) => `ref_image_${i + 1}`)];
const videoSlots = Array.from({ length: 3 }, (_, i) => `ref_video_${i + 1}`);
const audioSlots = ["hybrid_audio", ...Array.from({ length: 3 }, (_, i) => `ref_audio_${i + 1}`)];
const mediaSlots = [...imageSlots, ...videoSlots, ...audioSlots];
const MAX_SEGMENTS = 8;
const MAX_SEG_IMAGES = 6;
const MAX_SEG_VIDEOS = 2;
const MAX_SEG_AUDIOS = 3;
const MAX_SEG_LORAS = 4;
const SEGMENT_LORA_ROUTE = "/lk/minimax_h3/loras";

const DOM_TRANSLATIONS = {
    "adaptive": "自适应",
    "match": "匹配",
    "1.2x": "1.2倍",
    "1.5x": "1.5倍",
    "2x": "2倍",
    "max": "最大值",
    "Mute video": "不参考音频",
    "Unmute video": "参考音频",
    "None": "无",

    "Refresh": "刷新",
    "Loading": "加载中",

    "Add LoRA": "＋ LoRA",
    // Long video segment editor
    "Long video segments": "长视频分段",
    "Segment": "分段",
    "Segments": "分段数量",
    "Duration (seconds)": "时长（秒）",
    "Prompt": "提示词",
    "Reference media": "参考素材",
    "Run": "运行",
    "Skipped": "跳过",
    "Copy previous": "复制上一段",
    "Remove segment": "删除分段",
    "Segment prompt: click a reference chip to insert its tag": "分段提示词：点击素材标签插入标记",
    "First frame": "首帧",
    "Last frame": "尾帧",
    "Picture": "图片",
    "Video": "视频",
    "Audio": "音频",
    "(none)": "（无）",
    "LoRA stack (model strength)": "LoRA 栈（模型强度）",
    "Model strength": "模型强度",
    "Refresh LoRA list": "刷新 LoRA 列表",
    "No LoRA found: check ComfyUI/models/loras, then click the refresh button.": "未找到 LoRA：请确认 ComfyUI/models/loras 下有文件，然后点刷新按钮。",
    "Run range": "运行范围",
    "Start": "起",
    "End": "止",
    "Latent continuation": "潜空间链接",
    "Latent continuation is ON: the previous segment's last 7 latent tokens are carried into the next segment; no first-frame keyframe and no pixel overlap are used.": "潜空间续接开启：上一段最后 7 个 latent token 带入下一段；不使用首帧关键帧和像素重叠。",
    "Latent continuation is OFF: segments are joined with a hard cut and no seam blending.": "潜空间续接关闭：各段直接硬切，不做接缝融合。",
    "Shots 1-4": "镜头组 1–4",
    "Shots 5-8": "镜头组 5–8",
};

function currentLocale() {
    const candidates = [
        app?.ui?.settings?.getSettingValue?.("Comfy.Locale"),
        app?.ui?.settings?.getSettingValue?.("Comfy.Language"),
        document.documentElement?.lang,
        document.body?.dataset?.locale,
        document.body?.dataset?.language,
        localStorage.getItem("Comfy.Settings.Language"),
        localStorage.getItem("Comfy.Settings.language"),
        localStorage.getItem("language"),
        localStorage.getItem("locale"),
        navigator.language,
    ];
    return candidates.find(value => typeof value === "string" && value.trim())?.toLowerCase() || "en";
}

function isChineseLocale() { return /^(zh|cn|中文)/i.test(currentLocale()); }
function t(text) { return isChineseLocale() ? (DOM_TRANSLATIONS[text] || text) : text; }

const ASPECT_INTERNAL_VALUES = [
    "adaptive", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1", "21:9",
];
const ASPECT_DISPLAY_VALUES = {
    adaptive: "自适应",
    "16:9": "16:9",
    "9:16": "9:16",
    "3:2": "3:2",
    "2:3": "2:3",
    "4:3": "4:3",
    "3:4": "3:4",
    "1:1": "1:1",
    "21:9": "21:9",
};
function aspectInternalValue(value) {
    const aliases = {
        "自适应": "adaptive",
        "16:9 (Widescreen)": "16:9",
        "9:16 (Portrait)": "9:16",
        "21:9 (Ultrawide)": "21:9",
    };
    return aliases[value] || value;
}
function aspectOptionLabel(value) {
    const internal = aspectInternalValue(value);
    return isChineseLocale() ? (ASPECT_DISPLAY_VALUES[internal] || internal) : internal;
}
function normalizeAspectWidget(node) {
    const control = widget(node, "aspect");
    if (!control) return;
    const normalized = aspectInternalValue(control.value);
    if (!ASPECT_INTERNAL_VALUES.includes(normalized)) return;
    if (control.value !== normalized) control.value = normalized;
    control._ghH3AspectInternalValue = normalized;
}
function installAspectDisplay(node) {
    const control = widget(node, "aspect");
    if (!control || control._ghH3AspectDisplayWrapped || typeof control.draw !== "function") return;
    const originalDraw = control.draw;
    control.draw = function(ctx, nodeRef, width, y, height) {
        const savedValue = this.value;
        const savedValues = this.options?.values;
        const internalValues = ASPECT_INTERNAL_VALUES.slice();
        try {
            if (this.options && Array.isArray(savedValues)) {
                this.options.values = internalValues.map(aspectOptionLabel);
                this.value = aspectOptionLabel(savedValue);
            }
            return originalDraw.call(this, ctx, nodeRef, width, y, height);
        } finally {
            this.value = savedValue;
            if (this.options && Array.isArray(savedValues)) this.options.values = savedValues;
        }
    };
    control._ghH3AspectDisplayWrapped = true;
}
function drawAspectDisplayOverlay(node, ctx) {
    if (!isChineseLocale()) return;
    const control = widget(node, "aspect");
    if (!control || control.value !== "adaptive" || control === node.__ghH3ActiveWidget) return;
    const y = Number(control.y);
    const height = Number(control.height) || 30;
    const width = Number(node.size?.[0]) || 0;
    if (!Number.isFinite(y) || width <= 80) return;
    const left = 34;
    const right = width - 34;
    ctx.save();
    ctx.fillStyle = node.bgcolor || "#252a34";
    ctx.fillRect(left, y + 2, Math.max(0, right - left), Math.max(0, height - 4));
    ctx.fillStyle = "#e4e8ed";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.max(12, Math.round(height * 0.55))}px Arial,sans-serif`;
    ctx.fillText("自适应", (left + right) / 2, y + height / 2);
    ctx.restore();
}

function round32(v) { return Math.max(32, Math.floor(v / 32 + 0.5) * 32); }
function canvas(aspect, mp, adaptiveRatio = 16 / 9) {
    const area = Math.max(0.2, Number(mp) || 0.5) * 1024 * 1024;
    const ratio = ASPECTS[aspectInternalValue(aspect)] ?? adaptiveRatio;
    const width = round32(Math.sqrt(area * ratio));
    const height = ratio === 1 ? width : round32(Math.sqrt(area / ratio));
    return `${width}x${height}`;
}
function widget(node, name) { return node.widgets?.find(w => w.name === name); }
function setWidget(node, name, value) {
    const w = widget(node, name); if (!w) return;
    // Zero is a valid value for integer/float controls such as the optional
    // primary-audio ordinal. Only actual empty values should become (none).
    const next = value === null || value === undefined || value === "" ? "(none)" : value;
    if (w.value === next) return;
    w.value = next;
    w.callback?.call(w, w.value);
}
function cleanPrompt(value) { return value && value !== "(none)" ? value : ""; }
function normalizeIntWidget(node, name, fallback, minimum, maximum) {
    const w = widget(node, name); if (!w) return fallback;
    const parsed = Number(w.value);
    const value = Number.isFinite(parsed)
        ? Math.max(minimum, Math.min(maximum, Math.trunc(parsed)))
        : fallback;
    if (w.value !== value) {
        w.value = value;
        w.callback?.call(w, value);
    }
    return value;
}
function sanitizeHiddenInputs(node) {
    normalizeIntWidget(node, "drive_audio_ordinal", 1, 0, 6);
    const task = widget(node, "task_type");
    if (task && !["auto", "t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid"].includes(task.value)) {
        task.value = "auto";
    }
}
function hideWidget(w) {
    if (!w) return;
    w.hidden = true; w.options = w.options || {}; w.options.hidden = true;
    w.computeSize = () => [0, -4]; w.serialize = true;
}
function fileUrl(name) {
    if (!name || name === "(none)") return "";
    const parts = String(name).replaceAll("\\", "/").split("/").filter(Boolean);
    const filename = parts.pop() || "";
    const params = new URLSearchParams({ filename, type: "input", subfolder: parts.join("/") });
    const path = `/view?${params.toString()}`;
    return typeof api.apiURL === "function" ? api.apiURL(path) : path;
}
async function uploadFile(file, subfolder = "") {
    const body = new FormData(); body.append("image", file, file.name); body.append("type", "input");
    if (subfolder) body.append("subfolder", subfolder);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
    const result = await response.json();
    return [result.subfolder, result.name].filter(Boolean).join("/");
}


// ── shared drawing helpers for the canvas chips widget ─────────────────────
function roundRectPath(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
}

function createPanel(node) {
    const stateKey = "gh_h3_state";
    node.properties = node.properties || {};
    const savedState = (() => {
        try { return JSON.parse(node.properties?.[stateKey] || "{}"); } catch { return {}; }
    })();
    const stateWidget = widget(node, "gh_state_json");
    if (stateWidget?.value) {
        try { Object.assign(savedState, JSON.parse(stateWidget.value)); } catch {}
    }

    // During workflow restoration ComfyUI briefly exposes default widget values
    // before onConfigure has restored this panel's per-segment state. Never let
    // that transient state overwrite the serialized workflow.
    let restoringState = true;
    let restoreEpoch = 0;
    let segmentPersistTimer = null;
    let segmentLoras = null;
    let segmentLorasLoading = false;
    let adaptiveRatio = 16 / 9;

    const state = { mode: "multi_segment" };

    // ── segment data model (backend contract: gh/long_video.py parse_segments) ──
    const segmentClone = value => JSON.parse(JSON.stringify(value));
    const defaultSegment = (duration = 5) => ({
        duration: Math.max(2, Math.min(30, Math.round(Number(duration) || 5))),
        prompt: "", first_frame: "", last_frame: "", images: [], videos: [], audios: [], loras: [],
    });
    const normalizeSegment = (raw, index) => {
        const base = defaultSegment(raw?.duration);
        const segment = {
            duration: Math.max(2, Math.min(30, Math.round(Number(raw?.duration) || base.duration))),
            prompt: String(raw?.prompt || ""),
            first_frame: String(raw?.first_frame || ""),
            last_frame: String(raw?.last_frame || ""),
            images: (Array.isArray(raw?.images) ? raw.images : [])
                .map(item => String(item || "").trim()).filter(Boolean).slice(0, MAX_SEG_IMAGES),
            videos: (Array.isArray(raw?.videos) ? raw.videos : [])
                .map(item => (typeof item === "string" ? { name: item, muted: false } : {
                    name: String(item?.name || ""), muted: !!item?.muted,
                }))
                .filter(item => item.name).slice(0, MAX_SEG_VIDEOS),
            audios: (Array.isArray(raw?.audios) ? raw.audios : [])
                .map(item => (typeof item === "string" ? { name: item, trimStart: 0, trimEnd: null } : {
                    name: String(item?.name || ""),
                    trimStart: Math.max(0, Number(item?.trimStart) || 0),
                    trimEnd: item?.trimEnd == null ? null : Number(item.trimEnd),
                }))
                .filter(item => item.name).slice(0, MAX_SEG_AUDIOS),
            loras: (Array.isArray(raw?.loras) ? raw.loras : [])
                .map(item => (typeof item === "string" ? { name: item, strength_model: 1 } : {
                    name: String(item?.name || ""),
                    strength_model: Math.max(-10, Math.min(10, Number(item?.strength_model) || 1)),
                }))
                .filter(item => item.name && item.name !== "(none)" && item.name !== "None").slice(0, MAX_SEG_LORAS),
        };
        return segment;
    };
    let segments = (Array.isArray(savedState.segments) ? savedState.segments : [])
        .slice(0, MAX_SEGMENTS).map((raw, index) => normalizeSegment(raw, index));
    let segmentRange = {
        start: Math.max(1, Math.min(MAX_SEGMENTS, Math.round(Number(savedState?.segmentRange?.start) || 1))),
        end: Math.max(1, Math.min(MAX_SEGMENTS, Math.round(Number(savedState?.segmentRange?.end) || (segments.length || 1)))),
    };
    if (segmentRange.start > segmentRange.end) [segmentRange.start, segmentRange.end] = [segmentRange.end, segmentRange.start];
    if (segments.length) {
        segmentRange.start = Math.max(1, Math.min(segments.length, segmentRange.start));
        segmentRange.end = Math.max(segmentRange.start, Math.min(segments.length, segmentRange.end));
    }
    let latentContinue = savedState?.latentContinue !== false;
    let activeSegment = 0;

    const segmentsWidget = widget(node, "segments_json");
    function syncSegmentsWidget() {
        if (!segmentsWidget) return;
        segmentsWidget.value = JSON.stringify({
            version: 1,
            segments: segments.map(item => segmentClone(item)),
            range: { ...segmentRange },
            latent_continue: latentContinue,
        });
    }
    function scheduleSegmentSync() {
        clearTimeout(segmentPersistTimer);
        segmentPersistTimer = setTimeout(() => {
            syncSegmentsWidget();
            persistState();
        }, 250);
    }
    function ensureSegments() {
        // The eight shot blocks are fixed: pad the loaded state to the full
        // count so every block always has a backing segment object.
        while (segments.length < MAX_SEGMENTS) {
            const segment = defaultSegment(widget(node, "duration_seconds")?.value || 5);
            if (segments.length === 0) segment.prompt = cleanPrompt(widget(node, "prompt")?.value) || "";
            segments.push(segment);
        }
        segments.length = MAX_SEGMENTS;
        segmentRange.start = Math.max(1, Math.min(MAX_SEGMENTS, segmentRange.start));
        segmentRange.end = Math.max(segmentRange.start, Math.min(MAX_SEGMENTS, segmentRange.end));
    }
    const serializedState = () => JSON.stringify({
        segments: segments.map(item => segmentClone(item)),
        segmentRange: { ...segmentRange },
        latentContinue,
        activeSegment,
    });
    function persistState() {
        if (restoringState) return;
        const value = serializedState();
        node.properties[stateKey] = value;
        if (stateWidget && stateWidget.value !== value) stateWidget.value = value;
    }
    if (stateWidget) {
        stateWidget.serialize = true;
        stateWidget.options = stateWidget.options || {};
        stateWidget.options.serialize = true;
        stateWidget.serializeValue = () => {
            persistState();
            return stateWidget.value;
        };
    }

    // ── adaptive ratio for the "adaptive" aspect canvas size display ─────────
    const aspect = widget(node, "aspect");
    const mp = widget(node, "megapixels");
    function refreshAdaptiveRatio() {
        if (aspectInternalValue(aspect?.value) !== "adaptive") {
            adaptiveRatio = 16 / 9;
            return;
        }
        const name = segments[0]?.first_frame || segments[0]?.images?.[0] || "";
        if (!name) {
            adaptiveRatio = 16 / 9;
            return;
        }
        const image = new Image();
        image.onload = () => {
            if (image.naturalWidth && image.naturalHeight) {
                adaptiveRatio = image.naturalWidth / image.naturalHeight;
                redrawMediaWidgets();
            }
        };
        image.onerror = () => { adaptiveRatio = 16 / 9; };
        image.crossOrigin = "anonymous";
        image.src = fileUrl(name);
    }

    // ── segment tag numbering (must match the backend's dense numbering) ─────
    const segmentUnmutedVideos = seg => (seg.videos || []).filter(item => !item.muted).length;
    const segmentKeyframeCount = index => (segments[index]?.first_frame ? 1 : 0)
        + (segments[index]?.last_frame ? 1 : 0);
    function segmentTagText(seg, index, kind, refIndex) {
        if (kind === "first_frame") return "<Picture 1>";
        if (kind === "last_frame") return `<Picture ${seg.first_frame ? 2 : 1}>`;
        if (kind === "image") return `<Picture ${segmentKeyframeCount(index) + refIndex + 1}>`;
        if (kind === "video") return `<Video ${refIndex + 1}>`;
        if (kind === "audio") return `<Audio ${segmentUnmutedVideos(seg) + refIndex + 1}>`;
        return "";
    }

    // ── LoRA name source (ComfyUI object_info, fallback to this pack's route) ──
    function extractLoraNames(info) {
        if (!info || typeof info !== "object") return [];
        const className = Object.keys(info)[0];
        const spec = info?.[className]?.input;
        const field = spec?.required?.lora_name || spec?.optional?.lora_name;
        const options = Array.isArray(field?.[0]) ? field[0] : [];
        return options.filter(name => typeof name === "string" && name && name !== "None");
    }
    async function fetchLoraNames() {
        const names = new Set();
        // ComfyUI's own node definitions carry the authoritative loras list and
        // work even when this pack's Python side has not been reloaded yet.
        for (const className of ["LoraLoaderModelOnly", "LoraLoader"]) {
            try {
                const response = await api.fetchApi(`/object_info/${className}`, { cache: "no-store" });
                if (!response.ok) continue;
                extractLoraNames(await response.json()).forEach(name => names.add(name));
            } catch (error) {
                console.warn("[MiniMax H3 Integration] object_info LoRA lookup failed:", error);
            }
        }
        if (!names.size) {
            try {
                const response = await api.fetchApi(`${SEGMENT_LORA_ROUTE}?t=${Date.now()}`, { cache: "no-store" });
                if (response.ok) {
                    const data = await response.json();
                    (Array.isArray(data.loras) ? data.loras : []).forEach(name => names.add(name));
                }
            } catch (error) {
                console.warn("[MiniMax H3 Integration] /minimax_h3/loras failed:", error);
            }
        }
        return [...names].sort((a, b) => a.localeCompare(b));
    }
    function loadSegmentLoras() {
        if (segmentLorasLoading || segmentLoras) return;
        segmentLorasLoading = true;
        fetchLoraNames()
            .then(names => { segmentLoras = names; })
            .catch(error => {
                console.error("[MiniMax H3 Integration] LoRA list unavailable:", error);
                segmentLoras = [];
            })
            .finally(() => {
                segmentLorasLoading = false;
                // Always rebuild: the widget tree may have been restored while
                // this fetch was in flight, and the rebuild itself is safe at
                // any point (it renders whatever `segments` currently holds).
                rebuildSegmentWidgets();
            });
    }
    function refreshSegmentLoras() {
        segmentLoras = null;
        loadSegmentLoras();
    }

    // ── media uploads (same POST /upload/image endpoint as LoadImage) ────────
    function uploadSegment(seg, kind) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = kind === "video" ? "video/*" : kind === "audio" ? "audio/*" : "image/*";
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file) return;
            try {
                const name = await uploadFile(file);
                if (kind === "first_frame") seg.first_frame = name;
                else if (kind === "last_frame") seg.last_frame = name;
                else if (kind === "image" && seg.images.length < MAX_SEG_IMAGES) seg.images.push(name);
                else if (kind === "video" && seg.videos.length < MAX_SEG_VIDEOS) seg.videos.push({ name, muted: false });
                else if (kind === "audio" && seg.audios.length < MAX_SEG_AUDIOS) seg.audios.push({ name, trimStart: 0, trimEnd: null });
                syncSegmentState();
            } catch (error) {
                console.error("[MiniMax H3 Integration] segment upload failed:", error);
            }
        };
        input.click();
    }

    // ── prompt tag insertion into the native multiline widget ────────────────
    function insertSegmentPrompt(seg, index, text) {
        if (!text) return;
        const promptW = promptWidgetByIndex.get(index);
        const el = promptElByIndex.get(index);
        if (el && document.activeElement === el) {
            const start = el.selectionStart ?? el.value.length;
            const end = el.selectionEnd ?? start;
            const next = el.value.slice(0, start) + text + el.value.slice(end);
            seg.prompt = next;
            el.value = next;
            el.selectionStart = el.selectionEnd = start + text.length;
            if (promptW) promptW.value = next;
            scheduleSegmentSync();
            return;
        }
        seg.prompt = String(seg.prompt || "") + text;
        if (promptW?.setValue) promptW.setValue(seg.prompt);
        else if (promptW) promptW.value = seg.prompt;
        scheduleSegmentSync();
    }

    // ── native widget scaffolding ────────────────────────────────────────────
    // Every panel widget is a plain non-serializing litegraph widget: other
    // extensions can see and route them, while widgets_values (and therefore
    // old workflows/templates/API saves) stay byte-compatible with the DOM era.
    function addPanelWidget(type, name, value, callback, options = {}) {
        const w = node.addWidget(type, name, value, callback, options);
        w.serialize = false;
        w.options = { ...(w.options || {}), serialize: false };
        w._ghh3Panel = true;
        return w;
    }
    // Multiline prompts follow ComfyUI's own convention for prompt boxes: a
    // standard DOM widget carrying a `comfy-multiline-input` textarea (exactly
    // what the core prompt widgets render), so typing/IME/zoom behave like on
    // any native node. Canvas-drawn textareas are "Node 2.0 only" in the
    // current frontend, so this is the native path in both render modes.
    function addMultilineWidget(name, value, placeholder, onChange) {
        const el = document.createElement("textarea");
        el.className = "comfy-multiline-input";
        el.placeholder = placeholder || "";
        el.spellcheck = false;
        el.style.height = "110px";
        el.style.minHeight = "110px";
        el.style.resize = "none";
        el.style.boxSizing = "border-box";
        const w = node.addDOMWidget(name, "customtext", el, { serialize: false });
        w.serialize = false;
        w.options = { ...(w.options || {}), serialize: false };
        w._ghh3Panel = true;
        // The canvas layout must reserve the element's real height, otherwise
        // the following rows overlap the textarea.
        const promptHeight = 112;
        w.computeSize = () => [0, promptHeight];
        el.value = value || "";
        el.addEventListener("input", () => {
            w.value = el.value;
            onChange(el.value);
        });
        w.setValue = v => {
            w.value = String(v ?? "");
            if (el.value !== w.value) el.value = w.value;
        };
        return w;
    }

    const globalWidgets = {};
    function clampRangeIndex(value) {
        const parsed = Math.round(Number(value) || 1);
        return Math.max(1, Math.min(MAX_SEGMENTS, parsed));
    }
    function buildGlobalWidgets() {
        // The eight shot blocks are always present (4+4); the run range is the
        // switch that decides which blocks are active. Blocks outside the
        // range collapse to their dimmed header row.
        const countValues = Array.from({ length: MAX_SEGMENTS }, (_, i) => String(i + 1));
        globalWidgets.rangeStart = addPanelWidget("combo", `${t("Run range")} · ${t("Start")}`, String(segmentRange.start), value => {
            if (restoringState) return;
            segmentRange.start = clampRangeIndex(value);
            if (segmentRange.start > segmentRange.end) segmentRange.end = segmentRange.start;
            syncSegmentState();
            updateSegmentVisibility();
        }, { values: countValues });
        globalWidgets.rangeEnd = addPanelWidget("combo", `${t("Run range")} · ${t("End")}`, String(segmentRange.end), value => {
            if (restoringState) return;
            segmentRange.end = clampRangeIndex(value);
            if (segmentRange.end < segmentRange.start) segmentRange.start = segmentRange.end;
            syncSegmentState();
            updateSegmentVisibility();
        }, { values: countValues });
        globalWidgets.latent = addPanelWidget("toggle", t("Latent continuation"), latentContinue, value => {
            if (restoringState) return;
            latentContinue = !!value;
            syncSegmentState();
        });
        refreshRangeOptions();
    }
    function refreshRangeOptions() {
        const values = Array.from({ length: MAX_SEGMENTS }, (_, i) => String(i + 1));
        if (globalWidgets.rangeStart) globalWidgets.rangeStart.options = { ...(globalWidgets.rangeStart.options || {}), values };
        if (globalWidgets.rangeEnd) globalWidgets.rangeEnd.options = { ...(globalWidgets.rangeEnd.options || {}), values };
        if (globalWidgets.rangeStart) globalWidgets.rangeStart.value = String(segmentRange.start);
        if (globalWidgets.rangeEnd) globalWidgets.rangeEnd.value = String(segmentRange.end);
    }
    function redrawMediaWidgets() {
        for (const mediaW of mediaWidgetByIndex.values()) mediaW.triggerDraw?.();
        node.setDirtyCanvas?.(true, true);
    }
    function syncSegmentState() {
        syncSegmentsWidget();
        persistState();
        redrawMediaWidgets();
    }

    // ── per-segment mutations ────────────────────────────────────────────────
    // The eight shot blocks are fixed; no add/remove controls exist. Blocks
    // outside the run range collapse (updateSegmentVisibility) and keep their
    // data, so extending the range brings them back fully populated.
    function copyPreviousSegment(index) {
        const previous = segments[index - 1];
        if (!previous) return;
        segments[index] = normalizeSegment(segmentClone(previous), index);
        syncSegmentState();
        rebuildSegmentWidgets();
    }

    // ── canvas chips widget (one per segment) ────────────────────────────────
    // Drawn with LiteGraph primitives and routed through the standard widget
    // mouse pipeline, the same approach as rgthree's Power Lora Loader rows:
    // no DOM overlay, so hit-testing, screenshots and canvas interactions all
    // behave like on a fully native node.
    const measureCtx = document.createElement("canvas").getContext("2d");
    const CHIP_FONT = "10px Arial,sans-serif";
    const CHIP_H = 20;
    const CHIP_PAD = 8;
    const ROW_GAP = 4;
    const HEADER_H = 20;
    const FOOTER_H = 20;
    const NOTE_H = 16;
    const CHIP_COLORS = {
        chipBg: "#1a2938",
        chipBorder: "#35526a",
        chipDragBorder: "#0aa4d6",
        chipText: "#dbe8f1",
        chipRemove: "#c6d4de",
        addBg: "#0f1c27",
        addBorder: "#2f5c72",
        addText: "#08b4ed",
        headerText: "#e7f0f7",
        btnText: "#9db1c0",
        removeBtnText: "#d47d8b",
        badgeRunBg: "#134b3d",
        badgeRunText: "#8fe3c6",
        badgeSkipBg: "#26414f",
        badgeSkipText: "#9fc2cf",
        noteText: "#7f93a4",
        sizeText: "#0db5e8",
    };
    const thumbCache = new Map();
    function segmentThumb(name) {
        if (!name) return null;
        let entry = thumbCache.get(name);
        if (!entry) {
            const img = new Image();
            entry = { img, loaded: false };
            img.onload = () => { entry.loaded = true; redrawMediaWidgets(); };
            img.onerror = () => { entry.loaded = false; };
            img.src = fileUrl(name);
            thumbCache.set(name, entry);
        }
        return entry;
    }

    function makeMediaChipsWidget(seg, index) {
        const inRange = () => index + 1 >= segmentRange.start && index + 1 <= segmentRange.end;
        const isLast = () => index === segments.length - 1;
        const w = {
            name: `${t("Segment")} ${index + 1} · ${t("Reference media")}`,
            type: "ghh3_media",
            value: "",
            options: { serialize: false },
            serialize: false,
            _ghh3Panel: true,
            _ghh3MediaWidget: true,
            _originX: 0,
            last_y: 0,
        };
        const drag = {
            active: false, started: false, pending: null,
            downX: 0, downY: 0, list: null, fromIdx: -1, hoverIdx: -1,
        };
        let regions = [];

        function chipItems() {
            const items = [];
            if (seg.first_frame) items.push({ kind: "first_frame", label: t("First frame"), tagKind: "first_frame", value: seg.first_frame, thumb: true });
            else items.push({ kind: "add", label: `＋${t("First frame")}`, act: { type: "upload", arg: "first_frame" } });
            if (seg.last_frame) items.push({ kind: "last_frame", label: t("Last frame"), tagKind: "last_frame", value: seg.last_frame, thumb: true });
            else items.push({ kind: "add", label: `＋${t("Last frame")}`, act: { type: "upload", arg: "last_frame" } });
            (seg.images || []).forEach((name, i) => items.push({
                kind: "chip", label: `${t("Picture")} ${segmentKeyframeCount(index) + i + 1}`, tagKind: "image",
                value: name, thumb: true, draggable: true, list: "images", idx: i,
            }));
            items.push({ kind: "add", label: `＋${t("Picture")}`, act: { type: "upload", arg: "image" }, disabled: (seg.images || []).length >= MAX_SEG_IMAGES });
            (seg.videos || []).forEach((item, i) => items.push({
                kind: "video", label: `${t("Video")} ${i + 1}`, tagKind: "video",
                value: item.name, muted: !!item.muted, draggable: true, list: "videos", idx: i,
            }));
            items.push({ kind: "add", label: `＋${t("Video")}`, act: { type: "upload", arg: "video" }, disabled: (seg.videos || []).length >= MAX_SEG_VIDEOS });
            (seg.audios || []).forEach((item, i) => items.push({
                kind: "chip", label: `${t("Audio")} ${segmentUnmutedVideos(seg) + i + 1}`, tagKind: "audio",
                value: item.name, draggable: true, list: "audios", idx: i,
            }));
            items.push({ kind: "add", label: `＋${t("Audio")}`, act: { type: "upload", arg: "audio" }, disabled: (seg.audios || []).length >= MAX_SEG_AUDIOS });
            return items;
        }

        function measureChip(item) {
            measureCtx.font = CHIP_FONT;
            const textWidth = measureCtx.measureText(item.label).width;
            let extra = 6;
            if (item.thumb) extra += 18;
            if (item.kind === "video") extra += 15;
            if (item.kind !== "add") extra += 13;
            return Math.ceil(10 + textWidth + extra);
        }
        function layoutRows(width) {
            const items = chipItems();
            const rows = [];
            let row = [];
            let x = CHIP_PAD;
            const limit = Math.max(140, width - CHIP_PAD);
            for (const item of items) {
                const chipW = measureChip(item);
                if (row.length && x + chipW > limit) {
                    rows.push(row);
                    row = [];
                    x = CHIP_PAD;
                }
                item._w = chipW;
                item._x = x;
                row.push(item);
                x += chipW + ROW_GAP;
            }
            if (row.length) rows.push(row);
            return rows;
        }
        w.computeSize = function(width) {
            const rows = layoutRows(Number(width) || WIDTH);
            const note = isLast() ? NOTE_H + 2 : 0;
            return [width, 6 + HEADER_H + rows.length * (CHIP_H + ROW_GAP) + FOOTER_H + note];
        };

        function localMouse(e, pos) {
            let x = Number(pos?.[0]) || 0;
            let y = Number(pos?.[1]) || 0;
            const ox = Number(e?.offsetX), oy = Number(e?.offsetY);
            const sameAsOffsets = Number.isFinite(ox) && Number.isFinite(oy)
                && Math.abs(ox - x) < 0.75 && Math.abs(oy - y) < 0.75;
            if (!sameAsOffsets) {
                // Node-relative coordinates (classic renderer or the pointer-up
                // dispatch path); convert into this widget's local space.
                x -= w._originX || 0;
                y -= w.last_y || 0;
            }
            return [x, y];
        }
        function hitRegion(x, y) {
            // Forward iteration: sub-regions (mute / strength / remove) are
            // pushed before the chip-wide insert region and must win.
            for (const region of regions) {
                if (x >= region.x && x <= region.x + region.w && y >= region.y && y <= region.y + region.h) return region;
            }
            return null;
        }
        function drawTextButton(ctx, x, y, bw, bh, label, color) {
            ctx.fillStyle = CHIP_COLORS.addBg;
            ctx.strokeStyle = CHIP_COLORS.chipBorder;
            ctx.lineWidth = 1;
            roundRectPath(ctx, x, y, bw, bh, 4);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = color || CHIP_COLORS.btnText;
            ctx.font = CHIP_FONT;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, x + bw / 2, y + bh / 2 + 0.5);
        }
        function drawChip(ctx, item, cy) {
            const x = item._x;
            const cw = item._w;
            if (item.kind === "add") {
                ctx.globalAlpha = item.disabled ? 0.4 : 1;
                ctx.fillStyle = CHIP_COLORS.addBg;
                ctx.strokeStyle = CHIP_COLORS.addBorder;
                ctx.setLineDash([3, 3]);
                roundRectPath(ctx, x, cy, cw, CHIP_H, 4);
                ctx.fill();
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = CHIP_COLORS.addText;
                ctx.font = CHIP_FONT;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(item.label, x + 6, cy + CHIP_H / 2 + 0.5);
                ctx.globalAlpha = 1;
                if (!item.disabled) regions.push({ x, y: cy, w: cw, h: CHIP_H, act: "upload", arg: item.act.arg });
                return;
            }
            const dragging = drag.active && drag.started && drag.list === item.list && drag.fromIdx === item.idx;
            const hoverTarget = drag.active && drag.started && drag.list === item.list && drag.hoverIdx === item.idx && drag.fromIdx !== item.idx;
            ctx.globalAlpha = (!inRange() ? 0.45 : 1) * (dragging ? 0.55 : 1);
            ctx.fillStyle = CHIP_COLORS.chipBg;
            ctx.strokeStyle = hoverTarget ? CHIP_COLORS.chipDragBorder : CHIP_COLORS.chipBorder;
            ctx.lineWidth = hoverTarget ? 1.5 : 1;
            roundRectPath(ctx, x, cy, cw, CHIP_H, 4);
            ctx.fill();
            ctx.stroke();
            let tx = x + 5;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            if (item.thumb) {
                const entry = segmentThumb(item.value);
                if (entry?.loaded && entry.img.naturalWidth) {
                    const img = entry.img;
                    const size = CHIP_H - 6;
                    const scale = Math.min(size / img.naturalWidth, size / img.naturalHeight);
                    const dw = Math.max(1, img.naturalWidth * scale);
                    const dh = Math.max(1, img.naturalHeight * scale);
                    ctx.save();
                    roundRectPath(ctx, tx, cy + 3, size, size, 3);
                    ctx.clip();
                    ctx.drawImage(img, tx + (size - dw) / 2, cy + 3 + (size - dh) / 2, dw, dh);
                    ctx.restore();
                } else {
                    ctx.strokeStyle = CHIP_COLORS.chipBorder;
                    roundRectPath(ctx, tx, cy + 3, CHIP_H - 6, CHIP_H - 6, 3);
                    ctx.stroke();
                }
                tx += 18;
            }
            ctx.fillStyle = CHIP_COLORS.chipText;
            ctx.font = CHIP_FONT;
            ctx.fillText(item.label, tx, cy + CHIP_H / 2 + 0.5);
            measureCtx.font = CHIP_FONT;
            tx += measureCtx.measureText(item.label).width + 3;
            if (item.kind === "video") {
                // Mute toggle: muted videos contribute no reference audio.
                const muteX = x + cw - 12 - 15;
                ctx.fillStyle = item.muted ? "#8fe3c6" : CHIP_COLORS.chipText;
                ctx.font = "9px Arial,sans-serif";
                ctx.fillText(item.muted ? "🔇" : "🔊", muteX + 2, cy + CHIP_H / 2 + 0.5);
                regions.push({ x: muteX, y: cy + 1, w: 15, h: CHIP_H - 2, act: "mute", idx: item.idx });
            }
            // remove (×) occupies the right edge of every data chip
            const removeX = x + cw - 12;
            ctx.fillStyle = CHIP_COLORS.chipRemove;
            ctx.font = "10px Arial,sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("×", removeX + 6, cy + CHIP_H / 2 + 0.5);
            regions.push({ x: removeX, y: cy + 1, w: 12, h: CHIP_H - 2, act: "remove", list: item.list, idx: item.idx, kind: item.kind });
            regions.push({
                x, y: cy, w: cw, h: CHIP_H, act: "insert", tagKind: item.tagKind, idx: item.idx,
                list: item.list, draggable: !!item.draggable,
            });
            ctx.globalAlpha = 1;
        }

        w.draw = function(ctx, nodeRef, width, y, height) {
            // WidgetLegacy (vue nodes) renders into a per-widget canvas with y=1;
            // the classic renderer passes the widget's stacked y offset.
            const vueLike = Number(y) <= 2;
            const originX = vueLike ? 0 : 6;
            const originY = vueLike ? 0 : Number(y) || 0;
            this.last_y = Number(y) || 0;
            this._originX = originX;
            const widgetWidth = Number(width) || WIDTH;
            const rows = layoutRows(widgetWidth);
            regions = [];
            this._hitRegions = regions;
            ctx.save();
            ctx.translate(originX, originY);
            ctx.textBaseline = "middle";
            // header row: segment title + run badge (+ canvas size on segment 1)
            const dim = inRange() ? 1 : 0.6;
            ctx.globalAlpha = dim;
            const titleText = `${t("Segment")} ${index + 1}`;
            ctx.font = "bold 11px Arial,sans-serif";
            ctx.fillStyle = CHIP_COLORS.headerText;
            ctx.textAlign = "left";
            ctx.fillText(titleText, 0, HEADER_H / 2 + 1);
            const titleW = ctx.measureText(titleText).width;
            ctx.font = "9px Arial,sans-serif";
            const badgeText = inRange() ? t("Run") : t("Skipped");
            const badgeW = Math.ceil(ctx.measureText(badgeText).width) + 8;
            ctx.fillStyle = inRange() ? CHIP_COLORS.badgeRunBg : CHIP_COLORS.badgeSkipBg;
            roundRectPath(ctx, titleW + 8, 2, badgeW, HEADER_H - 4, 4);
            ctx.fill();
            ctx.fillStyle = inRange() ? CHIP_COLORS.badgeRunText : CHIP_COLORS.badgeSkipText;
            ctx.textAlign = "center";
            ctx.fillText(badgeText, titleW + 8 + badgeW / 2, HEADER_H / 2 + 1);
            // header buttons, right-aligned: [复制上一段]
            ctx.font = CHIP_FONT;
            const buttons = [];
            if (index > 0) buttons.push({ label: t("Copy previous"), act: "copy" });
            let bx = widgetWidth - 8;
            for (let i = buttons.length - 1; i >= 0; i -= 1) {
                const button = buttons[i];
                const bw = Math.ceil(ctx.measureText(button.label).width) + 12;
                bx -= bw;
                drawTextButton(ctx, bx, 1, bw, HEADER_H - 2, button.label, CHIP_COLORS.btnText);
                regions.push({ x: bx, y: 1, w: bw, h: HEADER_H - 2, act: button.act });
                bx -= 4;
            }
            if (index === 0) {
                ctx.fillStyle = CHIP_COLORS.sizeText;
                ctx.font = "10px Arial,sans-serif";
                ctx.textAlign = "right";
                ctx.fillText(canvas(aspectInternalValue(aspect?.value), mp?.value, adaptiveRatio), Math.max(0, bx - 8), HEADER_H / 2 + 1);
            }
            // chip rows
            let cy = HEADER_H + ROW_GAP;
            for (const rowItems of rows) {
                for (const item of rowItems) drawChip(ctx, item, cy);
                cy += CHIP_H + ROW_GAP;
            }
            // footer: +LoRA / refresh — then the latent hint on the last segment
            ctx.font = CHIP_FONT;
            const footer = [];
            if (seg.loras.length < MAX_SEG_LORAS) footer.push({ label: t("Add LoRA"), act: "add-lora" });
            footer.push({ label: "↻", act: "refresh-loras" });
            let fx = CHIP_PAD;
            for (const item of footer) {
                const bw = Math.ceil(ctx.measureText(item.label).width) + 12;
                drawTextButton(ctx, fx, cy, bw, CHIP_H - 2, item.label);
                regions.push({ x: fx, y: cy, w: bw, h: CHIP_H - 2, act: item.act });
                fx += bw + ROW_GAP;
            }
            if (isLast()) {
                const note = t(latentContinue
                    ? "Latent continuation is ON: the previous segment's last 7 latent tokens are carried into the next segment; no first-frame keyframe and no pixel overlap are used."
                    : "Latent continuation is OFF: segments are joined with a hard cut and no seam blending.");
                ctx.fillStyle = CHIP_COLORS.noteText;
                ctx.font = "9px Arial,sans-serif";
                ctx.textAlign = "left";
                let clipped = note;
                while (clipped.length > 4 && ctx.measureText(clipped).width > widgetWidth - CHIP_PAD * 2) clipped = clipped.slice(0, -1);
                ctx.fillText(clipped, 0, cy + CHIP_H + NOTE_H / 2);
            }
            ctx.restore();
            ctx.globalAlpha = 1;
        };

        function clearDrag() {
            drag.active = false;
            drag.started = false;
            drag.pending = null;
            drag.list = null;
            drag.fromIdx = -1;
            drag.hoverIdx = -1;
        }
        function commitReorder(listName, fromIdx, hoverIdx) {
            if (!listName || hoverIdx < 0 || fromIdx < 0 || fromIdx === hoverIdx) return;
            const items = seg[listName];
            if (!Array.isArray(items) || fromIdx >= items.length) return;
            const to = Math.max(0, Math.min(items.length - 1, hoverIdx));
            const [moved] = items.splice(fromIdx, 1);
            items.splice(to, 0, moved);
            syncSegmentState();
        }
        function runAction(region) {
            switch (region.act) {
                case "upload": uploadSegment(seg, region.arg); break;
                case "insert": insertSegmentPrompt(seg, index, segmentTagText(seg, index, region.tagKind, region.idx)); break;
                case "mute": {
                    const item = seg.videos?.[region.idx];
                    if (item) {
                        item.muted = !item.muted;
                        syncSegmentState();
                    }
                    break;
                }
                case "remove": {
                    if (region.kind === "first_frame") seg.first_frame = "";
                    else if (region.kind === "last_frame") seg.last_frame = "";
                    else if (Array.isArray(seg[region.list])) seg[region.list].splice(region.idx, 1);
                    syncSegmentState();
                    break;
                }
                case "copy": copyPreviousSegment(index); break;
                case "add-lora": {
                    if (seg.loras.length < MAX_SEG_LORAS) {
                        seg.loras.push({ name: "", strength_model: 1 });
                        syncSegmentState();
                        rebuildSegmentWidgets();
                    }
                    break;
                }
                case "refresh-loras": refreshSegmentLoras(); break;
                default: break;
            }
        }
        w.mouse = function(e, pos, nodeRef) {
            const type = String(e?.type || "");
            const [lx, ly] = localMouse(e, pos);
            if (type === "pointerdown") {
                const region = hitRegion(lx, ly);
                drag.pending = region || null;
                drag.downX = lx;
                drag.downY = ly;
                drag.active = !!(region && region.draggable);
                drag.started = false;
                drag.list = drag.active ? region.list : null;
                drag.fromIdx = drag.active ? region.idx : -1;
                drag.hoverIdx = drag.fromIdx;
                return true;
            }
            if (type === "pointermove") {
                if (drag.active) {
                    if (!drag.started && Math.abs(lx - drag.downX) + Math.abs(ly - drag.downY) > 6) drag.started = true;
                    if (drag.started) {
                        const target = hitRegion(lx, ly);
                        drag.hoverIdx = target && target.list === drag.list ? target.idx : -1;
                        return true;
                    }
                }
                return false;
            }
            if (type === "pointerup" || type === "pointercancel") {
                const pending = drag.pending;
                const moved = Math.abs(lx - drag.downX) + Math.abs(ly - drag.downY);
                const wasDragging = drag.active && drag.started;
                // Capture the drag target before clearDrag wipes the state.
                const reorder = { list: drag.list, fromIdx: drag.fromIdx, hoverIdx: drag.hoverIdx };
                clearDrag();
                if (wasDragging) {
                    commitReorder(reorder.list, reorder.fromIdx, reorder.hoverIdx);
                    return true;
                }
                if (pending && moved <= 6) runAction(pending);
                return true;
            }
            return false;
        };
        return w;
    }

    // ── segment widget tree ──────────────────────────────────────────────────
    const segmentWidgets = [];
    const promptWidgetByIndex = new Map();
    const promptElByIndex = new Map();
    const mediaWidgetByIndex = new Map();
    const loraOptions = entry => {
        const names = Array.isArray(segmentLoras) ? segmentLoras : [];
        const values = ["None", ...names];
        if (entry?.name && !values.includes(entry.name)) values.push(entry.name);
        return values;
    };
    function clearSegmentWidgets() {
        for (const w of segmentWidgets) {
            // DOM-backed widgets (segment prompts) must drop their overlay
            // element explicitly, or removed rows leave ghost textareas behind.
            if (w.element?.parentNode) w.element.parentNode.removeChild(w.element);
            w.onRemove?.();
            const at = node.widgets.indexOf(w);
            if (at >= 0) node.widgets.splice(at, 1);
        }
        segmentWidgets.length = 0;
        segmentRowWidgets.clear();
        promptWidgetByIndex.clear();
        promptElByIndex.clear();
        mediaWidgetByIndex.clear();
    }
    // One canvas-drawn non-interactive label row separating the two 4-shot
    // groups (the 4+4 arrangement).
    function makeGroupSeparator(label) {
        const w = {
            name: `ghh3_group_${label}`,
            type: "ghh3_label",
            value: "",
            options: { serialize: false },
            serialize: false,
            _ghh3Panel: true,
        };
        w.computeSize = () => [0, 18];
        w.draw = function(ctx, nodeRef, width, y) {
            const vueLike = Number(y) <= 2;
            const originX = vueLike ? 0 : 6;
            const originY = vueLike ? 0 : Number(y) || 0;
            this.last_y = Number(y) || 0;
            ctx.save();
            ctx.translate(originX, originY);
            ctx.globalAlpha = 0.9;
            ctx.strokeStyle = "#2a4353";
            ctx.beginPath();
            ctx.moveTo(0, 15);
            ctx.lineTo(Math.max(0, (Number(width) || WIDTH) - 8), 15);
            ctx.stroke();
            ctx.fillStyle = "#7f93a4";
            ctx.font = "bold 10px Arial,sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(label, 0, 8);
            ctx.restore();
            ctx.globalAlpha = 1;
        };
        return w;
    }

    // Segments outside the run range collapse to their dimmed chips header:
    // their editable rows are hidden but the data stays intact, so extending
    // the range brings the block back exactly as it was left.
    const segmentRowWidgets = new Map();
    function applyRowHidden(w, hidden) {
        if (!w || w.hidden === hidden) return;
        w.hidden = hidden;
        w.options = { ...(w.options || {}), hidden };
    }
    function segmentInRange(index) {
        return index + 1 >= segmentRange.start && index + 1 <= segmentRange.end;
    }
    function updateSegmentVisibility() {
        for (const [index, rows] of segmentRowWidgets) {
            const hidden = !segmentInRange(index);
            for (const w of rows) applyRowHidden(w, hidden);
        }
        if (node.graph) fitNodeHeight();
    }

    function buildSegmentWidgets() {
        segmentRowWidgets.clear();
        segments.forEach((seg, index) => {
            if (index === 0 || index === 4) {
                const separator = makeGroupSeparator(index === 0 ? t("Shots 1-4") : t("Shots 5-8"));
                node.addCustomWidget(separator);
                segmentWidgets.push(separator);
            }
            const labelPrefix = `${t("Segment")} ${index + 1}`;
            const rows = [];
            const durationW = addPanelWidget("number", `${labelPrefix} · ${t("Duration (seconds)")}`, seg.duration, value => {
                seg.duration = Math.max(2, Math.min(30, Math.round(Number(value) || 5)));
                durationW.value = seg.duration;
                scheduleSegmentSync();
            }, { min: 2, max: 30, step: 1, precision: 0 });
            rows.push(durationW);
            const promptW = addMultilineWidget(`${labelPrefix} · ${t("Prompt")}`, seg.prompt, t("Segment prompt: click a reference chip to insert its tag"), value => {
                seg.prompt = String(value ?? "");
                scheduleSegmentSync();
            });
            promptWidgetByIndex.set(index, promptW);
            promptElByIndex.set(index, promptW.element);
            rows.push(promptW);
            const mediaW = makeMediaChipsWidget(seg, index);
            node.addCustomWidget(mediaW);
            mediaWidgetByIndex.set(index, mediaW);
            segmentWidgets.push(durationW, promptW, mediaW);
            // Connector rows: the shot's image/video/audio link points live
            // inside the block. The matching input slots (conn_image_i etc.)
            // are bound to these rows, so their sockets render right here.
            // They stay visible even when the block is collapsed by the run
            // range: wiring points are permanent, hidden rows would leave
            // their sockets floating at stale positions.
            for (const { group, label } of CONN_SLOT_GROUPS) {
                const row = makeConnRowWidget(group, index, label);
                node.addCustomWidget(row);
                segmentWidgets.push(row);
            }
            seg.loras.forEach((entry, slot) => {
                const comboW = addPanelWidget("combo", `LoRA ${slot + 1}`, entry.name || "None", value => {
                    if (restoringState) return;
                    const name = !value || value === "None" ? "" : String(value);
                    if (!name) {
                        seg.loras.splice(slot, 1);
                        syncSegmentState();
                        rebuildSegmentWidgets();
                    } else {
                        entry.name = name;
                        syncSegmentState();
                    }
                }, { values: loraOptions(entry) });
                const strengthW = addPanelWidget("number", `LoRA ${slot + 1} · ${t("Model strength")}`, entry.strength_model ?? 1, value => {
                    entry.strength_model = Math.max(-10, Math.min(10, Number(value) || 0));
                    scheduleSegmentSync();
                }, { min: -10, max: 10, step: 0.05, precision: 2 });
                rows.push(comboW, strengthW);
                segmentWidgets.push(comboW, strengthW);
            });
            segmentRowWidgets.set(index, rows);
        });
        updateSegmentVisibility();
    }
    // ── per-segment connection slots (fixed pool of 8 per kind) ──────────────
    // The backend declares conn_image/conn_video/conn_audio Autogrow groups
    // with a MAX_SEGMENTS pool, matching the fixed 4+4 shot blocks: one input
    // slot per shot group, materialized once. In "per segment (in order)"
    // mode slot N feeds shot N.
    const CONN_SLOT_GROUPS = [
        { group: "conn_image", type: "IMAGE", label: t("Picture") },
        { group: "conn_video", type: "VIDEO", label: t("Video") },
        { group: "conn_audio", type: "AUDIO", label: t("Audio") },
    ];
    // A thin row inside a shot block that hosts that shot's link sockets:
    // the input slots carry a {name} widget stub pointing at this row, and
    // the frontend's widget-input-slot layout renders their sockets here.
    function makeConnRowWidget(group, index, label) {
        const name = `${group}_row_${index}`;
        const w = {
            name,
            type: "ghh3_connrow",
            value: "",
            options: { serialize: false },
            serialize: false,
            _ghh3Panel: true,
        };
        w.computeSize = () => [0, 22];
        w.draw = function(ctx, nodeRef, width, y) {
            const vueLike = Number(y) <= 2;
            const originX = vueLike ? 0 : 6;
            const originY = vueLike ? 0 : Number(y) || 0;
            this.last_y = Number(y) || 0;
            ctx.save();
            ctx.translate(originX, originY);
            ctx.globalAlpha = 0.55;
            ctx.fillStyle = "#7f93a4";
            ctx.font = "9px Arial,sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(`└ ${label}（连线）`, 26, 12);
            ctx.restore();
            ctx.globalAlpha = 1;
        };
        return w;
    }
    function syncConnSlots() {
        if (!node.graph || !Array.isArray(node.inputs)) return;
        for (const { group, type, label } of CONN_SLOT_GROUPS) {
            for (let i = 0; i < MAX_SEGMENTS; i += 1) {
                const name = `${group}_${i}`;
                const existing = node.inputs.find(inp => inp.name === name);
                if (existing) {
                    // Ensure the row binding exists even on inputs restored
                    // from older workflows.
                    if (!existing.widget?.name) existing.widget = { name: `${group}_row_${i}` };
                    continue;
                }
                let insertAt = node.inputs.length;
                for (let j = node.inputs.length - 1; j >= 0; j -= 1) {
                    const inp = node.inputs[j];
                    if (typeof inp?.name === "string" && inp.name.startsWith(group + "_")) { insertAt = j + 1; break; }
                }
                node.inputs.splice(insertAt, 0, {
                    name,
                    label: `${label} ${i + 1}（连线）`,
                    type,
                    optional: true,
                    widget: { name: `${group}_row_${i}` },
                });
            }
        }
        node.setDirtyCanvas?.(true, true);
    }

    // Fit the node body to its widgets in BOTH directions — grow after an add,
    // shrink after a removal — so the body never keeps large blank areas.
    // node.computeSize() ignores the live layout in the current frontend, so
    // measure the geometry arrange() just computed instead: the bottom edge of
    // the deepest visible widget row plus a small padding.
    function fitNodeHeight() {
        if (!node.graph) return;
        node.arrange?.();
        let bottom = 0;
        for (const w of node.widgets || []) {
            if (w.hidden) continue;
            const y = Number.isFinite(w.y) ? w.y : 0;
            const h = Number(w.computedHeight);
            bottom = Math.max(bottom, y + (Number.isFinite(h) && h > 0 ? h : 30));
        }
        const target = Math.ceil(bottom + 10);
        if (Number.isFinite(target) && target > 120 && Math.abs((node.size?.[1] ?? 0) - target) > 2) {
            node.setSize([node.size[0], target]);
        }
    }

    function rebuildSegmentWidgets() {
        clearSegmentWidgets();
        buildSegmentWidgets();
        // The current frontend assigns widget geometry during layout passes
        // (node.arrange), not per frame — run it after every structural change
        // or newly added rows stay at y=0 and DOM widgets misplace. arrange
        // asserts a graph reference, so skip it while the node is still being
        // constructed (the first draw arranges it once added to the graph).
        if (node.graph) node.arrange?.();
        syncConnSlots();
        fitNodeHeight();
        node.setDirtyCanvas?.(true, true);
    }

    // ── aspect / megapixels widget integration (canvas-drawn, kept native) ───
    normalizeAspectWidget(node);
    installAspectDisplay(node);
    if (aspect) aspect.callback = ((old) => function(v) {
        const internalValue = ASPECT_INTERNAL_VALUES.includes(v) ? v : aspectInternalValue(v);
        old?.call(this, internalValue);
        this.value = internalValue;
        this._ghH3AspectInternalValue = internalValue;
        refreshAdaptiveRatio();
        redrawMediaWidgets();
    })(aspect.callback);
    if (mp) mp.callback = ((old) => function(v) { old?.call(this, v); redrawMediaWidgets(); })(mp.callback);

    // ── legacy schema widgets stay hidden; their values are submitted as before ──
    const legacyHiddenNames = [
        "main_mode", "prompt", "task_type", ...mediaSlots,
        "audio_denoise_strength", "drive_audio_ordinal", "strict_prompt_tags",
        "gh_state_json", "refmods_json", "segments_json",
    ];
    function applyHiddenWidgets() {
        for (const name of legacyHiddenNames) hideWidget(widget(node, name));
        sanitizeHiddenInputs(node);
    }
    applyHiddenWidgets();

    // ── locale switches re-create the widget tree with translated labels ─────
    function rebuildAllWidgets() {
        for (const key of Object.keys(globalWidgets)) {
            const w = globalWidgets[key];
            const at = node.widgets.indexOf(w);
            if (at >= 0) node.widgets.splice(at, 1);
            delete globalWidgets[key];
        }
        clearSegmentWidgets();
        buildGlobalWidgets();
        rebuildSegmentWidgets();
    }
    const onLocaleStorage = () => rebuildAllWidgets();
    window.addEventListener("storage", onLocaleStorage);

    // ── serialization: segments_json stays the authoritative backend payload ──
    const oldSerialize = node.onSerialize;
    node.onSerialize = function(...args) {
        sanitizeHiddenInputs(this);
        syncSegmentsWidget();
        persistState();
        const result = oldSerialize?.apply(this, args);
        const serializedNode = args[0];
        if (serializedNode) {
            serializedNode.properties = serializedNode.properties || {};
            serializedNode.properties[stateKey] = node.properties[stateKey];
        }
        return result;
    };

    // ── restore: rebuild the widget tree from segments_json ──────────────────
    function restoreFrom(configuredState) {
        let restored = {};
        // Priority: the configured snapshot (onConfigure args) and the node
        // properties applied by configure() are authoritative — they come from
        // the saved workflow. The segments_json widget value can be stale: a
        // freshly created node seeds it with a default payload before
        // configure() runs, so it must be consulted last.
        const sources = [configuredState, node.properties?.[stateKey], segmentsWidget?.value, stateWidget?.value];
        for (const source of sources) {
            if (!source) continue;
            try {
                const parsed = JSON.parse(source);
                if (parsed && (Array.isArray(parsed.segments) || parsed.range || parsed.segmentRange)) {
                    restored = parsed;
                    break;
                }
            } catch {}
        }
        const restoredSegments = Array.isArray(restored.segments) ? restored.segments
            : (Array.isArray(savedState.segments) ? savedState.segments : null);
        if (restoredSegments?.length) {
            segments = restoredSegments.slice(0, MAX_SEGMENTS).map((raw, idx) => normalizeSegment(raw, idx));
        }
        // Always pad to the fixed block count: the eight shot groups exist
        // whether or not the saved workflow used them.
        ensureSegments();
        const restoredRange = restored.range || restored.segmentRange;
        segmentRange = {
            start: Math.max(1, Math.min(MAX_SEGMENTS, Math.round(Number(restoredRange?.start) || 1))),
            end: Math.max(1, Math.min(MAX_SEGMENTS, Math.round(Number(restoredRange?.end) || MAX_SEGMENTS))),
        };
        if (segmentRange.start > segmentRange.end) [segmentRange.start, segmentRange.end] = [segmentRange.end, segmentRange.start];
        const restoredLatent = restored.latentContinue !== undefined ? restored.latentContinue : restored.latent_continue;
        latentContinue = restoredLatent !== undefined ? restoredLatent !== false : savedState.latentContinue !== false;
        activeSegment = Math.max(0, Math.min(segments.length - 1, Math.round(Number(restored.activeSegment ?? savedState.activeSegment) || 0)));
        if (globalWidgets.latent) globalWidgets.latent.value = latentContinue;
        refreshRangeOptions();
        refreshAdaptiveRatio();
        rebuildSegmentWidgets();
        syncSegmentsWidget();
    }
    for (const hook of ["onAdded", "onConfigure", "onGraphConfigured"]) {
        const old = node[hook];
        node[hook] = function(...args) {
            const configuredState = hook === "onConfigure" ? args[0]?.properties?.[stateKey] : null;
            const result = old?.apply(this, args);
            applyHiddenWidgets();
            if (hook === "onAdded") {
                // The node just entered the graph: run one layout pass so the
                // panel rows get their stacked positions immediately (rAF is
                // throttled in background tabs, so this runs synchronously),
                // and materialize the per-shot connection slots.
                if (this.graph) {
                    this.arrange?.();
                    syncConnSlots();
                }
                return result;
            }
            restoringState = true;
            const currentRestoreEpoch = ++restoreEpoch;
            // setTimeout, not requestAnimationFrame: rAF never fires while the
            // tab is hidden or occluded, which would leave the panel unrestored
            // until the tab becomes visible again.
            setTimeout(() => {
                if (currentRestoreEpoch !== restoreEpoch) return;
                restoreFrom(configuredState);
                if (currentRestoreEpoch === restoreEpoch) {
                    restoringState = false;
                    persistState();
                }
            }, 0);
            return result;
        };
    }

    const oldDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(...args) {
        const result = oldDrawForeground?.apply(this, args);
        drawAspectDisplayOverlay(this, args[0]);
        return result;
    };

    const oldRemoved = node.onRemoved;
    node.onRemoved = function(...args) {
        clearTimeout(segmentPersistTimer);
        window.removeEventListener("storage", onLocaleStorage);
        return oldRemoved?.apply(this, args);
    };

    // ── initial build ────────────────────────────────────────────────────────
    setWidget(node, "main_mode", state.mode);
    buildGlobalWidgets();
    ensureSegments();
    rebuildSegmentWidgets();
    syncSegmentsWidget();
    loadSegmentLoras();
    refreshAdaptiveRatio();
    if (!Number(node.size?.[0]) || node.size[0] < WIDTH) {
        node.size[0] = WIDTH;
        node.setDirtyCanvas?.(true, true);
    }
    // Same reasoning as the restore hooks: a timer fires even in hidden tabs,
    // so the initial persist cannot be starved by rAF throttling.
    setTimeout(() => {
        if (restoreEpoch !== 0) return; // a workflow configure is already queued
        restoringState = false;
        persistState();
    }, 0);
    return true;
}

// ── MiniMaxH3分镜 (LKH3Shot) ─────────────────────────────────────────────────
// Eight fixed shot blocks; global model/canvas settings arrive from the
// MiniMaxH3模型设置 node through the settings link, so this node has no
// settings widgets of its own. Run range + latent continuation are plain
// schema widgets on the node and act as the per-block on/off switch: blocks
// outside the range collapse to their dimmed chips header (data preserved).
// ── MiniMaxH3分镜 (LKH3Shot) ─────────────────────────────────────────────────
// One compact shot node: duration/prompt/media live on the schema (sockets for
// media), while per-shot LoRA rows are managed here through the
// hidden shot_json widget.
function createShotPanel(node) {
    const shotJsonW = widget(node, "shot_json");
    // The frontend's widget-store restore can lose hidden-widget payloads on
    // round trips, so the panel also mirrors its state into node.properties
    // (which always survive serialization) and prefers them on restore.
    const stateKey = "gh_h3_shot_state";
    node.properties = node.properties || {};

    let loras = [];
    // audio_mode 曾是裸标识符：非严格模式下首次赋值静默落成 window 全局，
    // 但全局尚不存在时的首次"读取"（syncShotJson 序列化）就是
    // ReferenceError——看门狗在加载窗口触发 shot_json 回调后踩中过一次。
    // 正式声明，彻底排掉这颗雷。
    let audioMode = "";
    let loras_fetched = null;
    let loras_fetching = false;
    let restoring = true;
    let restoreEpoch = 0;
    let syncTimer = null;
    const shotWidgets = [];
    let lastRowCount = 0;

    function parseShotJson(strict = false) {
        const sources = [node.properties?.[stateKey], shotJsonW?.value];
        for (const source of sources) {
            if (!source) continue;
            try {
                const parsed = JSON.parse(source);
                if (parsed && Array.isArray(parsed.loras)) {
                    loras = parsed.loras;
                    audioMode = typeof parsed.audio_mode === "string"
                        ? parsed.audio_mode.trim().toLowerCase() : "";
                    return true;
                }
            } catch {}
        }
        if (!strict) {
            loras = [];
            audioMode = "";
        }
        return false;
    }
    function syncShotJson() {
        // Never write while a workflow restore is in flight: the restored
        // widget value arrives asynchronously and an early write would clobber
        // it with the pre-restore (empty) state.
        if (restoring) return;
        const value = JSON.stringify({ loras, audio_mode: audioMode });
        if (shotJsonW && shotJsonW.value !== value) shotJsonW.value = value;
        node.properties[stateKey] = value;
    }
    function scheduleSync() {
        clearTimeout(syncTimer);
        syncTimer = setTimeout(syncShotJson, 250);
    }
    function fetchLoras() {
        if (loras_fetching || loras_fetched) return Promise.resolve(loras_fetched || []);
        loras_fetching = true;
        const names = new Set();
        const jobs = ["LoraLoaderModelOnly", "LoraLoader"].map(async className => {
            try {
                const response = await api.fetchApi(`/object_info/${className}`, { cache: "no-store" });
                if (!response.ok) return;
                const info = await response.json();
                const classNameKey = Object.keys(info)[0];
                const spec = info?.[classNameKey]?.input;
                const field = spec?.required?.lora_name || spec?.optional?.lora_name;
                (Array.isArray(field?.[0]) ? field[0] : []).forEach(n => {
                    if (typeof n === "string" && n && n !== "None") names.add(n);
                });
            } catch {}
        });
        return Promise.all(jobs).then(async () => {
            if (!names.size) {
                try {
                    const response = await api.fetchApi(`/lk/minimax_h3/loras?t=${Date.now()}`, { cache: "no-store" });
                    if (response.ok) {
                        const data = await response.json();
                        (Array.isArray(data.loras) ? data.loras : []).forEach(n => names.add(n));
                    }
                } catch {}
            }
            loras_fetched = [...names].sort((a, b) => a.localeCompare(b));
            loras_fetching = false;
            return loras_fetched;
        });
    }
    function clearPanelWidgets() {
        for (const w of shotWidgets) {
            const at = node.widgets.indexOf(w);
            if (at >= 0) node.widgets.splice(at, 1);
        }
        shotWidgets.length = 0;
    }
    function addRow(type, name, value, callback, options = {}) {
        const w = node.addWidget(type, name, value, callback, options);
        w.serialize = false;
        w.options = { ...(w.options || {}), serialize: false };
        w._ghh3Shot = true;
        shotWidgets.push(w);
        return w;
    }
    function addButtonRow(label, onClick) {
        const w = node.addWidget("button", label, null, () => onClick());
        w.serialize = false;
        w.options = { ...(w.options || {}), serialize: false };
        w._ghh3Shot = true;
        shotWidgets.push(w);
        return w;
    }
    function fit() {
        if (node.graph) node.arrange?.();
        const fitted = typeof node.computeSize === "function" ? node.computeSize() : null;
        if (fitted && Number.isFinite(fitted[1]) && Math.abs(node.size[1] - Math.ceil(fitted[1])) > 2) {
            node.setSize([node.size[0], Math.ceil(fitted[1])]);
        }
        node.setDirtyCanvas?.(true, true);
    }

    function rebuildRows() {
        clearPanelWidgets();
        // ── per-shot audio mode (empty = follow the composer's global mode) ──
        const AUDIO_MODES = [
            ["", "跟随编排"],
            ["lock_source", "锁定原声"],
            ["reference_only", "仅参考"],
            ["native", "原生生成"],
        ];
        const audioLabel = (key) => (AUDIO_MODES.find(m => m[0] === key) || AUDIO_MODES[0])[1];
        addRow("combo", "音频模式", audioLabel(audioMode), v => {
            if (restoring) return;
            audioMode = (AUDIO_MODES.find(m => m[1] === v) || AUDIO_MODES[0])[0];
            scheduleSync();
        }, { values: AUDIO_MODES.map(m => m[1]) });
        const loraNames = loras_fetched || [];
        loras.forEach((entry, i) => {
            const values = ["None", ...loraNames];
            if (entry.name && !values.includes(entry.name)) values.push(entry.name);
            const combo = addRow("combo", `LoRA ${i + 1}`, entry.name || "None", v => {
                if (restoring) return;
                const name = !v || v === "None" ? "" : String(v);
                if (!name) { loras.splice(i, 1); scheduleSync(); rebuildRows(); }
                else { entry.name = name; scheduleSync(); }
            }, { values });
            addRow("number", `LoRA ${i + 1} 强度`, entry.strength_model ?? 1, v => {
                entry.strength_model = Math.max(-10, Math.min(10, Number(v) || 0));
                scheduleSync();
            }, { min: -10, max: 10, step: 0.05, precision: 2 });
        });
        if (loras.length < MAX_SEG_LORAS) addButtonRow("＋ LoRA", () => {
            if (loras.length >= MAX_SEG_LORAS) return;
            loras.push({ name: "", strength_model: 1 });
            scheduleSync();
            rebuildRows();
            fetchLoras().then(() => rebuildRows());
        });
        lastRowCount = shotWidgets.length;
        fit();
    }

    async function refreshLibraries() {
        await fetchLoras();
        rebuildRows();
    }

    // ── lifecycle ────────────────────────────────────────────────────────────
    parseShotJson();
    rebuildRows();
    refreshLibraries();
    // Creation-time init is done: allow panel edits to sync (workflow
    // configure re-arms `restoring` through the hooks below).
    restoring = false;

    // Watchdog for newer frontends (1.57.6+): they rebuild node.widgets from
    // the schema definition and can wipe dynamically added (serialize:false)
    // panel rows at any widget-store sync.  When the panel rows vanish while
    // the state still holds rows, re-apply them.  Cheap (a filter per 1.5s),
    // and it keeps the LoRA editor alive across frontend updates.
    const panelWatchdog = setInterval(() => {
        if (!node.graph || restoring) return;
        const alive = (node.widgets || []).filter(w => w._ghh3Shot).length;
        if (lastRowCount > 0 && alive === 0) {
            rebuildRows();
            // 同一场 store 重建也会把 schema widget 重置回默认渲染——
            // 顺路把它们的显示值拉回序列化真值。
            resyncSchemaWidgets(node, "分镜面板重建");
        }
    }, 1500);

    const oldSerialize = node.onSerialize;
    node.onSerialize = function(...args) {
        syncShotJson();
        const serializedNode = args[0];
        if (serializedNode) {
            serializedNode.properties = serializedNode.properties || {};
            serializedNode.properties[stateKey] = node.properties[stateKey];
        }
        return oldSerialize?.apply(this, args);
    };
    // First layout pass once the node is in the graph (the creation-time fit
    // is skipped because the graph reference doesn't exist yet).
    const oldAdded = node.onAdded;
    node.onAdded = function(...args) {
        const result = oldAdded?.apply(this, args);
        if (this.graph) fit();
        return result;
    };
    for (const hook of ["onConfigure", "onGraphConfigured"]) {
        const old = node[hook];
        node[hook] = function(...args) {
            const result = old?.apply(this, args);
            restoring = true;
            const epoch = ++restoreEpoch;
            // The frontend restores widget values asynchronously: retry until
            // the saved rows actually arrive (bounded), never clobber while
            // restoring.
            let attempt = 0;
            const tryRestore = () => {
                if (epoch !== restoreEpoch) return;
                const accepted = parseShotJson(true);
                if (accepted || attempt >= 4) {
                    rebuildRows();
                    restoring = false;
                    return;
                }
                attempt += 1;
                setTimeout(tryRestore, 300);
            };
            setTimeout(tryRestore, 0);
            return result;
        };
    }
    return true;
}

app.registerExtension({
    name: "longvideo.minimax_h3_integration",
    rh: {
        type: "nodes",
        nodes: RH_NODE_IDS,
    },
    async setup() {
        if (typeof app.ensureNodesRegistered !== "function") return;
        await app.ensureNodesRegistered(new Set(RH_NODE_IDS));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "LKH3LongVideo") {
            // The sampler face stays minimal: the sampling recipe lives in the
            // 采样设置 node / external NOISE-SIGMAS-SAMPLER primitives. Hidden
            // widgets still serialize, so legacy workflows keep their values.
            const previousLongVideo = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = previousLongVideo?.apply(this, arguments);
                const hideNames = new Set([
                    "steps", "shift_video", "shift_audio", "sampler_name", "scheduler",
                    "seed", "control_after_generate", "seed_policy", "context_noise", "cfg",
                ]);
                for (const w of this.widgets || []) {
                    if (hideNames.has(w.name)) {
                        w.hidden = true;
                        w.options = { ...(w.options || {}), hidden: true };
                    }
                }
                return result;
            };
            return;
        }
        if (nodeData.name === "LKH3SegmentGuider") {
            // Output-slot self-healing: canvases saved across output-list changes
            // keep stale serialized names/types (e.g. the removed conditioning
            // output). The Vue render layer can also re-apply the serialized
            // labels after onConfigure, so the fix runs immediately, on the next
            // animation frame and once more after 100ms.
            const EXPECTED_OUTPUTS = [
                { name: "GUIDER", type: "GUIDER" },
                { name: "sampler", type: "SAMPLER" },
                { name: "sigmas", type: "SIGMAS" },
            ];
            const fixOutputs = (node) => {
                if (!node.outputs) return;
                for (let i = 0; i < EXPECTED_OUTPUTS.length; i++) {
                    const want = EXPECTED_OUTPUTS[i];
                    const slot = node.outputs[i];
                    if (slot) {
                        if (slot.name !== want.name) slot.name = want.name;
                        if (slot.type !== want.type) slot.type = want.type;
                    } else {
                        node.addOutput(want.name, want.type);
                    }
                }
                // Ghost sockets: canvases saved across output-list changes can
                // carry MORE serialized outputs than the definition (the removed
                // conditioning output lives on as a 4th socket, and wires may
                // still hang there — the server would index past the returned
                // tuple). Relocate those wires to the expected slot of the same
                // type, then truncate to the definition.
                if (node.outputs.length > EXPECTED_OUTPUTS.length) {
                    for (let i = EXPECTED_OUTPUTS.length; i < node.outputs.length; i++) {
                        for (const lid of (node.outputs[i].links || []).slice()) {
                            const all = node.graph?._links;
                            const lk = typeof all?.get === "function" ? all.get(lid) : all?.[lid];
                            if (!lk || lk.origin_id !== node.id) continue;
                            const target = EXPECTED_OUTPUTS.findIndex((e) => e.type === lk.type);
                            if (target < 0) continue;
                            const dst = node.outputs[target];
                            dst.links = dst.links || [];
                            if (!dst.links.includes(lid)) dst.links.push(lid);
                            lk.origin_slot = target;
                        }
                    }
                    node.outputs.length = EXPECTED_OUTPUTS.length;
                    node.setDirtyCanvas?.(true, true);
                }
            };
            const previousGuiderCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = previousGuiderCreated?.apply(this, arguments);
                fixOutputs(this);
                return result;
            };
            const previousGuiderConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const result = previousGuiderConfigure?.apply(this, arguments);
                fixOutputs(this);
                requestAnimationFrame(() => fixOutputs(this));
                setTimeout(() => fixOutputs(this), 100);
                return result;
            };
            return;
        }
        // Output-list self-healing for the runtime-bundle consolidation: old
        // canvases serialize the composer's 5 outputs (CLIP/VAEs removed),
        // the decode node's 4 outputs (latents removed) and the refine's 2
        // (conditioning removed) - truncate ghosts and relocate same-type
        // wires, mirroring the guider's fix.
        const RUNTIME_OUTPUTS = {
            // LKH3Composer deliberately NOT listed: its bundle output is
            // appended after the original five slots, so pre-bundle canvases
            // keep every bare wire on its original slot and need no healing.
            LKH3AVDecode: [
                { name: "frames", type: "IMAGE" },
                { name: "audio", type: "AUDIO" },
            ],
            LKH3RefineSegments: [
                { name: "AV 潜空间", type: "LATENT" },
            ],
        };
        if (RUNTIME_OUTPUTS[nodeData.name]) {
            const EXPECTED = RUNTIME_OUTPUTS[nodeData.name];
            const fixNodeOutputs = (node) => {
                if (!node.outputs) return;
                for (let i = 0; i < EXPECTED.length; i++) {
                    const want = EXPECTED[i];
                    const slot = node.outputs[i];
                    if (slot) {
                        if (slot.name !== want.name) slot.name = want.name;
                        if (slot.type !== want.type) slot.type = want.type;
                    } else {
                        node.addOutput(want.name, want.type);
                    }
                }
                if (node.outputs.length > EXPECTED.length) {
                    for (let i = EXPECTED.length; i < node.outputs.length; i++) {
                        for (const lid of (node.outputs[i].links || []).slice()) {
                            const all = node.graph?._links;
                            const lk = typeof all?.get === "function" ? all.get(lid) : all?.[lid];
                            if (!lk || lk.origin_id !== node.id) continue;
                            const target = EXPECTED.findIndex((e) => e.type === lk.type);
                            if (target < 0) continue;
                            const dst = node.outputs[target];
                            dst.links = dst.links || [];
                            if (!dst.links.includes(lid)) dst.links.push(lid);
                            lk.origin_slot = target;
                        }
                    }
                    node.outputs.length = EXPECTED.length;
                    node.setDirtyCanvas?.(true, true);
                }
            };
            const previousCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = previousCreated?.apply(this, arguments);
                fixNodeOutputs(this);
                return result;
            };
            const previousConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const result = previousConfigure?.apply(this, arguments);
                fixNodeOutputs(this);
                requestAnimationFrame(() => fixNodeOutputs(this));
                setTimeout(() => fixNodeOutputs(this), 100);
                return result;
            };
            return;
        }
        if (nodeData.name === "LKH3Shot") {
            const previousShot = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = previousShot?.apply(this, arguments);
                if (!this._ghH3PanelReady && createShotPanel(this)) this._ghH3PanelReady = true;
                return result;
            };
            return;
        }
        if (nodeData.name === "LKH3ShotGroup") {
            const previousGroup = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = previousGroup?.apply(this, arguments);
                if (!this._ghh3GroupPanelReady) {
                    const panel = createGroupPanel(this);
                    if (panel) {
                        this._ghh3GroupPanelReady = true;
                        const prevAdded = this.onAdded;
                        this.onAdded = function() {
                            prevAdded?.apply(this, arguments);
                            panel.onAdded();
                        };
                        const prevConfigure = this.onConfigure;
                        this.onConfigure = function(state) {
                            prevConfigure?.apply(this, arguments);
                            panel.onConfigure(state);
                        };
                        const prevGraphConfigured = this.onGraphConfigured;
                        this.onGraphConfigured = function() {
                            prevGraphConfigured?.apply(this, arguments);
                            panel.onGraphConfigured();
                        };
                        const prevSerialize = this.onSerialize;
                        this.onSerialize = function(data) {
                            prevSerialize?.apply(this, arguments);
                            panel.onSerialize(data);
                        };
                        const prevRemoved = this.onRemoved;
                        this.onRemoved = function() {
                            prevRemoved?.apply(this, arguments);
                            panel.onRemoved();
                        };
                    }
                }
                return result;
            };
            return;
        }
        if (nodeData.name !== NODE) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = previous?.apply(this, arguments);
            if (!this._ghH3PanelReady && createPanel(this)) this._ghH3PanelReady = true;
            return result;
        };
    },
});

// ── 分镜组面板 v2（LKH3ShotGroup）─────────────────────────────────────────
// 视觉对齐 lanxi「长视频操作台」：青色描边卡片 + 头部徽章行 + 提示词框 +
// LoRA 快速行 + 虚线「＋添加分镜」。整块面板是单个 DOM widget，
// 状态仍双镜像（properties + group_json），后端语义不变。

const GROUP_AUDIO_MODES = [
    ["", "跟随编排"],
    ["lock_source", "锁定原声"],
    ["reference_only", "仅参考"],
    ["native", "原生生成"],
];
const GROUP_MAX_LORAS = 4;
// 粘贴仲裁：同页多个分镜组面板时，一次粘贴只由最近活跃的面板响应
// （token 比对 + 活跃度门槛，避免抢 ComfyUI 原生的画布图片粘贴）。
let lkgrpActiveToken = null;
let lkgrpActiveTs = 0;
const GROUP_PANEL_WIDTH = 460;
const GROUP_MAX_SHOTS = 3;

let groupLorasCache = null;
async function fetchGroupLoras(force = false) {
    if (groupLorasCache && !force) return groupLorasCache;
    let names = [];
    try {
        for (const route of ["/object_info/LoraLoaderModelOnly", "/object_info/LoraLoader"]) {
            const response = await api.fetchApi(route, { cache: "no-store" });
            if (!response.ok) continue;
            const info = await response.json();
            for (const entry of Object.values(info || {})) {
                const input = entry?.input?.required?.lora_name;
                if (Array.isArray(input) && Array.isArray(input[0])) names.push(...input[0]);
            }
            if (names.length) break;
        }
    } catch (error) { /* 落到本包路由 */ }
    if (!names.length) {
        try {
            const response = await api.fetchApi(`${SEGMENT_LORA_ROUTE}?t=${Date.now()}`, { cache: "no-store" });
            const data = response.ok ? await response.json() : {};
            if (Array.isArray(data.loras)) names = data.loras;
        } catch (error) {
            console.warn("[MiniMax H3 Integration] LoRA 列表不可用:", error);
        }
    }
    groupLorasCache = [...new Set(names)].sort((a, b) => a.localeCompare(b));
    return groupLorasCache;
}

let lkgrpStyleInstalled = false;
function ensureGroupPanelStyle() {
    if (lkgrpStyleInstalled || typeof document === "undefined") return;
    const style = document.createElement("style");
    style.textContent = `
      .lkgrp-root{display:flex;flex-direction:column;gap:8px;padding:2px 2px 6px;font:12px/1.4 Arial,sans-serif;color:#e1e9ef}
      .lkgrp-card{border:1px solid #2e6f86;border-radius:8px;background:#1a2938;overflow:hidden;box-shadow:0 0 0 1px rgba(24,189,211,.12)}
      .lkgrp-card.lkgrp-collapsed{background:#16222d}
      .lkgrp-card.lkgrp-empty{opacity:.45}
      .lkgrp-head{display:flex;align-items:center;gap:6px;padding:6px 8px;background:#213548;border-bottom:1px solid #2e6f86}
      .lkgrp-card.lkgrp-collapsed .lkgrp-head{border-bottom:0}
      .lkgrp-badge{font-size:11px;font-weight:bold;color:#7fe3ef;white-space:nowrap}
      .lkgrp-head .lkgrp-label{font-size:10px;color:#7f93a4;white-space:nowrap}
      .lkgrp-badge .lkgrp-main-tag{font-size:9px;color:#0db5e8;border:1px solid #1e6f86;border-radius:4px;padding:0 4px;margin-left:4px;vertical-align:1px}
      .lkgrp-head input[type=number],.lkgrp-head select{background:#12202c;color:#dbe8f1;border:1px solid #354b5d;border-radius:4px;font:11px Arial,sans-serif;padding:2px 4px;height:22px}
      .lkgrp-head input[type=number]{width:52px}
      .lkgrp-head select{max-width:96px}
      .lkgrp-spacer{flex:1}
      .lkgrp-toggle{border:1px solid #3a5363;border-radius:4px;background:#1d2b36;color:#cfe0ea;padding:1px 7px;cursor:pointer;font-size:10px}
      .lkgrp-toggle:hover{border-color:#18bdd3;color:#fff}
      .lkgrp-body{display:flex;flex-direction:column;gap:6px;padding:7px 8px 8px}
      .lkgrp-prompt{width:100%;min-height:84px;resize:vertical;overflow:auto;box-sizing:border-box;border:1px solid #2c4356;border-radius:6px;background:#1d2731;color:#e1e9ef;padding:7px;font:12px/1.4 Arial,sans-serif;outline:none;scrollbar-width:thin;scrollbar-color:#1f3540 transparent}
      .lkgrp-prompt::placeholder{color:#52616d}
      .lkgrp-prompt:focus{border-color:#18bdd3}
      .lkgrp-row{display:flex;align-items:center;gap:5px;width:100%;min-width:0}
      .lkgrp-row select{flex:1 1 120px;min-width:0;width:auto;max-width:100%}
      .lkgrp-row input[type=number]{flex:0 0 56px}
      .lkgrp-row .lkgrp-x{flex:0 0 auto}
      .lkgrp-row select{flex:1;min-width:0;background:#12202c;color:#dbe8f1;border:1px solid #354b5d;border-radius:4px;font:11px Arial,sans-serif;padding:2px 4px;height:22px}
      .lkgrp-row input[type=number]{width:56px;background:#12202c;color:#dbe8f1;border:1px solid #354b5d;border-radius:4px;font:11px Arial,sans-serif;height:22px;padding:2px 4px}
      .lkgrp-row .lkgrp-x{border:0;background:transparent;color:#c6d4de;font-size:12px;cursor:pointer;padding:0 3px}
      .lkgrp-row .lkgrp-x:hover{color:#ff8f8f}
      .lkgrp-label{font-size:10px;color:#7f93a4;white-space:nowrap}
      .lkgrp-number{display:flex;align-items:center;border:1px solid #354b5d;border-radius:4px;overflow:hidden;background:#12202c;height:24px}
      .lkgrp-number button{border:0;background:#1d2b36;color:#cfe0ea;width:20px;height:100%;cursor:pointer;font-size:10px;padding:0}
      .lkgrp-number button:hover{background:#294955;color:#fff}
      .lkgrp-number input{width:44px;height:100%;border:0;background:#12202c;color:#dbe8f1;font:11px Arial,sans-serif;text-align:center;padding:0 2px;outline:none}
      .lkgrp-head select,.lkgrp-row select{-webkit-appearance:none;appearance:none;background-image:linear-gradient(45deg,transparent 50%,#7f93a4 50%),linear-gradient(135deg,#7f93a4 50%,transparent 50%);background-position:calc(100% - 12px) 9px,calc(100% - 7px) 9px;background-size:5px 5px;background-repeat:no-repeat;padding-right:18px}
      .lkgrp-chip{border:1px dashed #35526a;border-radius:5px;background:#16242f;color:#9fc6d4;padding:2px 8px;cursor:pointer;font-size:10px}
      .lkgrp-chip:hover{border-color:#18bdd3;color:#dff4f7}
      .lkgrp-chips{display:flex;gap:5px;flex-wrap:wrap}
      .lkgrp-add{border:1px dashed #35526a;border-radius:7px;background:transparent;color:#9fc6d4;padding:6px;cursor:pointer;font-size:11px;text-align:center;width:100%}
      .lkgrp-badge[draggable="true"]{cursor:grab}
      .lkgrp-badge[draggable="true"]:active{cursor:grabbing}
      .lkgrp-mchip{cursor:grab}
      .lkgrp-drop-target{border-color:#18bdd3 !important;box-shadow:0 0 0 2px rgba(24,189,211,.35) inset;background:#1b3142 !important}
      .lkgrp-add:hover{border-color:#18bdd3;color:#dff4f7}
      .lkgrp-hint{font-size:10px;color:#52616d;line-height:1.3}
    `;
    document.head.appendChild(style);
    lkgrpStyleInstalled = true;
}

// ── 全局编号引擎（Composer 权威）───────────────────────────────────────────
// 从每个 Composer 的「分镜序列」口正向遍历链：散点 = 1 镜；组 = 参与镜数
// （enabled/有内容）。产出 (a) 激活节点集合——只有挂在 Composer 主链上的
// 组/散点才显示全局序号（分支链不激活，防止分支导致编号混乱）；
// (b) 每个节点的链前镜数（groupBase）。
let lkgrpNumbering = { stamp: 0, baseByNode: new Map(), activeNodes: new Set() };

function lkgrpRefreshNumbering() {
    const baseByNode = new Map();
    const activeNodes = new Set();
    const composers = (app.graph?._nodes || []).filter(n => n.type === "LKH3Composer");
    const groupInOf = (node) => {
        // Composer 的链入口叫 shot_group_chain，组/散点叫 group_in
        const inName = node.type === "LKH3Composer" ? "shot_group_chain" : "group_in";
        const linkId = (node.inputs || []).find(i => i.name === inName)?.link;
        if (linkId == null) return null;
        const links = app.graph.links;
        const link = typeof links.get === "function" ? links.get(linkId) : links[linkId];
        return link ? app.graph.getNodeById(link.origin_id) : null;
    };
    const participating = (node) => {
        if (node.type === "LKH3Shot") return 1;
        if (node.type === "LKH3ShotGroup") {
            const st = node.properties?.["gh_h3_group_state"];
            let count = 0;
            (st?.shots || []).forEach((s, i) => {
                // 与后端展平口径一致：主镜卡（首卡）只看内容不看 enabled
                // （主镜 enabled 恒为 true，空主镜不参与）；弹出镜看
                // enabled/内容。
                const content = String(s.prompt || "").trim() ||
                    (s.images || []).length || (s.videos || []).length ||
                    (s.audios || []).length || s.first_frame || s.last_frame;
                if (i === 0 ? !!content : (s.enabled || !!content)) count += 1;
            });
            return count;
        }
        return 0;
    };
    for (const composer of composers) {
        // 分镜序列口支持多连线（分支扩展）：每条入线是一条链，按连线顺序
        // 依次展平编号。Composer 入口名 shot_group_chain，组/散点 group_in。
        const inSlot = (composer.inputs || []).find(i => i.name === "shot_group_chain");
        const linkIds = inSlot?.link != null ? [inSlot.link]
            : (inSlot?.links || []);
        for (const linkId of linkIds) {
            if (linkId == null) continue;
            const links = app.graph.links;
            const firstLink = typeof links.get === "function" ? links.get(linkId) : links[linkId];
            const chainTail = firstLink ? app.graph.getNodeById(firstLink.origin_id) : null;
            if (!chainTail) continue;
            // 沿链尾反向收集整条链（链尾在前）：链尾自身也算链成员——
            // 它才是直接挂在 Composer 上的节点，漏掉它会让整条链的
            // 激活判定颠倒（直接接编排的节点永远「未接编排」）。
            const chain = [];
            let cur = chainTail;
            const seenChain = new Set();
            while (cur && !seenChain.has(cur.id)) {
                seenChain.add(cur.id);
                chain.push(cur);
                cur = groupInOf(cur);
            }
            chain.reverse(); // 链头在前
            let run = 0;
            for (const upstreamNode of chain) {
                if (activeNodes.has(upstreamNode.id)) continue; // 多分支重复引用：首个激活
                activeNodes.add(upstreamNode.id);
                const n = participating(upstreamNode);
                baseByNode.set(upstreamNode.id, run);
                run += n;
            }
        }
    }
    lkgrpNumbering = { stamp: Date.now(), baseByNode, activeNodes };
}

function createGroupPanel(node) {
    ensureGroupPanelStyle();
    const stateKey = "gh_h3_group_state";
    const groupJsonWidget = (node.widgets || []).find(w => w.name === "group_json");
    const defaultShot = () => ({
        duration: 5, prompt: "", audio_mode: "", loras: [],
        images: [], videos: [], audios: [], first_frame: "", last_frame: "",
        inherit_refs: true, enabled: false, expanded: false,
    });
    let shots = [defaultShot(), defaultShot(), defaultShot()];
    let restoring = false;
    let restoreEpoch = 0;
    let epochAtRestoreStart = 0;
    let syncTimer = null;
    let loras = [];
    let rootEl = null;
    let panelWidget = null;
    // （高度改为状态估算，见 estimateHeight）

    const snapshot = () => JSON.parse(JSON.stringify({ shots }));

    const parseGroupState = (strict) => {
        let raw = node.properties?.[stateKey];
        if ((!raw || !strict) && groupJsonWidget?.value) {
            try { raw = JSON.parse(groupJsonWidget.value); } catch { raw = null; }
        }
        let parsed = null;
        try { parsed = typeof raw === "string" ? JSON.parse(raw) : raw; } catch { parsed = null; }
        const rows = parsed?.shots;
        if (!Array.isArray(rows)) {
            if (!strict) shots = [defaultShot(), defaultShot(), defaultShot()];
            return false;
        }
        const clamp = (value, low, high, fallback) => {
            const numeric = Math.round(Number(value));
            return Number.isFinite(numeric) ? Math.max(low, Math.min(high, numeric)) : fallback;
        };
        // normalizeRow（结构照搬 lanxi normalizeSegment）：逐字段类型收敛 +
        // 钳制 + 过滤，手改 group_json 也不会让面板出现坏值。
        const normalizeRow = (raw, index) => {
            const row = raw && typeof raw === "object" ? raw : {};
            const audio = typeof row.audio_mode === "string" ? row.audio_mode.trim().toLowerCase() : "";
            return {
                duration: clamp(row.duration, 2, 30, 5),
                prompt: String(row.prompt || ""),
                audio_mode: ["lock_source", "reference_only", "native"].includes(audio) ? audio : "",
                loras: (Array.isArray(row.loras) ? row.loras : [])
                    .map(item => typeof item === "string"
                        ? { name: item, strength_model: 1 }
                        : { name: String(item?.name || ""),
                            strength_model: Math.max(-10, Math.min(10, Number(item?.strength_model) || 1)) })
                    .filter(item => item.name).slice(0, GROUP_MAX_LORAS),
                images: (Array.isArray(row.images) ? row.images : [])
                    .map(item => String(item || "").trim()).filter(Boolean),
                videos: (Array.isArray(row.videos) ? row.videos : [])
                    .map(item => String(item || "").trim()).filter(Boolean),
                audios: (Array.isArray(row.audios) ? row.audios : [])
                    .map(item => String(item || "").trim()).filter(Boolean),
                first_frame: String(row.first_frame || "").trim(),
                last_frame: String(row.last_frame || "").trim(),
                inherit_refs: row.inherit_refs !== false,
                enabled: index === 0 ? true : !!row.enabled,
                expanded: index === 0 ? true : !!row.expanded,
            };
        };
        shots = [0, 1, 2].map((i) => normalizeRow(rows[i], i));
        return true;
    };

    const syncGroupJson = () => {
        if (restoring) return;
        const payload = JSON.stringify(snapshot());
        if (groupJsonWidget) groupJsonWidget.value = payload;
        node.properties = node.properties || {};
        node.properties[stateKey] = JSON.parse(payload);
    };
    const scheduleSync = () => {
        if (restoring) return;
        if (syncTimer) clearTimeout(syncTimer);
        syncTimer = setTimeout(syncGroupJson, 250);
    };

    const el = (tag, cls, text) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
    };
    // 面板高度按状态确定性估算（卡片头 36 / 展开体 = 提示词 100 + 行 26 /
    // 加钮 30 / 提示 20 / 卡距 8 / 底部 42），不依赖运行时测量。
    const estimateHeight = () => {
        let h = 14;
        for (const shot of shots) {
            h += 36;
            if (shot.expanded) {
                h += 100;
                h += shot.loras.length * 26 + 30;
                if (shot !== shots[0]) h += 20;
            }
            h += 8;
        }
        h += 42;
        return Math.max(150, h);
    };
    let fitScheduled = false;
    const fit = () => {
        if (!node.graph || fitScheduled) return; // createNode 阶段未入图：布局由 onAdded 补
        fitScheduled = true;
        requestAnimationFrame(() => {
            fitScheduled = false;
            if (!node.graph) return;
            try {
                node.setSize?.([GROUP_PANEL_WIDTH, estimateHeight() + 8]);
                app.graph.setDirtyCanvas?.(true, true);
            } catch { /* 未入图阶段静默 */ }
        });
    };
    const measure = () => {
        if (!rootEl) return;
        fit();
    };

    // 全局序号：沿 上游分镜组 连线回溯，统计链上位于本组之前的参与镜数
    // （散点 = 1 镜；组 = enabled/有内容的镜数；成环或断线即停）。
    // 全局序号：查 Composer 权威编号引擎。本组不在任何 Composer 主链上
    // （未激活）→ 返回 null → 徽章回落组内号；激活 → 上游链前镜数。
    const countShotsBefore = () => {
        if (!lkgrpNumbering.activeNodes.has(node.id)) return null;
        return lkgrpNumbering.baseByNode.get(node.id) ?? 0;
    };
    let lastUpstreamShots = null;
    let dragInProgress = false;
    // 粘贴上传目标：最近点击/编辑的卡（默认主镜）。剪贴板里的图片直接
    // 进入该卡的参考列表——截图类参考无需经过文件选择框。
    let activeCardIndex = 0;
    const myToken = Symbol();
    const handlePasteImage = async (ev) => {
        if (restoring) return;
        // 焦点驱动：提示词栏（或任何输入框）处于输入状态 = 这次粘贴是文字，
        // 完全交给默认行为——即使剪贴板是"文字+图片"混合内容也不拦截上传
        // （否则文字会被错误吞掉）。非输入状态才尝试粘贴图片进参考列表。
        const target = ev.target;
        const inTextEntry = target instanceof HTMLTextAreaElement
            || target instanceof HTMLInputElement
            || (target && target.isContentEditable);
        if (inTextEntry) return;
        const items = ev.clipboardData?.items;
        if (!items) return;
        const imageItem = [...items].find(it => it.kind === "file" && it.type.startsWith("image/"));
        if (!imageItem) return; // 剪贴板无图片：不拦截
        // 仲裁：只有最近活跃的分镜组面板响应这一次粘贴（10s 活跃度门槛，
        // 超时放行给 ComfyUI 原生的画布图片粘贴）。
        if (lkgrpActiveToken !== myToken) return;
        if (Date.now() - lkgrpActiveTs > 10000) return;
        ev.preventDefault(); ev.stopPropagation();
        const file = imageItem.getAsFile();
        if (!file) return;
        try {
            // 专用子目录：绝不与 ComfyUI 原生粘贴目录（input/pasted/）或
            // 其他来源混用——清理时不存在归属歧义。
            const name = await uploadFile(file, "LK_paste");
            const shot = shots[activeCardIndex];
            if (!shot) return;
            shot.images.push(name);
            scheduleSync(); render();
        } catch (error) {
            console.warn("[MiniMax H3 Integration] 粘贴上传失败:", error);
        }
    };
    document.addEventListener("paste", handlePasteImage, true);

    const renderCard = (shot, index, globalNo, globalActive) => {
        const main = index === 0;
        const card = el("div", "lkgrp-card" + (shot.expanded ? "" : " lkgrp-collapsed"));
        // 粘贴目标追踪：点击/编辑某卡时它成为粘贴目标
        card.addEventListener("pointerdown", () => {
            activeCardIndex = index;
            lkgrpActiveToken = myToken;
            lkgrpActiveTs = Date.now();
        }, true);
        const head = el("div", "lkgrp-head");
        const badge = el("span", "lkgrp-badge",
            globalActive ? `分镜 ${globalNo}` : `分镜 ${index + 1}（未接编排）`);
        if (!globalActive) badge.style.color = "#8298a9";
        badge.draggable = true;
        badge.title = "拖拽徽章可与其它分镜卡换序";
        if (main) {
            const tag = el("span", "lkgrp-main-tag", "主");
            badge.appendChild(tag);
        }
        // 卡序拖拽：徽章拖起，落到目标卡上换序（shots 数组 splice）
        badge.addEventListener("dragstart", ev => {
            dragInProgress = true;
            ev.dataTransfer.setData("text/lkgrp-shot", JSON.stringify({ idx: index, nodeId: node.id }));
            ev.dataTransfer.effectAllowed = "move";
        });
        badge.addEventListener("dragend", () => { dragInProgress = false; });
        card.addEventListener("dragover", ev => {
            if ([...ev.dataTransfer.types].includes("text/lkgrp-shot")) {
                ev.preventDefault();
                ev.stopPropagation();
                card.classList.add("lkgrp-drop-target");
            }
        });
        card.addEventListener("dragleave", () => card.classList.remove("lkgrp-drop-target"));
        card.addEventListener("drop", ev => {
            ev.preventDefault(); ev.stopPropagation();
            card.classList.remove("lkgrp-drop-target");
            let src = null;
            try { src = JSON.parse(ev.dataTransfer.getData("text/lkgrp-shot") || "null"); } catch { src = null; }
            if (!src) return;
            if (src.nodeId !== node.id) return; // 跨节点拖拽：忽略（各自组内排序）
            if (src.idx === index) return;
            const moved = shots.splice(src.idx, 1)[0];
            if (!moved) return;
            shots.splice(index, 0, moved);
            scheduleSync(); render();
        });
        head.appendChild(badge);
        const duration = el("input");
        duration.type = "number"; duration.min = "2"; duration.max = "30"; duration.step = "1";
        duration.value = shot.duration;
        duration.title = "时长（秒）";
        duration.addEventListener("change", () => {
            shot.duration = Math.max(2, Math.min(30, Math.round(Number(duration.value) || 5)));
            duration.value = shot.duration;
            scheduleSync();
        });
        head.appendChild(duration);
        duration.style.display = "none";
        // 步进器（结构照搬 lanxi number 控件）：◀ 值 ▶
        const durWrap = el("div", "lkgrp-number");
        durWrap.title = "时长（秒）";
        const durInput = el("input");
        durInput.type = "number"; durInput.min = "2"; durInput.max = "30";
        durInput.value = shot.duration;
        const setDuration = value => {
            shot.duration = Math.max(2, Math.min(30, Math.round(Number(value)) || 5));
            durInput.value = Number.isInteger(shot.duration) ? shot.duration : shot.duration.toFixed(2);
            scheduleSync();
        };
        const leftB = el("button", undefined, "\u25C0");
        leftB.addEventListener("click", () => setDuration(Number(durInput.value || 2) - 1));
        const rightB = el("button", undefined, "\u25B6");
        rightB.addEventListener("click", () => setDuration(Number(durInput.value || 2) + 1));
        durInput.addEventListener("change", () => setDuration(durInput.value));
        durWrap.append(leftB, durInput, rightB);
        head.insertBefore(durWrap, head.querySelector("select"));
        const audioSel = el("select");
        for (const [key, text] of GROUP_AUDIO_MODES) {
            const opt = el("option", undefined, text); opt.value = key;
            audioSel.appendChild(opt);
        }
        audioSel.value = shot.audio_mode || "";
        audioSel.title = "音频参考模式：本镜生成的音频如何对待参考音频（跟随编排 = 沿用编排节点的全局设置）";
        head.appendChild(el("span", "lkgrp-label", "音频参考模式"));
        audioSel.addEventListener("change", () => { shot.audio_mode = audioSel.value; scheduleSync(); });
        head.appendChild(audioSel);
        head.appendChild(el("span", "lkgrp-spacer"));
        if (!main) {
            const inheritSel = el("select");
            for (const [value, text] of [["inherit", "继承主分镜"], ["own", "独立"]]) {
                const opt = el("option", undefined, text); opt.value = value;
                inheritSel.appendChild(opt);
            }
            inheritSel.value = shot.inherit_refs ? "inherit" : "own";
            inheritSel.title = "参考内容：本镜的参考图/视频/音频沿用主分镜（继承主分镜）还是用按钮上传的独立参考（独立）";
            head.appendChild(el("span", "lkgrp-label", "参考内容"));
            inheritSel.addEventListener("change", () => {
                shot.inherit_refs = inheritSel.value === "inherit";
                scheduleSync();
            });
            head.appendChild(inheritSel);
        }
        const toggle = el("button", "lkgrp-toggle", shot.expanded ? "▲" : "▼");
        toggle.addEventListener("click", () => {
            shot.expanded = !shot.expanded;
            if (shot.expanded && !main) shot.enabled = true;
            syncGroupJson();
            render();
        });
        head.appendChild(toggle);
        card.appendChild(head);

        if (shot.expanded) {
            const body = el("div", "lkgrp-body");
            const prompt = el("textarea", "lkgrp-prompt");
            prompt.placeholder = "分段提示词（叠加在全片提示词之后）…";
            prompt.value = shot.prompt;
            prompt.addEventListener("focus", () => {
                activeCardIndex = index;
                lkgrpActiveToken = myToken;
                lkgrpActiveTs = Date.now();
            });
            prompt.addEventListener("input", () => { shot.prompt = prompt.value; scheduleSync(); });
            body.appendChild(prompt);

            const loraBox = el("div", "lkgrp-chips");
            shot.loras.forEach((lora, li) => {
                const row = el("div", "lkgrp-row");
                row.appendChild(el("span", "lkgrp-label", `LoRA${li + 1}`));
                const sel = el("select");
                for (const name of (loras.length ? loras : [lora.name])) {
                    const opt = el("option", undefined, name); opt.value = name;
                    sel.appendChild(opt);
                }
                sel.value = lora.name;
                sel.addEventListener("change", () => { lora.name = sel.value; scheduleSync(); });
                row.appendChild(sel);
                const strength = el("input");
                strength.type = "number"; strength.min = "-10"; strength.max = "10"; strength.step = "0.05";
                strength.value = lora.strength_model; strength.title = "强度";
                strength.addEventListener("change", () => {
                    lora.strength_model = Math.max(-10, Math.min(10, Number(strength.value) || 0));
                    scheduleSync();
                });
                row.appendChild(strength);
                const x = el("button", "lkgrp-x", "×"); x.title = "移除";
                x.addEventListener("click", () => { shot.loras.splice(li, 1); scheduleSync(); render(); });
                row.appendChild(x);
                loraBox.appendChild(row);
            });
            if (shot.loras.length < GROUP_MAX_LORAS) {
                const addLora = el("button", "lkgrp-chip", "＋LoRA");
                addLora.addEventListener("click", () => {
                    shot.loras.push({ name: loras[0] || "", strength_model: 1.0 });
                    scheduleSync(); render();
                });
                loraBox.appendChild(addLora);
            }
            body.appendChild(loraBox);

            {
                // 媒体上传区（lanxi 操作台按钮驱动方式）：所有镜都可按钮上传
                const mediaBox = el("div", "lkgrp-chips");
                const mediaChips = el("div", "lkgrp-chips");
                const chipRemove = (chip, list, idx) => {
                    chip.className = "lkgrp-chip";
                    chip.style.color = "#ffb0b0";
                    chip.textContent = "× " + chip.textContent;
                    chip.addEventListener("click", ev => {
                        ev.stopPropagation();
                        list.splice(idx, 1);
                        scheduleSync(); render();
                    });
                };
                const wireMediaDrag = (chip, kind, idx) => {
                    chip.draggable = true;
                    chip.classList.add("lkgrp-mchip");
                    chip.addEventListener("dragstart", ev => {
                        dragInProgress = true;
                        ev.dataTransfer.setData("text/lkgrp-media", JSON.stringify({ kind, idx, nodeId: node.id }));
                        ev.dataTransfer.effectAllowed = "move";
                    });
                    chip.addEventListener("dragend", () => { dragInProgress = false; });
                    chip.addEventListener("dragover", ev => {
                        if ([...ev.dataTransfer.types].includes("text/lkgrp-media")) {
                            ev.preventDefault();
                            ev.stopPropagation();
                            chip.classList.add("lkgrp-drop-target");
                        }
                    });
                    chip.addEventListener("dragleave", () => chip.classList.remove("lkgrp-drop-target"));
                    chip.addEventListener("drop", ev => {
                        ev.preventDefault(); ev.stopPropagation();
                        chip.classList.remove("lkgrp-drop-target");
                        let src = null;
                        try { src = JSON.parse(ev.dataTransfer.getData("text/lkgrp-media") || "null"); } catch { src = null; }
                        if (!src || src.nodeId !== node.id) return; // 跨组拖拽：忽略
                        if (src.kind !== kind || src.idx === idx) return;
                        const list = shot[kind];
                        const moved = list.splice(src.idx, 1)[0];
                        if (!moved) return;
                        list.splice(idx, 0, moved);
                        scheduleSync(); render();
                    });
                };
                shot.images.forEach((name, idx) => {
                    const chip = el("button", "lkgrp-chip lkgrp-mchip", "🖼 " + name.split("/").pop());
                    chip.title = "拖拽可排序，点击 × 移除";
                    chipRemove(chip, shot.images, idx);
                    wireMediaDrag(chip, "images", idx);
                    mediaChips.appendChild(chip);
                });
                shot.videos.forEach((name, idx) => {
                    const chip = el("button", "lkgrp-chip lkgrp-mchip", "🎬 " + name.split("/").pop());
                    chip.title = "拖拽可排序，点击 × 移除";
                    chipRemove(chip, shot.videos, idx);
                    wireMediaDrag(chip, "videos", idx);
                    mediaChips.appendChild(chip);
                });
                shot.audios.forEach((name, idx) => {
                    const chip = el("button", "lkgrp-chip lkgrp-mchip", "🔊 " + name.split("/").pop());
                    chip.title = "拖拽可排序，点击 × 移除";
                    chipRemove(chip, shot.audios, idx);
                    wireMediaDrag(chip, "audios", idx);
                    mediaChips.appendChild(chip);
                });
                if (shot.first_frame) {
                    const chip = el("button", "lkgrp-chip", "首帧 " + shot.first_frame.split("/").pop());
                    chip.style.color = "#ffb0b0";
                    chip.addEventListener("click", ev => {
                        ev.stopPropagation();
                        shot.first_frame = "";
                        scheduleSync(); render();
                    });
                    mediaChips.appendChild(chip);
                }
                if (shot.last_frame) {
                    const chip = el("button", "lkgrp-chip", "尾帧 " + shot.last_frame.split("/").pop());
                    chip.addEventListener("click", ev => {
                        ev.stopPropagation();
                        shot.last_frame = "";
                        scheduleSync(); render();
                    });
                    chip.style.color = "#ffb0b0";
                    mediaChips.appendChild(chip);
                }
                if (mediaChips.childElementCount) body.appendChild(mediaChips);
                const pickers = [
                    ["＋图片", "image/*", "images"],
                    ["＋视频", "video/*", "videos"],
                    ["＋音频", "audio/*", "audios"],
                    ["＋首帧", "image/*", "first"],
                    ["＋尾帧", "image/*", "last"],
                ];
                const pickerRow = el("div", "lkgrp-chips");
                for (const [label, accept, target] of pickers) {
                    const btn = el("button", "lkgrp-chip", label);
                    btn.addEventListener("click", () => {
                        const input = document.createElement("input");
                        input.type = "file"; input.accept = accept;
                        input.addEventListener("change", async () => {
                            const file = input.files?.[0];
                            if (!file) return;
                            try {
                                const name = await uploadFile(file);
                                if (target === "first") shot.first_frame = name;
                                else if (target === "last") shot.last_frame = name;
                                else shot[target].push(name);
                                scheduleSync(); render();
                            } catch (error) {
                                console.warn("[MiniMax H3 Integration] 上传失败:", error);
                            }
                        });
                        input.click();
                    });
                    pickerRow.appendChild(btn);
                }
                mediaBox.appendChild(pickerRow);
                body.appendChild(mediaBox);
                body.appendChild(el("div", "lkgrp-hint", main
                    ? "参考：跟随上游链上下文（默认）。点击卡片后在空白处 Ctrl+V 粘贴截图进参考（提示词框内粘贴 = 文字），独立参考优先于继承。"
                    : "参考：继承主分镜（与主分镜相同）。点击卡片后在空白处 Ctrl+V 粘贴截图进参考（提示词框内粘贴 = 文字），独立参考优先于继承。"));
            }
            card.appendChild(body);
        }
        return card;
    };

    const render = () => {
        if (!rootEl) return;
        rootEl.replaceChildren();
        // 全局序号：上游参与镜数 + 本组内前序参与镜数；未参与卡不占号。
        // 本组未挂 Composer 主链（激活外）→ globalActive=false → 组内号。
        const globalActive = lkgrpNumbering.activeNodes.has(node.id);
        const upstreamShots = globalActive
            ? (lkgrpNumbering.baseByNode.get(node.id) ?? 0) : 0;
        let run = upstreamShots;
        const numbers = shots.map((shot, i) => {
            const content = String(shot.prompt || "").trim() || (shot.images || []).length ||
                (shot.videos || []).length || (shot.audios || []).length ||
                shot.first_frame || shot.last_frame;
            const part = i === 0 ? !!content : (shot.enabled || !!content);
            if (part) { run += 1; return run; }
            return null;
        });
        shots.forEach((shot, index) => rootEl.appendChild(renderCard(
            shot, index,
            globalActive ? (numbers[index] || index + 1) : index + 1,
            globalActive,
        )));
        const nextDisabled = shots.findIndex((shot, i) => i > 0 && !shot.enabled);
        if (nextDisabled > 0) {
            const add = el("button", "lkgrp-add", "＋ 添加分镜");
            add.addEventListener("click", () => {
                const shot = shots[nextDisabled];
                shot.enabled = true;
                shot.expanded = true;
                syncGroupJson();
                render();
            });
            rootEl.appendChild(add);
        } else {
            rootEl.appendChild(el("div", "lkgrp-hint", "已达 3 镜上限（一个项目束分镜组最多 3 镜；需要更多请再拖一个分镜组首尾相接）。"));
        }
        measure();
    };

    const PANEL_WIDGET_NAME = "分镜组面板";
    const ensureWidget = () => {
        // widget-store 重建会产生同名副本/拆下根节点：按名复用 + 清理副本。
        const byName = (node.widgets || []).filter(w => w.name === PANEL_WIDGET_NAME);
        for (let i = byName.length - 1; i >= 1; i--) {
            const idx = node.widgets.indexOf(byName[i]);
            if (idx >= 0) node.widgets.splice(idx, 1);
        }
        const existing = byName.find(w => w._lkgrpRoot);
        // 重建壳（同名但无 _lkgrpRoot）一律移除，避免每次 store 同步累积副本
        for (const shell of byName) {
            if (shell !== existing) {
                const idx = node.widgets.indexOf(shell);
                if (idx >= 0) node.widgets.splice(idx, 1);
            }
        }
        if (existing && existing._lkgrpRoot) {
            panelWidget = existing;
            rootEl = existing._lkgrpRoot;
            if (!rootEl.isConnected) {
                const container = existing.inputEl?.parentElement
                    || existing.domElement?.parentElement
                    || existing.element?.parentElement;
                if (container) container.appendChild(rootEl);
            }
            return;
        }
        rootEl = document.createElement("div");
        rootEl.className = "lkgrp-root";
        // 显式像素宽度：DOM-widget 容器在恢复期会瞬时报窄/归零（lanxi 同款
        // 坑），显式 px 宽度保证面板不被压塌。
        rootEl.style.width = "448px";
        rootEl.style.minWidth = "448px";
        panelWidget = node.addDOMWidget(PANEL_WIDGET_NAME, "lkgrp_panel", rootEl,
            { serialize: false, hideOnZoom: false });
        // lanxi 操作台结构：容器高度交给管理器（getHeight 100%），节点高度
        // 由面板自行 setSize 管理；computeSize 只固定宽度。
        panelWidget.options = panelWidget.options || {};
        panelWidget.options.serialize = false;
        panelWidget.options.getMinHeight = () => 150;
        panelWidget.options.getHeight = () => "100%";
        panelWidget.options.minNodeSize = [GROUP_PANEL_WIDTH, 0];
        panelWidget._ghh3Group = true;
        panelWidget._lkgrpRoot = rootEl;

        render();
    };
    ensureWidget();

    const refreshLibraries = async () => {
        loras = await fetchGroupLoras();
        render();
    };

    const tryRestore = (attempt = 0) => {
        if (restoreEpoch !== epochAtRestoreStart) return; // 更新的恢复已接管
        parseGroupState(true);
        render();
        if (attempt >= 3) {
            restoring = false;
            refreshLibraries();
            return;
        }
        setTimeout(() => {
            if (!restoring || restoreEpoch !== epochAtRestoreStart) return;
            tryRestore(attempt + 1);
        }, 300);
    };

    const baseComputeSize = node.computeSize.bind(node);
    node.computeSize = function(out) {
        const measured = baseComputeSize(out);
        measured[0] = GROUP_PANEL_WIDTH;
        return measured;
    };
    parseGroupState(false);
    render();
    // 生成瞬间同步定宽（前端会先写一个 270 宽的默认尺寸，绕过 setSize；
    // 高度随卡片展开由 fit 持续修正）。
    node.size = [GROUP_PANEL_WIDTH, estimateHeight() + 8];
    refreshLibraries();

    const watchdog = setInterval(() => {
        if (restoring) return;
        const widgetAlive = (node.widgets || []).some(w => w.name === "分镜组面板");
        if (!widgetAlive) { ensureWidget(); return; }
        // 宽度再断言：前端恢复/同步可能把节点宽度重置回窄值，这里自愈
        if (node.size && node.size[0] < GROUP_PANEL_WIDTH) {
            node.setSize?.([GROUP_PANEL_WIDTH, Math.max(node.size[1], estimateHeight() + 8)]);
        }
        // 编号引擎刷新（Composer 侧正向遍历）→ 本组激活态或基准号漂移
        // → 刷新徽章（连线/断线/分支/上游增删镜都在此通道生效）
        lkgrpRefreshNumbering();
        const upstreamShots = countShotsBefore();
        if (upstreamShots !== lastUpstreamShots) {
            lastUpstreamShots = upstreamShots;
            render();
        }
    }, 2500);

    return {
        onAdded() {
            // 入图后补一次布局（创建阶段 graph 为 null 时 fit 被跳过）。
            setTimeout(() => { ensureWidget(); fit(); }, 0);
        },
        onConfigure() {
            restoring = true;
            restoreEpoch += 1;
            epochAtRestoreStart = restoreEpoch;
            setTimeout(() => tryRestore(), 0);
        },
        onGraphConfigured() {
            restoring = true;
            restoreEpoch += 1;
            epochAtRestoreStart = restoreEpoch;
            setTimeout(() => tryRestore(), 0);
        },
        onRemoved() {
            clearInterval(watchdog);
            if (syncTimer) clearTimeout(syncTimer);
            document.removeEventListener("paste", handlePasteImage, true);
        },
        onSerialize(serializedNode) {
            syncGroupJson();
            serializedNode.properties = serializedNode.properties || {};
            serializedNode.properties[stateKey] = snapshot();
        },
    };
}


// ── Schema widget 纠偏看门狗 ──────────────────────────────────────────
// 1.57.x widget-store 同步会按 schema 重建 node.widgets：重建后的 widget
// 渲染 schema 默认值，却不写回 widgets_values 数组——而数组才是队列/保存
// 时真正发往后台的值（实例：release_clip 显示 false、序列化 true，TE 热
// 重载退化就这样静默发生）。在配置/重建窗口内把 widget 显示值向序列化
// 数组对齐；窗口外绝不碰，避免对抗用户正在进行的编辑。每次纠偏都留痕。
function resyncSchemaWidgets(node, why) {
    const values = node.widgets_values;
    if (!Array.isArray(values)) return;
    let healed = 0;
    (node.widgets || []).forEach((widget, index) => {
        if (!widget || widget.serialize === false) return;
        if (widget._ghh3Shot || widget._ghh3Group) return; // 面板行由各自看门狗管理
        if (index >= values.length) return;
        const truth = values[index];
        if (truth === null || truth === undefined) return;
        if (typeof truth !== typeof widget.value) return; // 形状变更交给配置流程
        if (JSON.stringify(widget.value) === JSON.stringify(truth)) return;
        if (widget._lkSchemaResynced) return; // 单次纠偏：之后用户的合法编辑不再回退
        console.warn(
            `[MiniMax H3 Integration] "${widget.name}" 显示=${JSON.stringify(widget.value)} `
            + `≠ 序列化=${JSON.stringify(truth)}（${why}）— 已向序列化值纠偏`,
        );
        widget.value = truth;
        widget._lkSchemaResynced = true;
        try { widget.callback?.(truth); } catch { /* 回调可选 */ }
        healed += 1;
    });
    if (healed) node.setDirtyCanvas?.(true, true);
}

function scheduleSchemaResync(node, why) {
    for (const delay of [0, 1000, 3000, 6000]) {
        setTimeout(() => {
            if (node.graph) resyncSchemaWidgets(node, why);
        }, delay);
    }
}

const LK_TITLES = {
    LKH3Shot: "MiniMaxH3分镜（LK）",
    LKH3Composer: "MiniMaxH3 长视频编排（LK）",
    LKH3SegmentGuider: "MiniMax-H3 分段引导（LK）",
    LKH3RefineSegments: "MiniMax-H3 二跑分段精修（LK）",
    LKH3AVUpscale: "MiniMax-H3 AV潜空间放大（LK）",
    LKH3AVDecode: "MiniMax-H3 AV解码（LK）",
};
app.registerExtension({
    name: "longvideo.titles",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LK_TITLES[nodeData.name]) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(...args) {
            const result = previous?.apply(this, args);
            this.title = LK_TITLES[nodeData.name];
            this.setDirtyCanvas?.(true, true);
            return result;
        };
        // Serialized canvases keep node titles saved under the old lanxi
        // branding; onConfigure restores that stored title AFTER onNodeCreated
        // applied the LK one, so re-apply here - but only when the stored
        // title still carries the stale branding, leaving genuine user
        // renames untouched.
        const previousConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(...args) {
            const result = previousConfigure?.apply(this, args);
            const retitle = () => {
                if (/lanxi/i.test(this.title || "")) {
                    this.title = LK_TITLES[nodeData.name];
                    this.setDirtyCanvas?.(true, true);
                }
            };
            retitle();
            requestAnimationFrame(retitle);
            setTimeout(retitle, 100);
            // Widget-store 重建风暴窗口：加载后多次纠偏，把重建造成的
            // "显示=schema默认 / 数组=用户真值" 脱钩压回去。
            scheduleSchemaResync(this, "工作流配置");
            return result;
        };
        const previousGraphConfigured = nodeType.prototype.onGraphConfigured;
        nodeType.prototype.onGraphConfigured = function(...args) {
            const result = previousGraphConfigured?.apply(this, args);
            scheduleSchemaResync(this, "画布加载");
            return result;
        };
    },
});
