let state = {
  uid: null,
  dataset: null,
  research: [],
  tracks: new Map(),
  progressSocket: null,
  progressPoll: null,
  lastProgressAt: Date.now(),
  currentProgressEvent: null,
  progressEvents: [],
};

const rawVideo = document.getElementById("raw-video");
const annotatedVideo = document.getElementById("annotated-video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const PHASES = [
  { id: 1, title: "Ingest", backed: "FFmpeg + OpenCV", detail: "Normalize video and extract sampled frames." },
  { id: 2, title: "Segment", backed: "PySceneDetect + duration rules", detail: "Split the video into action clips." },
  { id: 3, title: "Annotate", backed: "YOLO-World + hand/contact modules", detail: "Detect hands, objects, contact, grasp, and overlay tracks." },
  { id: 4, title: "Describe", backed: "Gemini prompt or fallback", detail: "Generate narration, verb, noun, and tool." },
  { id: 5, title: "Assemble", backed: "Ego4D-style schema", detail: "Merge all phase files into dataset.json." },
  { id: 6, title: "Validate", backed: "Rules + CLIP-style QA", detail: "Check quality, confidence, and consistency." },
];

async function boot() {
  renderTimeline();
  setInterval(updateHeartbeat, 1000);
  state.research = await fetch("/api/research").then((res) => res.json());
  window.renderResearchRibbon(state.research);
  window.renderResearchPanel(state.research);
  window.attachFieldTooltips(state.research);

  const videos = await fetch("/api/videos").then((res) => res.json());
  if (videos.length) {
    await loadVideo(videos[videos.length - 1]);
  }
}

async function loadVideo(uid) {
  state.uid = uid;
  document.getElementById("video-id").textContent = uid;
  state.dataset = await fetchJson(`/api/video/${uid}/dataset.json`);
  await loadTracks(uid);
  document.getElementById("download-json").href = `/api/video/${uid}/dataset.json`;
  document.getElementById("download-json").classList.remove("disabled");
  rawVideo.src = `/stream/${uid}/normalized.mp4`;
  annotatedVideo.src = `/stream/${uid}/normalized.mp4`;
  rawVideo.loop = false;
  annotatedVideo.loop = false;
  rawVideo.onplay = () => annotatedVideo.play();
  rawVideo.onpause = () => annotatedVideo.pause();
  rawVideo.onseeked = syncVideos;
  rawVideo.ontimeupdate = onTimeUpdate;
  annotatedVideo.onloadedmetadata = resizeCanvas;
  window.onresize = resizeCanvas;
  setPipelineProgress(6, "completed", `Loaded ${uid}`);
  appendEventLog({ phase: 6, status: "completed", message: `Loaded ${uid}`, phase_name: "Validate" });
  stopProgressWatch();
}

async function loadTracks(uid) {
  state.tracks = new Map();
  const clips = state.dataset?.videos?.[0]?.segments || [];
  await Promise.all(
    clips.map(async (clip) => {
      const clipName = clip.segment_id.replace("seg_", "clip_");
      try {
        const track = await fetch(`/api/video/${uid}/tracks/${clipName}.json`).then((res) => {
          if (!res.ok) throw new Error("track missing");
          return res.json();
        });
        state.tracks.set(clip.segment_id, track.frames || []);
      } catch {
        state.tracks.set(clip.segment_id, []);
      }
    })
  );
}

function syncVideos() {
  if (Math.abs(annotatedVideo.currentTime - rawVideo.currentTime) > 0.1) {
    annotatedVideo.currentTime = rawVideo.currentTime;
  }
}

function onTimeUpdate() {
  syncVideos();
  const clip = activeClip(rawVideo.currentTime);
  if (!clip) return;
  updatePanel(clip);
  drawOverlay(clip);
}

function activeClip(time) {
  const video = state.dataset?.videos?.[0];
  return video?.segments?.find((clip) => time >= clip.start_time && time <= clip.end_time) || video?.segments?.[0];
}

function updatePanel(clip) {
  document.getElementById("segment-title").textContent = `${clip.segment_id} (${clip.start_time.toFixed(1)} to ${clip.end_time.toFixed(1)} sec)`;
  document.getElementById("caption").textContent = clip.narration;
  const frames = state.tracks.get(clip.segment_id) || [];
  document.getElementById("annotation-source").textContent = frames.length
    ? `Overlay source: frame-derived track at ${frames.length} sampled frames`
    : "Overlay source: segment keyframe annotation";
  document.getElementById("verb").textContent = clip.verb || "-";
  document.getElementById("noun").textContent = clip.noun || "-";
  document.getElementById("tool").textContent = clip.tool || "none";
  setHand("left", clip.hands?.left);
  setHand("right", clip.hands?.right);
  const score = clip.qa_metrics?.flagged_for_review ? 0.55 : Math.max(0, Math.min(1, clip.qa_metrics?.avg_detection_confidence || 0));
  document.getElementById("quality-meter").value = score;
  document.getElementById("quality-score").textContent = `${Math.round(score * 100)}%`;
}

function setHand(side, hand) {
  document.getElementById(`${side}-contact`).textContent = hand?.contact_state || "none";
  document.getElementById(`${side}-grasp`).textContent = hand?.grasp_type || "none";
}

function resizeCanvas() {
  const rect = annotatedVideo.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
}

function drawOverlay(clip) {
  resizeCanvas();
  const annotation = nearestTrackFrame(clip) || clip;
  const rect = annotatedVideo.getBoundingClientRect();
  const meta = state.dataset?.videos?.[0]?.meta;
  const sourceWidth = annotatedVideo.videoWidth || meta?.resolution?.[0] || 1920;
  const sourceHeight = annotatedVideo.videoHeight || meta?.resolution?.[1] || 1080;
  const sx = rect.width / sourceWidth;
  const sy = rect.height / sourceHeight;
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.lineWidth = 2;
  ctx.font = "13px system-ui";

  (annotation.objects || []).forEach((obj) => {
    drawBox(obj.bbox_2d, sx, sy, "#1FAE4B", `${obj.label} ${Math.round(obj.confidence * 100)}%`);
  });
  Object.entries(annotation.hands || {}).forEach(([side, hand]) => {
    if (!hand) return;
    drawBox(hand.bbox_2d, sx, sy, "#111111", `${side} ${hand.contact_state}`);
    ctx.fillStyle = "#111111";
    (hand.keypoints_2d || []).forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * sx, y * sy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function nearestTrackFrame(clip) {
  const frames = state.tracks.get(clip.segment_id) || [];
  if (!frames.length) return null;
  const time = rawVideo.currentTime;
  return frames.reduce((best, frame) => {
    if (!best) return frame;
    return Math.abs(frame.time_sec - time) < Math.abs(best.time_sec - time) ? frame : best;
  }, null);
}

function drawBox(bbox, sx, sy, color, label) {
  const [x1, y1, x2, y2] = bbox;
  const x = x1 * sx;
  const y = y1 * sy;
  const w = (x2 - x1) * sx;
  const h = (y2 - y1) * sy;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.strokeRect(x, y, w, h);
  ctx.fillText(label, x + 4, Math.max(14, y - 6));
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("upload-input").files[0];
  if (!file) return;
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  clearCurrentVideo(`Uploading ${file.name}`);
  try {
    const response = await uploadWithProgress(file);
    state.uid = response.video_uid;
    document.getElementById("video-id").textContent = response.video_uid;
    setPipelineProgress(0, "queued", response.message || "Processing started");
    connectProgress(response.video_uid);
    await waitForDataset(response.video_uid);
    await loadVideo(response.video_uid);
  } catch (error) {
    setPipelineProgress(0, "failed", error.message || "Upload failed");
  } finally {
    button.disabled = false;
  }
});

boot();

function uploadWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?filename=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        const progress = document.getElementById("progress");
        progress.max = 100;
        progress.value = percent;
        document.getElementById("progress-text").textContent = `Uploading ${percent}%`;
        document.getElementById("active-phase-title").textContent = "Uploading video";
        document.getElementById("active-phase-backing").textContent = "Browser upload stream to FastAPI";
        document.getElementById("active-phase-message").textContent = `Uploading ${file.name}: ${percent}% complete. Processing starts after upload finishes.`;
      } else {
        document.getElementById("progress-text").textContent = "Uploading";
      }
    };
    xhr.onload = () => {
      let payload = {};
      try {
        payload = JSON.parse(xhr.responseText || "{}");
      } catch {
        payload = {};
      }
      if (xhr.status >= 200 && xhr.status < 300 && payload.video_uid) {
        resolve(payload);
      } else {
        reject(new Error(payload.detail || `Upload failed with HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed. Check that the server is still running."));
    xhr.send(file);
  });
}

async function waitForDataset(uid) {
  const started = Date.now();
  const timeoutMs = 45 * 60 * 1000;
  while (Date.now() - started < timeoutMs) {
    const ready = await fetch(`/api/video/${uid}/dataset.json`, { cache: "no-store" }).then((res) => res.ok).catch(() => false);
    if (ready) return;
    const progress = await refreshProgress(uid);
    const current = progress?.current || progress;
    if (current?.status === "failed") {
      throw new Error(current.message || "Pipeline failed");
    }
    await sleep(2500);
  }
  throw new Error("Processing timed out before dataset.json was ready.");
}

function connectProgress(uid) {
  stopProgressWatch();
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/progress/${uid}`);
  state.progressSocket = socket;
  socket.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    applyProgress(progress);
  };
  socket.onerror = () => {
    socket.close();
  };
  state.progressPoll = setInterval(() => refreshProgress(uid), 3000);
}

function stopProgressWatch() {
  if (state.progressSocket) {
    state.progressSocket.close();
    state.progressSocket = null;
  }
  if (state.progressPoll) {
    clearInterval(state.progressPoll);
    state.progressPoll = null;
  }
}

async function refreshProgress(uid) {
  const progress = await fetchJson(`/api/video/${uid}/progress`).catch(() => null);
  if (!progress) return null;
  applyProgress(progress);
  return progress;
}

function setPipelineProgress(phase, status, message) {
  const progress = document.getElementById("progress");
  progress.max = 6;
  progress.value = Math.max(0, Math.min(6, Number(phase || 0)));
  document.getElementById("progress-text").textContent = `${status}: ${message}`;
  updateTimeline({ phase, status, message });
}

function applyProgress(payload) {
  const history = Array.isArray(payload.history) ? payload.history : [];
  const current = payload.current || payload;
  mergeProgressEvents(history.length ? history : [current]);
  renderEventLog(state.progressEvents);
  setPipelineProgress(current.phase, current.status, current.message);
  updateTimeline(current);
  updateActivePhase(current);
}

function mergeProgressEvents(events) {
  events.forEach((event) => {
    const key = event.timestamp || `${event.phase}-${event.status}-${event.message}`;
    const exists = state.progressEvents.some((item) => (item.timestamp || `${item.phase}-${item.status}-${item.message}`) === key);
    if (!exists) {
      state.progressEvents.push(event);
    }
  });
  state.progressEvents.sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));
  state.progressEvents = state.progressEvents.slice(-200);
}

function updateActivePhase(event) {
  state.lastProgressAt = Date.now();
  state.currentProgressEvent = event;
  const phase = PHASES.find((item) => item.id === Number(event.phase));
  const phaseName = event.phase_name || phase?.title || (Number(event.phase) === 0 ? "Upload queued" : "Processing");
  document.getElementById("active-phase-title").textContent = `${phaseName}: ${event.status || "working"}`;
  document.getElementById("active-phase-backing").textContent = event.backed_by || phase?.backed || "EgoFlow pipeline";
  const message = event.message || phase?.detail || "Working";
  const messageNode = document.getElementById("active-phase-message");
  messageNode.dataset.baseMessage = message;
  messageNode.textContent = message;
}

function updateHeartbeat() {
  const event = state.currentProgressEvent;
  if (!event || !["working", "started", "queued"].includes(event.status)) return;
  const elapsed = Math.max(0, Math.floor((Date.now() - state.lastProgressAt) / 1000));
  const messageNode = document.getElementById("active-phase-message");
  const base = messageNode.dataset.baseMessage || event.message || "Working";
  if (elapsed >= 5) {
    messageNode.textContent = `${base} - still running (${elapsed}s since the last backend update). Long videos can stay in this step for a few minutes.`;
  } else {
    messageNode.textContent = base;
  }
}

function renderTimeline() {
  const timeline = document.getElementById("phase-timeline");
  timeline.innerHTML = "";
  PHASES.forEach((phase) => {
    const card = document.createElement("div");
    card.className = "phase-card";
    card.id = `phase-card-${phase.id}`;
    card.innerHTML = `
      <span class="phase-number">Phase ${phase.id}</span>
      <span class="phase-title">${phase.title}</span>
      <span class="phase-backed">${phase.backed}</span>
      <span class="phase-status" id="phase-status-${phase.id}">Waiting</span>
    `;
    timeline.appendChild(card);
  });
}

function updateTimeline(event) {
  const currentPhase = Number(event.phase || 0);
  const status = event.status || "waiting";
  const subProgress = Number.isFinite(Number(event.progress)) ? Number(event.progress) : status === "completed" ? 100 : 20;
  PHASES.forEach((phase) => {
    const card = document.getElementById(`phase-card-${phase.id}`);
    const label = document.getElementById(`phase-status-${phase.id}`);
    if (!card || !label) return;
    card.classList.remove("active", "done", "failed");
    if (phase.id < currentPhase || (phase.id === currentPhase && status === "completed")) {
      card.classList.add("done");
      card.style.setProperty("--phase-progress", "100%");
      label.textContent = "Completed";
    } else if (phase.id === currentPhase) {
      card.classList.add(status === "failed" ? "failed" : "active");
      card.style.setProperty("--phase-progress", `${Math.max(8, Math.min(100, subProgress))}%`);
      label.textContent = status === "failed" ? "Failed" : "Running";
    } else {
      card.style.setProperty("--phase-progress", "0%");
      label.textContent = "Waiting";
    }
  });
}

function renderEventLog(events) {
  const log = document.getElementById("event-log");
  log.innerHTML = "";
  events.slice(-16).reverse().forEach((event) => appendEventLog(event, false));
}

function appendEventLog(event, prepend = true) {
  mergeProgressEvents([event]);
  const log = document.getElementById("event-log");
  if (!log) return;
  const item = document.createElement("li");
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "now";
  const phase = event.phase_name || (event.phase ? `Phase ${event.phase}` : "Upload");
  item.textContent = `${time} - ${phase} - ${event.status || "working"}: ${event.message || ""}`;
  if (prepend) {
    log.prepend(item);
  } else {
    log.appendChild(item);
  }
  while (log.children.length > 16) {
    log.removeChild(log.lastChild);
  }
}

function clearCurrentVideo(message) {
  rawVideo.pause();
  annotatedVideo.pause();
  rawVideo.removeAttribute("src");
  annotatedVideo.removeAttribute("src");
  rawVideo.load();
  annotatedVideo.load();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  state.dataset = null;
  state.tracks = new Map();
  state.progressEvents = [];
  document.getElementById("video-id").textContent = "Upload in progress";
  document.getElementById("download-json").removeAttribute("href");
  document.getElementById("download-json").classList.add("disabled");
  document.getElementById("segment-title").textContent = "Processing new video";
  document.getElementById("caption").textContent = message;
  document.getElementById("annotation-source").textContent = "Overlay source: waiting for annotations";
  document.getElementById("verb").textContent = "-";
  document.getElementById("noun").textContent = "-";
  document.getElementById("tool").textContent = "-";
  setHand("left", null);
  setHand("right", null);
  document.getElementById("quality-meter").value = 0;
  document.getElementById("quality-score").textContent = "0%";
  renderTimeline();
  renderEventLog([{ phase: 0, status: "uploading", message }]);
}

function fetchJson(url) {
  return fetch(url, { cache: "no-store" }).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
