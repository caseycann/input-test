import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const modelUrl = (variant) =>
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task`;

// The 33 BlazePose landmarks, in index order.
const LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
  "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
  "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
  "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
  "right_heel", "left_foot_index", "right_foot_index",
];

const DATA_UPDATE_MS = 100; // throttle DOM updates for the data panel to ~10 Hz

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const drawing = new DrawingUtils(ctx);
const statusEl = document.getElementById("status");
const toggleBtn = document.getElementById("toggle");
const modelSel = document.getElementById("model");
const numPosesSel = document.getElementById("numPoses");
const tableEl = document.getElementById("table");
const jsonEl = document.getElementById("json");
const copyBtn = document.getElementById("copy");
const fpsEl = document.getElementById("fps");
const inferenceEl = document.getElementById("inference");
const poseCountEl = document.getElementById("poseCount");

let vision;
let landmarker;
let stream = null;
let running = false;
let lastVideoTime = -1;
let lastDataUpdate = 0;
let latestResult = null;
let frameTimes = [];

function setStatus(text) {
  statusEl.textContent = text ?? "";
  statusEl.hidden = !text;
}

async function createLandmarker() {
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: modelUrl(modelSel.value), delegate },
    runningMode: "VIDEO",
    numPoses: Number(numPosesSel.value),
  });
  landmarker?.close();
  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    landmarker = await PoseLandmarker.createFromOptions(vision, options("CPU"));
  }
}

async function reloadModel() {
  toggleBtn.disabled = true;
  const wasRunning = running;
  running = false;
  setStatus("Loading model…");
  await createLandmarker();
  lastVideoTime = -1;
  toggleBtn.disabled = false;
  if (wasRunning) {
    setStatus(null);
    running = true;
    requestAnimationFrame(loop);
  } else {
    setStatus("Press “Start camera” to begin");
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    setStatus(`Camera unavailable: ${err.message}`);
    return;
  }
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  document.querySelector(".stage").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  setStatus(null);
  toggleBtn.textContent = "Stop camera";
  running = true;
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  toggleBtn.textContent = "Start camera";
  setStatus("Press “Start camera” to begin");
}

function loop(now) {
  if (!running) return;

  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    const t0 = performance.now();
    const result = landmarker.detectForVideo(video, t0);
    const inferenceMs = performance.now() - t0;

    draw(result);
    latestResult = result;

    frameTimes.push(now);
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();

    if (now - lastDataUpdate > DATA_UPDATE_MS) {
      lastDataUpdate = now;
      fpsEl.textContent = `${frameTimes.length} fps`;
      inferenceEl.textContent = `${inferenceMs.toFixed(1)} ms`;
      renderData(result);
    }
  }

  requestAnimationFrame(loop);
}

function draw(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const landmarks of result.landmarks) {
    drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: "#4ade80",
      lineWidth: 4,
    });
    drawing.drawLandmarks(landmarks, {
      color: "#f472b6",
      fillColor: "#0f1115",
      lineWidth: 2,
      radius: (d) => DrawingUtils.lerp(d.from?.z ?? 0, -0.15, 0.1, 6, 2),
    });
  }
}

const round = (n, d = 4) => (n == null ? null : Number(n.toFixed(d)));

function toPlain(result) {
  return {
    timestamp: Math.round(performance.now()),
    poses: result.landmarks.map((lms, p) => ({
      // x, y normalized to [0,1] of the image; z is depth relative to the hips (smaller = closer)
      landmarks: lms.map((l, i) => ({
        name: LANDMARK_NAMES[i],
        x: round(l.x), y: round(l.y), z: round(l.z), visibility: round(l.visibility, 3),
      })),
      // real-world 3D coordinates in meters, origin at the center of the hips
      worldLandmarks: (result.worldLandmarks[p] ?? []).map((l, i) => ({
        name: LANDMARK_NAMES[i],
        x: round(l.x), y: round(l.y), z: round(l.z),
      })),
    })),
  };
}

function view() {
  return document.querySelector('input[name="view"]:checked').value;
}

function renderData(result) {
  const n = result.landmarks.length;
  poseCountEl.textContent = `${n} pose${n === 1 ? "" : "s"}`;

  if (view() === "json") {
    jsonEl.textContent = JSON.stringify(toPlain(result), null, 2);
    return;
  }

  if (!n) {
    tableEl.innerHTML = `<div class="empty">No pose detected.</div>`;
    return;
  }

  const fmt = (v, d = 4) => (v == null ? "–" : v.toFixed(d));
  let rows = "";
  result.landmarks.forEach((lms, p) => {
    const world = result.worldLandmarks[p] ?? [];
    lms.forEach((l, i) => {
      const w = world[i] ?? {};
      const low = (l.visibility ?? 1) < 0.5 ? " class=\"low\"" : "";
      const brk = p > 0 && i === 0 ? ` class="pose-break"` : "";
      rows += `<tr${brk}><td>${p}</td><td>${i} ${LANDMARK_NAMES[i]}</td>` +
        `<td${low}>${fmt(l.x)}</td><td${low}>${fmt(l.y)}</td><td${low}>${fmt(l.z)}</td>` +
        `<td${low}>${fmt(l.visibility, 3)}</td>` +
        `<td${low}>${fmt(w.x)}</td><td${low}>${fmt(w.y)}</td><td${low}>${fmt(w.z)}</td></tr>`;
    });
  });

  tableEl.innerHTML =
    `<table><thead><tr><th>pose</th><th>landmark</th>` +
    `<th>x</th><th>y</th><th>z</th><th>vis</th>` +
    `<th>world x (m)</th><th>world y (m)</th><th>world z (m)</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>`;
}

document.querySelectorAll('input[name="view"]').forEach((el) =>
  el.addEventListener("change", () => {
    const isJson = view() === "json";
    jsonEl.hidden = !isJson;
    tableEl.hidden = isJson;
    if (latestResult) renderData(latestResult);
  })
);

copyBtn.addEventListener("click", async () => {
  if (!latestResult) return;
  await navigator.clipboard.writeText(JSON.stringify(toPlain(latestResult), null, 2));
  copyBtn.textContent = "Copied";
  setTimeout(() => (copyBtn.textContent = "Copy JSON"), 1200);
});

toggleBtn.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
modelSel.addEventListener("change", reloadModel);
numPosesSel.addEventListener("change", reloadModel);

tableEl.innerHTML = `<div class="empty">Waiting for camera…</div>`;
vision = await FilesetResolver.forVisionTasks(WASM_URL);
await reloadModel();
