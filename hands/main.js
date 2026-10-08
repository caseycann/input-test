import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task";

// MediaPipe labels handedness as if the image were already mirrored (selfie view).
// We feed it the raw, un-mirrored camera frame, so its labels come out swapped.
const SWAP_HANDEDNESS = true;

// Built-in gestures from the canned model, plus "Pinch", which we compute from landmarks.
const INPUTS = [
  { id: "Closed_Fist", label: "Fist", emoji: "✊" },
  { id: "Open_Palm", label: "Open palm", emoji: "🖐️" },
  { id: "Pointing_Up", label: "Point up", emoji: "☝️" },
  { id: "Victory", label: "Victory", emoji: "✌️" },
  { id: "Thumb_Up", label: "Thumb up", emoji: "👍" },
  { id: "Thumb_Down", label: "Thumb down", emoji: "👎" },
  { id: "ILoveYou", label: "I love you", emoji: "🤟" },
  { id: "Pinch", label: "Pinch", emoji: "🤏" },
];

// Pinch = thumb tip to index tip distance, divided by palm size (wrist to middle knuckle)
// so it works at any distance from the camera.
const PINCH_ON = 0.3; // ratio at or below this counts as a pinch
const PINCH_OPEN = 0.8; // ratio at or above this is "fully open" for the strength value

// The 21 hand landmarks, in index order.
const LANDMARK_NAMES = [
  "wrist",
  "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
  "index_mcp", "index_pip", "index_dip", "index_tip",
  "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
  "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
  "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
];

const DATA_UPDATE_MS = 100; // throttle DOM updates for the data panel to ~10 Hz

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const drawing = new DrawingUtils(ctx);
const statusEl = document.getElementById("status");
const toggleBtn = document.getElementById("toggle");
const minScoreEl = document.getElementById("minScore");
const minScoreVal = document.getElementById("minScoreVal");
const jsonEl = document.getElementById("json");
const copyBtn = document.getElementById("copy");
const fpsEl = document.getElementById("fps");
const inferenceEl = document.getElementById("inference");

let recognizer;
let stream = null;
let running = false;
let lastVideoTime = -1;
let lastDataUpdate = 0;
let latestHands = [];
let frameTimes = [];

// Build the input buttons for each hand panel.
const panels = {};
for (const panel of document.querySelectorAll(".hand")) {
  const inputsEl = panel.querySelector(".inputs");
  const buttons = {};
  for (const input of INPUTS) {
    const btn = document.createElement("button");
    btn.className = "input-btn";
    btn.setAttribute("aria-pressed", "false");
    btn.innerHTML =
      `<span class="emoji">${input.emoji}</span><span>${input.label}</span><span class="pct">–</span>`;
    inputsEl.append(btn);
    buttons[input.id] = btn;
  }
  panels[panel.dataset.hand] = { el: panel, state: panel.querySelector(".hand-state"), buttons };
}

function setStatus(text) {
  statusEl.textContent = text ?? "";
  statusEl.hidden = !text;
}

function updateLabels() {
  minScoreVal.textContent = (minScoreEl.value / 100).toFixed(2);
}

async function createRecognizer() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
  try {
    recognizer = await GestureRecognizer.createFromOptions(vision, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    recognizer = await GestureRecognizer.createFromOptions(vision, options("CPU"));
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
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  latestHands = [];
  renderPanels([]);
  toggleBtn.textContent = "Start camera";
  setStatus("Press “Start camera” to begin");
}

function loop(now) {
  if (!running) return;

  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    const t0 = performance.now();
    const result = recognizer.recognizeForVideo(video, t0);
    const inferenceMs = performance.now() - t0;

    const hands = toHands(result);
    latestHands = hands;
    draw(hands);
    renderPanels(hands);

    frameTimes.push(now);
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();

    if (now - lastDataUpdate > DATA_UPDATE_MS) {
      lastDataUpdate = now;
      fpsEl.textContent = `${frameTimes.length} fps`;
      inferenceEl.textContent = `${inferenceMs.toFixed(1)} ms`;
      jsonEl.textContent = JSON.stringify(toPlain(hands), null, 2);
    }
  }

  requestAnimationFrame(loop);
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const clamp01 = (v) => Math.min(1, Math.max(0, v));

// Flatten the recognizer result into one object per hand, with the inputs we care about.
function toHands(result) {
  return result.landmarks.map((landmarks, i) => {
    const hand = result.handedness[i]?.[0];
    let side = hand?.categoryName ?? "Unknown";
    if (SWAP_HANDEDNESS) side = side === "Left" ? "Right" : side === "Right" ? "Left" : side;

    const gesture = result.gestures[i]?.[0];
    // Use world landmarks (meters) for pinch so it's independent of image aspect ratio.
    const world = result.worldLandmarks[i] ?? landmarks;
    const pinchRatio = dist(world[4], world[8]) / dist(world[0], world[9]);

    return {
      side,
      handednessScore: hand?.score ?? 0,
      gesture: gesture?.categoryName ?? "None",
      gestureScore: gesture?.score ?? 0,
      pinchRatio,
      pinchStrength: clamp01((PINCH_OPEN - pinchRatio) / (PINCH_OPEN - PINCH_ON)),
      landmarks,
      worldLandmarks: result.worldLandmarks[i] ?? [],
    };
  });
}

// Which inputs are "on" for a hand, after applying the confidence slider.
function activeInputs(hand) {
  const on = new Set();
  if (hand.gesture !== "None" && hand.gestureScore >= minScoreEl.value / 100) on.add(hand.gesture);
  if (hand.pinchRatio <= PINCH_ON) on.add("Pinch");
  return on;
}

function renderPanels(hands) {
  for (const [side, panel] of Object.entries(panels)) {
    const hand = hands.find((h) => h.side === side);
    panel.el.classList.toggle("present", !!hand);
    panel.state.textContent = hand ? `Detected · ${hand.handednessScore.toFixed(2)}` : "Not detected";
    const on = hand ? activeInputs(hand) : new Set();

    for (const [id, btn] of Object.entries(panel.buttons)) {
      btn.setAttribute("aria-pressed", String(on.has(id)));
      let pct = "–";
      if (hand && id === "Pinch") pct = hand.pinchStrength.toFixed(2);
      else if (hand && id === hand.gesture) pct = hand.gestureScore.toFixed(2);
      btn.querySelector(".pct").textContent = pct;
    }
  }
}

function draw(hands) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  ctx.clearRect(0, 0, w, h);

  for (const hand of hands) {
    const on = activeInputs(hand);
    const color = on.size ? "#facc15" : "#4ade80";

    // Draw the skeleton mirrored so it lines up with the mirrored video.
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    drawing.drawConnectors(hand.landmarks, GestureRecognizer.HAND_CONNECTIONS, { color, lineWidth: 4 });
    drawing.drawLandmarks(hand.landmarks, { color: "#f472b6", fillColor: "#0f1115", lineWidth: 2, radius: 4 });
    ctx.restore();

    // Label above the wrist, drawn unmirrored so the text reads correctly.
    const wrist = hand.landmarks[0];
    const label = [hand.side, ...[...on].map((id) => INPUTS.find((i) => i.id === id).label)].join(" · ");
    const fontPx = Math.round(h * 0.035);
    ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const x = (1 - wrist.x) * w;
    const y = Math.min(wrist.y * h + fontPx * 0.6, h - fontPx * 1.6);
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(15, 17, 21, 0.75)";
    ctx.fillRect(x - tw / 2 - 8, y - 4, tw + 16, fontPx + 8);
    ctx.fillStyle = color;
    ctx.fillText(label, x, y);
  }
}

const round = (n, d = 4) => (n == null ? null : Number(n.toFixed(d)));

function toPlain(hands) {
  return {
    timestamp: Math.round(performance.now()),
    hands: hands.map((h) => ({
      side: h.side,
      handednessScore: round(h.handednessScore, 3),
      gesture: h.gesture,
      gestureScore: round(h.gestureScore, 3),
      pinchRatio: round(h.pinchRatio, 3),
      pinchStrength: round(h.pinchStrength, 3),
      activeInputs: [...activeInputs(h)],
      // x, y normalized to [0,1] of the image (un-mirrored); z is depth relative to the wrist
      landmarks: h.landmarks.map((l, i) => ({ name: LANDMARK_NAMES[i], x: round(l.x), y: round(l.y), z: round(l.z) })),
      // real-world 3D coordinates in meters, origin at the hand's approximate center
      worldLandmarks: h.worldLandmarks.map((l, i) => ({ name: LANDMARK_NAMES[i], x: round(l.x), y: round(l.y), z: round(l.z) })),
    })),
  };
}

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(JSON.stringify(toPlain(latestHands), null, 2));
  copyBtn.textContent = "Copied";
  setTimeout(() => (copyBtn.textContent = "Copy JSON"), 1200);
});

toggleBtn.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
minScoreEl.addEventListener("input", updateLabels);
updateLabels();
renderPanels([]);

await createRecognizer();
toggleBtn.disabled = false;
setStatus("Press “Start camera” to begin");
