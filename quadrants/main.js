// Splits the camera view into a grid of zones (2×2, 4×2 or 4×4) and lights a
// button when enough of a zone is darker than the chosen grey.

// Frames are analyzed at low resolution: it's plenty for "is something dark here"
// and keeps the per-pixel loop cheap.
const ANALYSIS_WIDTH = 160;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const toggleBtn = document.getElementById("toggle");
const thresholdEl = document.getElementById("threshold");
const thresholdVal = document.getElementById("thresholdVal");
const swatch = document.getElementById("swatch");
const coverageEl = document.getElementById("coverage");
const coverageVal = document.getElementById("coverageVal");
const showMaskEl = document.getElementById("showMask");
const zonesEl = document.getElementById("zones");
const buttonsEl = document.getElementById("buttons");

let cols = 2;
let rows = 2;
let zoneButtons = [];

// Rebuild the button row to match the grid, laid out in the same shape as the zones.
function buildButtons() {
  [cols, rows] = zonesEl.value.split("x").map(Number);
  buttonsEl.style.setProperty("--cols", cols);
  buttonsEl.classList.toggle("dense", cols * rows > 4);
  buttonsEl.innerHTML = "";
  zoneButtons = Array.from({ length: cols * rows }, (_, z) => {
    const btn = document.createElement("button");
    btn.className = "quad-btn";
    btn.setAttribute("aria-pressed", "false");
    btn.innerHTML = `<span class="name">${z + 1}</span><span class="pct">0% dark</span>`;
    buttonsEl.append(btn);
    return btn;
  });
}

// Small canvas holding the mirrored, downscaled frame we read pixels from.
const analysis = document.createElement("canvas");
const actx = analysis.getContext("2d", { willReadFrequently: true });
// Same size as `analysis`; holds the tinted dark-pixel mask, scaled up onto the overlay.
const mask = document.createElement("canvas");
const mctx = mask.getContext("2d");

let stream = null;
let running = false;

// "60% grey" = 60% ink, i.e. #666 = brightness 40%. A pixel is dark if its brightness is below that.
function darkLimit() {
  return 255 * (1 - thresholdEl.value / 100);
}

function updateLabels() {
  const g = Math.round(darkLimit());
  const hex = g.toString(16).padStart(2, "0");
  thresholdVal.textContent = `${thresholdEl.value}% grey`;
  swatch.style.background = `#${hex}${hex}${hex}`;
  swatch.title = `#${hex}${hex}${hex}`;
  coverageVal.textContent = `${coverageEl.value}% of zone`;
}

function setStatus(text) {
  statusEl.textContent = text ?? "";
  statusEl.hidden = !text;
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

  const { videoWidth: w, videoHeight: h } = video;
  document.querySelector(".stage").style.aspectRatio = `${w} / ${h}`;
  analysis.width = mask.width = ANALYSIS_WIDTH;
  analysis.height = mask.height = Math.round((ANALYSIS_WIDTH * h) / w);

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
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const zeros = zoneButtons.map(() => 0);
  setButtons(zeros, zeros.map(() => false));
  toggleBtn.textContent = "Start camera";
  setStatus("Press “Start camera” to begin");
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    const fractions = analyze();
    const active = fractions.map((f) => f * 100 >= Number(coverageEl.value));
    drawOverlay(active);
    setButtons(fractions, active);
  }
  requestAnimationFrame(loop);
}

// Returns the fraction of dark pixels in each zone, row by row from the top left.
function analyze() {
  const { width: w, height: h } = analysis;
  actx.save();
  actx.scale(-1, 1); // mirror to match what the user sees
  actx.drawImage(video, -w, 0, w, h);
  actx.restore();

  const frame = actx.getImageData(0, 0, w, h);
  const px = frame.data;
  const maskImg = mctx.createImageData(w, h);
  const mpx = maskImg.data;
  const limit = darkLimit();
  const n = cols * rows;
  const dark = new Array(n).fill(0);
  const total = new Array(n).fill(0);
  // Precompute each column's zone column, so the inner loop is just a lookup.
  const colOf = Array.from({ length: w }, (_, x) => Math.min(cols - 1, Math.floor((x * cols) / w)));

  for (let y = 0; y < h; y++) {
    const rowBase = Math.min(rows - 1, Math.floor((y * rows) / h)) * cols;
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      // Rec. 709 luma
      const lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
      const q = rowBase + colOf[x];
      total[q]++;
      if (lum < limit) {
        dark[q]++;
        mpx[i] = 250; mpx[i + 1] = 204; mpx[i + 2] = 21; mpx[i + 3] = 110;
      }
    }
  }

  mctx.putImageData(maskImg, 0, 0);
  return dark.map((d, q) => d / total[q]);
}

function drawOverlay(active) {
  // Keep the overlay's backing store matched to its on-screen size for crisp lines.
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(overlay.clientWidth * dpr);
  const ch = Math.round(overlay.clientHeight * dpr);
  if (overlay.width !== cw || overlay.height !== ch) {
    overlay.width = cw;
    overlay.height = ch;
  }

  octx.clearRect(0, 0, cw, ch);

  if (showMaskEl.checked) {
    octx.imageSmoothingEnabled = false;
    octx.drawImage(mask, 0, 0, cw, ch);
  }

  const zw = cw / cols;
  const zh = ch / rows;
  const fontPx = Math.round(Math.min(zw, zh) * 0.12);
  octx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
  octx.textBaseline = "top";

  active.forEach((on, q) => {
    const x = (q % cols) * zw;
    const y = Math.floor(q / cols) * zh;
    octx.fillStyle = on ? "#facc15" : "rgba(255, 255, 255, 0.7)";
    octx.fillText(String(q + 1), x + fontPx * 0.5, y + fontPx * 0.4);
    if (!on) return;
    octx.fillStyle = "rgba(250, 204, 21, 0.15)";
    octx.fillRect(x, y, zw, zh);
    octx.strokeStyle = "#facc15";
    octx.lineWidth = 6 * dpr;
    octx.strokeRect(x + 3 * dpr, y + 3 * dpr, zw - 6 * dpr, zh - 6 * dpr);
  });

  octx.strokeStyle = "rgba(255, 255, 255, 0.8)";
  octx.lineWidth = 2 * dpr;
  octx.setLineDash([10 * dpr, 8 * dpr]);
  octx.beginPath();
  for (let c = 1; c < cols; c++) {
    octx.moveTo(c * zw, 0); octx.lineTo(c * zw, ch);
  }
  for (let r = 1; r < rows; r++) {
    octx.moveTo(0, r * zh); octx.lineTo(cw, r * zh);
  }
  octx.stroke();
  octx.setLineDash([]);
}

function setButtons(fractions, active) {
  zoneButtons.forEach((btn, q) => {
    btn.setAttribute("aria-pressed", String(active[q]));
    btn.querySelector(".pct").textContent = `${Math.round(fractions[q] * 100)}% dark`;
  });
}

toggleBtn.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
thresholdEl.addEventListener("input", updateLabels);
coverageEl.addEventListener("input", updateLabels);
zonesEl.addEventListener("change", buildButtons);
buildButtons();
updateLabels();
