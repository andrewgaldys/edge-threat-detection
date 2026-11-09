// Browser edition of the Edge Threat Detection dashboard.
//
// GitHub Pages can only serve files, so this page cannot talk to the Python
// pipelines. Instead it does the same job on the visitor's own device: the
// same YOLOv8n model, confidence limit, and 30-frame persistence rule. Video
// stays in this browser, and the Content Security Policy in index.html stops
// the page from sending data to any other site.

const INPUT_SIZE = 320;
const CONFIDENCE = 0.45;
const OVERLAP = 0.5;
const PERSISTENCE_FRAMES = 30;
const GRACE_FRAMES = 2;
const MAX_FRAME_WIDTH = 640;
const MAX_ROWS = 200;
const MAX_SNAPSHOTS = 6;
const TRACK_MATCH_OVERLAP = 0.2;
const TRACK_MAX_MISSES = 15;

// COCO class numbers. COCO has no drone class, so "airplane" stands in for it,
// exactly as CLASS_ALIASES does in the Python configuration.
const TARGETS = new Map([
  [0, "person"],
  [2, "car"],
  [4, "drone"],
  [16, "dog"],
]);

// One cycle of the demonstration scene: look at an empty area, tilt down to
// bring people into view, hold them long enough to breach, then tilt away.
const SCENE = { left: 40, width: 640, height: 370, clearTop: 8, targetTop: 400 };
const CYCLE = { clear: 3, tilt: 1, hold: 5 };

const COLORS = {
  clear: "rgb(80, 200, 80)",
  tracking: "rgb(255, 190, 0)",
  breach: "rgb(255, 60, 60)",
  backing: "rgb(20, 20, 20)",
  zone: "rgb(235, 235, 235)",
};

const STATES = {
  loading: ["○", "Loading", "warn"],
  running: ["●", "Running", "good"],
  simulated: ["●", "Simulated", "good"],
  stopped: ["■", "Stopped", "bad"],
};

const byId = (id) => document.getElementById(id);
const view = byId("view");
const viewContext = view.getContext("2d");
const raw = document.createElement("canvas");
const rawContext = raw.getContext("2d");
const input = document.createElement("canvas");
input.width = INPUT_SIZE;
input.height = INPUT_SIZE;
const inputContext = input.getContext("2d", { willReadFrequently: true });

let activeSource = null;
let detectorReady = false;
let inferenceBusy = false;
let generation = 0;
let letterbox = { scale: 1, dx: 0, dy: 0 };
let detections = [];
let zone = null;
let draggingFrom = null;
let framesPerSecond = 0;
let lastResultAt = 0;
let lastMilliseconds = 0;
const counts = { HIGH: 0, MEDIUM: 0 };
const watching = new Set(TARGETS.values());
const snapshots = [];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  // Alert text is only ever inserted as text, so it can never become markup.
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------------------------------------------------------------------------
// Persistence and tracking: ports of the Python pipeline's logic
// ---------------------------------------------------------------------------
class PersistenceMonitor {
  constructor() {
    this.streaks = new Map();
  }

  update(labels) {
    const present = new Set(labels);
    const breaches = [];
    for (const label of [...present].sort()) {
      const streak =
        this.streaks.get(label) ?? { frames: 0, missed: 0, alerted: false };
      this.streaks.set(label, streak);
      streak.frames += 1;
      streak.missed = 0;
      if (streak.frames > PERSISTENCE_FRAMES && !streak.alerted) {
        streak.alerted = true;
        breaches.push([label, streak.frames]);
      }
    }
    for (const [label, streak] of this.streaks) {
      if (!present.has(label)) {
        streak.missed += 1;
        if (streak.missed > GRACE_FRAMES) this.streaks.delete(label);
      }
    }
    return breaches;
  }

  isBreached(label) {
    return this.streaks.get(label)?.alerted === true;
  }

  reset() {
    this.streaks.clear();
  }
}

function overlapOf(a, b) {
  const width = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const height = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (width <= 0 || height <= 0) return 0;
  const shared = width * height;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return shared / (areaA + areaB - shared);
}

// A small overlap-based tracker that gives each object a stable number.
class BoxTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(found) {
    const pairs = [];
    found.forEach((item, itemIndex) => {
      this.tracks.forEach((track, trackIndex) => {
        const amount = item.label === track.label ? overlapOf(item, track) : 0;
        if (amount >= TRACK_MATCH_OVERLAP) {
          pairs.push([amount, itemIndex, trackIndex]);
        }
      });
    });
    pairs.sort((a, b) => b[0] - a[0]);

    const usedItems = new Set();
    const usedTracks = new Set();
    for (const [, itemIndex, trackIndex] of pairs) {
      if (!usedItems.has(itemIndex) && !usedTracks.has(trackIndex)) {
        usedItems.add(itemIndex);
        usedTracks.add(trackIndex);
        const track = this.tracks[trackIndex];
        Object.assign(track, found[itemIndex], { missed: 0 });
        found[itemIndex].trackId = track.id;
      }
    }

    this.tracks.forEach((track, index) => {
      if (!usedTracks.has(index)) track.missed += 1;
    });
    this.tracks = this.tracks.filter((t) => t.missed <= TRACK_MAX_MISSES);

    found.forEach((item, index) => {
      if (!usedItems.has(index)) {
        item.trackId = this.nextId;
        this.tracks.push({ ...item, id: this.nextId, missed: 0 });
        this.nextId += 1;
      }
    });
    return found;
  }
}

const monitor = new PersistenceMonitor();
const tracker = new BoxTracker();

// ---------------------------------------------------------------------------
// Video sources
// ---------------------------------------------------------------------------
class DemoScene {
  constructor(image) {
    this.image = image;
    this.simulated = true;
  }

  size() {
    return [SCENE.width, SCENE.height];
  }

  top(seconds) {
    const { clear, tilt, hold } = CYCLE;
    const position = seconds % (clear + tilt + hold + tilt);
    const travel = SCENE.targetTop - SCENE.clearTop;
    if (position < clear) return SCENE.clearTop;
    if (position < clear + tilt) {
      return SCENE.clearTop + travel * ((position - clear) / tilt);
    }
    if (position < clear + tilt + hold) return SCENE.targetTop;
    return SCENE.targetTop - travel * ((position - clear - tilt - hold) / tilt);
  }

  capture(context, now) {
    const seconds = now / 1000;
    // A slight sway, so that a held scene still reads as a live feed.
    const top = this.top(seconds) + 8 * Math.sin(seconds * 2.8) + 8;
    context.drawImage(
      this.image,
      SCENE.left,
      top,
      SCENE.width,
      SCENE.height,
      0,
      0,
      SCENE.width,
      SCENE.height
    );
    return true;
  }

  close() {}
}

class LiveCamera {
  constructor(stream, video) {
    this.stream = stream;
    this.video = video;
    this.simulated = false;
    this.label = stream.getVideoTracks()[0]?.label ?? "Camera";
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (activeSource === this) feedLost();
    });
  }

  static async open(deviceId) {
    const size = { width: { ideal: 640 }, height: { ideal: 480 } };
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: deviceId ? { ...size, deviceId: { exact: deviceId } } : size,
    });
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    return new LiveCamera(stream, video);
  }

  size() {
    const scale = Math.min(1, MAX_FRAME_WIDTH / this.video.videoWidth);
    return [
      Math.round(this.video.videoWidth * scale),
      Math.round(this.video.videoHeight * scale),
    ];
  }

  capture(context) {
    if (this.video.readyState < 2) return false;
    const [width, height] = this.size();
    if (width !== raw.width || height !== raw.height) setFrameSize(width, height);
    context.drawImage(this.video, 0, 0, width, height);
    return true;
  }

  close() {
    this.stream.getTracks().forEach((track) => track.stop());
  }
}

function setFrameSize(width, height) {
  raw.width = width;
  raw.height = height;
  view.width = width;
  view.height = height;
}

// Called whenever continuity is broken: nothing is known about what happened
// in the gap, so a streak that spanned it would not be truly consecutive.
function resetTracking() {
  generation += 1;
  detections = [];
  monitor.reset();
  tracker.reset();
}

async function useSource(next) {
  activeSource?.close();
  activeSource = next;
  resetTracking();
  const [width, height] = next.size();
  setFrameSize(width, height);
  renderNotice();
  pump();
}

function feedLost() {
  addAlert(
    "VISUAL",
    "MEDIUM",
    "Video feed lost: the camera stopped delivering frames. " +
      "The demonstration scene has been restored."
  );
  byId("source").value = "demo";
  useSource(demoScene);
}

// Prefers the computer's own camera, because a nearby phone acting as a
// webcam can otherwise become the default without any warning.
function preferredCamera(cameras) {
  return cameras.find((c) => /facetime|built-?in|integrated/i.test(c.label));
}

async function fillCameraList(selected) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  const select = byId("source");
  select.replaceChildren(new Option("Demonstration scene", "demo"));
  for (const camera of cameras) {
    select.add(new Option(camera.label || "Camera", camera.deviceId));
  }
  select.value = selected;
  return cameras;
}

async function chooseSource(value) {
  try {
    if (value === "demo") {
      await useSource(demoScene);
    } else if (value === "camera") {
      let camera = await LiveCamera.open(null);
      const current = camera.stream.getVideoTracks()[0].getSettings().deviceId;
      const cameras = await fillCameraList(current);
      const preferred = preferredCamera(cameras);
      if (preferred && preferred.deviceId !== current) {
        camera.close();
        camera = await LiveCamera.open(preferred.deviceId);
        byId("source").value = preferred.deviceId;
      }
      await useSource(camera);
    } else {
      await useSource(await LiveCamera.open(value));
    }
  } catch (error) {
    byId("source").value = "demo";
    await useSource(demoScene);
    const denied = error.name === "NotAllowedError";
    renderNotice(
      denied
        ? "Camera permission was denied, so the demonstration scene is " +
            "showing. Allow camera access for this site to use a camera."
        : "No camera could be opened, so the demonstration scene is showing."
    );
  }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
function drawLabel(context, text, x, y, size, color) {
  context.font = `600 ${size}px system-ui, sans-serif`;
  const pad = Math.max(3, size / 4);
  const width = context.measureText(text).width + pad * 2;
  const height = size + pad * 2;
  const left = Math.max(0, Math.min(x, view.width - width));
  const top = Math.max(0, Math.min(y, view.height - height));
  context.fillStyle = COLORS.backing;
  context.fillRect(left, top, width, height);
  context.fillStyle = color;
  context.textBaseline = "middle";
  context.fillText(text, left + pad, top + height / 2);
}

function renderView() {
  viewContext.drawImage(raw, 0, 0);
  const size = Math.max(11, Math.round(view.width / 45));
  viewContext.lineWidth = Math.max(2, view.width / 320);

  if (zone) {
    viewContext.strokeStyle = COLORS.zone;
    viewContext.strokeRect(
      zone.left * view.width,
      zone.top * view.height,
      (zone.right - zone.left) * view.width,
      (zone.bottom - zone.top) * view.height
    );
  }

  let statusColor = detections.length ? COLORS.tracking : COLORS.clear;
  for (const item of detections) {
    const breached = monitor.isBreached(item.label);
    const color = breached ? COLORS.breach : COLORS.tracking;
    if (breached) statusColor = COLORS.breach;
    viewContext.strokeStyle = color;
    viewContext.strokeRect(item.x1, item.y1, item.x2 - item.x1, item.y2 - item.y1);
    drawLabel(
      viewContext,
      `${item.label} #${item.trackId} ${item.score.toFixed(2)}`,
      item.x1,
      // Kept below the status strip, which is drawn over the top of the frame.
      Math.max(item.y1, size * 1.6),
      Math.round(size * 0.85),
      color
    );
  }
  drawLabel(
    viewContext,
    `SYSTEM ACTIVE / THREATS: ${detections.length}`,
    0,
    0,
    size,
    statusColor
  );
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------
const worker = new Worker(new URL("./detector-worker.js", import.meta.url), {
  type: "module",
});

worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    const megabytes = (data.received / 1048576).toFixed(0);
    byId("video-note").textContent = `Downloading the model: ${megabytes} of 12 MB`;
  } else if (data.type === "ready") {
    detectorReady = true;
    byId("video-note").hidden = true;
    renderDetectorTiles();
    pump();
  } else if (data.type === "result") {
    onResult(data);
  } else if (data.type === "error") {
    detectorReady = false;
    inferenceBusy = false;
    const note = byId("video-note");
    note.hidden = false;
    note.textContent = "The detection model could not run in this browser.";
    renderNotice(`Detection stopped: ${data.message}`);
    setOverall("stopped", "Stopped");
  }
};

worker.onerror = () => {
  const note = byId("video-note");
  note.hidden = false;
  note.textContent = "This browser could not start the detection model.";
  setOverall("stopped", "Stopped");
};

worker.postMessage({
  type: "load",
  settings: {
    modelUrl: "./model/yolov8n.onnx",
    confidence: CONFIDENCE,
    overlap: OVERLAP,
    targetIds: [...TARGETS.keys()],
  },
});

function analyse() {
  const scale = INPUT_SIZE / Math.max(raw.width, raw.height);
  const width = raw.width * scale;
  const height = raw.height * scale;
  letterbox = {
    scale,
    dx: (INPUT_SIZE - width) / 2,
    dy: (INPUT_SIZE - height) / 2,
  };
  inputContext.fillStyle = "#808080";
  inputContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  inputContext.drawImage(raw, letterbox.dx, letterbox.dy, width, height);
  const pixels = inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  inferenceBusy = true;
  worker.postMessage(
    { type: "frame", generation, pixels: pixels.buffer },
    [pixels.buffer]
  );
}

function insideZone(item) {
  if (!zone) return true;
  const x = (item.x1 + item.x2) / 2 / view.width;
  const y = (item.y1 + item.y2) / 2 / view.height;
  return x >= zone.left && x <= zone.right && y >= zone.top && y <= zone.bottom;
}

function onResult(result) {
  // Cleared first: only one frame may be in the model at a time, even when
  // this result turns out to be out of date.
  inferenceBusy = false;
  if (result.generation !== generation) {
    pump();
    return;
  }

  const { scale, dx, dy } = letterbox;
  const found = result.boxes
    .map((box) => ({
      label: TARGETS.get(box.classId),
      score: box.score,
      x1: Math.max(0, (box.x1 - dx) / scale),
      y1: Math.max(0, (box.y1 - dy) / scale),
      x2: Math.min(view.width, (box.x2 - dx) / scale),
      y2: Math.min(view.height, (box.y2 - dy) / scale),
    }))
    .filter((item) => watching.has(item.label) && insideZone(item));

  detections = tracker.update(found);
  const breaches = monitor.update(detections.map((item) => item.label));
  renderView();

  for (const [label, frames] of breaches) {
    const matching = detections.filter((item) => item.label === label);
    const tracks = matching.map((item) => item.trackId).sort((a, b) => a - b);
    const peak = Math.max(...matching.map((item) => item.score));
    addAlert(
      "VISUAL",
      "HIGH",
      `Potential perimeter breach: '${label}' in frame for ${frames} ` +
        `consecutive frames (track IDs: ${tracks.join(", ")}; ` +
        `peak confidence: ${peak.toFixed(2)}).`
    );
    takeSnapshot(label);
  }

  const now = performance.now();
  if (lastResultAt) {
    const instant = 1000 / (now - lastResultAt);
    framesPerSecond = framesPerSecond
      ? framesPerSecond * 0.9 + instant * 0.1
      : instant;
  }
  lastResultAt = now;
  lastMilliseconds = result.milliseconds;
  renderDetectorTiles();
  pump();
}

// Keeps the picture moving. Browsers stop this while the tab is in the
// background, which is why it does not also drive the analysis.
function loop(now) {
  requestAnimationFrame(loop);
  if (activeSource?.capture(rawContext, now)) renderView();
}

// Analyses the next frame as soon as the model is free. It is driven by the
// model's own results, so a real camera keeps being watched in a background
// tab. The demonstration scene is not worth the battery, so it pauses there.
function pump() {
  if (!detectorReady || inferenceBusy || !activeSource) return;
  if (document.hidden && activeSource.simulated) return;
  if (activeSource.capture(rawContext, performance.now())) analyse();
  else setTimeout(pump, 50);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && activeSource?.simulated) {
    // The scene moved on while paused, so a streak would not be consecutive.
    resetTracking();
    pump();
  }
});

// ---------------------------------------------------------------------------
// Snapshots: kept in this page's memory only, and gone when it closes
// ---------------------------------------------------------------------------
function takeSnapshot(label) {
  view.toBlob(
    (blob) => {
      if (!blob) return;
      snapshots.unshift({
        url: URL.createObjectURL(blob),
        caption: `${label} ${new Date().toLocaleTimeString()}`,
      });
      for (const old of snapshots.splice(MAX_SNAPSHOTS)) {
        URL.revokeObjectURL(old.url);
      }
      renderSnapshots();
    },
    "image/jpeg",
    0.85
  );
}

function renderSnapshots() {
  const strip = byId("snapshots");
  strip.replaceChildren(
    ...snapshots.map((snapshot) => {
      const link = element("a");
      link.href = snapshot.url;
      link.target = "_blank";
      link.rel = "noopener";
      const image = element("img");
      image.src = snapshot.url;
      image.alt = "Breach snapshot";
      link.append(image, element("span", "", snapshot.caption));
      return link;
    })
  );
  strip.hidden = snapshots.length === 0;
}

// ---------------------------------------------------------------------------
// Perimeter zone: drag on the video to draw it
// ---------------------------------------------------------------------------
function pointerPosition(event) {
  const box = view.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
    y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)),
  };
}

function zoneBetween(a, b) {
  return {
    left: Math.min(a.x, b.x),
    top: Math.min(a.y, b.y),
    right: Math.max(a.x, b.x),
    bottom: Math.max(a.y, b.y),
  };
}

view.addEventListener("pointerdown", (event) => {
  draggingFrom = pointerPosition(event);
  view.setPointerCapture(event.pointerId);
});

view.addEventListener("pointermove", (event) => {
  if (draggingFrom) zone = zoneBetween(draggingFrom, pointerPosition(event));
});

view.addEventListener("pointerup", (event) => {
  if (!draggingFrom) return;
  const drawn = zoneBetween(draggingFrom, pointerPosition(event));
  draggingFrom = null;
  // A plain click, or a sliver too small to use, is not a zone.
  const usable = drawn.right - drawn.left > 0.03 && drawn.bottom - drawn.top > 0.03;
  setZone(usable ? drawn : null);
});

function setZone(next) {
  zone = next;
  byId("clear-zone").hidden = !zone;
  resetTracking();
}

byId("clear-zone").addEventListener("click", () => setZone(null));

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
function addAlert(source, severity, message) {
  counts[severity] += 1;
  byId(severity.toLowerCase()).textContent = counts[severity].toLocaleString();

  const row = element("li", "alert");
  row.dataset.severity = severity;
  row.dataset.source = source;
  const meta = element("div", "meta");
  meta.append(
    element("span", `badge ${severity}`, severity),
    element("span", "", source),
    element("span", "time", new Date().toLocaleTimeString())
  );
  const split = message.indexOf(": ");
  const title = split < 0 ? message : message.slice(0, split);
  row.append(meta, element("div", "title", title));
  if (split >= 0) {
    row.append(element("div", "detail", message.slice(split + 2)));
  }
  const feed = byId("feed");
  feed.prepend(row);
  while (feed.children.length > MAX_ROWS) feed.lastChild.remove();
  byId("feed-empty").hidden = true;
}

function setState(node, name, text) {
  const [mark, label, tone] = STATES[name];
  node.className = `${node.dataset.base ?? "state"} ${tone}`;
  node.replaceChildren(
    element("span", "mark", `${mark} `),
    document.createTextNode(text ?? label)
  );
}

function setOverall(name, text) {
  setState(byId("overall"), name, text);
}

function setValue(id, value, unit) {
  byId(id).replaceChildren(
    document.createTextNode(value),
    element("small", "", ` ${unit}`)
  );
}

function renderDetectorTiles() {
  if (!detectorReady) return;
  setValue("camera-rate", framesPerSecond ? Math.round(framesPerSecond) : 0, "FPS");
  const sub = byId("camera-sub");
  sub.replaceChildren();
  const state = element("span");
  state.dataset.base = "state";
  setState(state, "running");
  sub.append(state, ` · ${Math.round(lastMilliseconds)} ms per frame`);
  setOverall("running", "Live");
}

function renderNotice(problem) {
  const notice = byId("notice");
  if (problem) {
    notice.textContent = problem;
    notice.className = "notice problem";
    return;
  }
  notice.className = "notice";
  notice.textContent = activeSource?.simulated
    ? "Demonstration scene and simulated traffic: detection really runs in " +
      "this browser, but these alerts are not real events."
    : "Camera alerts are real, and your video stays on this device. " +
      "Network alerts come from simulated traffic and are not real events.";
}

byId("source").addEventListener("change", (event) => {
  chooseSource(event.target.value);
});

document.querySelectorAll(".controls input").forEach((box) => {
  box.addEventListener("change", () => {
    if (box.checked) watching.add(box.value);
    else watching.delete(box.value);
  });
});

document.querySelectorAll(".filters button").forEach((button, _, all) => {
  button.addEventListener("click", () => {
    byId("feed").dataset.filter = button.dataset.filter;
    all.forEach((other) => {
      other.setAttribute("aria-pressed", String(other === button));
    });
  });
});

// ---------------------------------------------------------------------------
// Simulated network telemetry, screened by the same robust z-score rules as
// the Python pipeline. The IsolationForest runs only in the Python version.
// ---------------------------------------------------------------------------
const uniform = (low, high) => low + Math.random() * (high - low);
const exponential = (mean) => -Math.log(1 - Math.random()) * mean;

function normal(mean, deviation) {
  const u = 1 - Math.random();
  const v = Math.random();
  return mean + deviation * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

class RobustBaseline {
  constructor(values, minimumScale = 0) {
    this.median = median(values);
    const spread = median(values.map((value) => Math.abs(value - this.median)));
    this.scale = Math.max(1.4826 * spread, minimumScale) || 1;
  }

  z(value) {
    return (value - this.median) / this.scale;
  }
}

class NetworkSimulator {
  constructor() {
    this.lineRate = 50;
    this.windowSeconds = 10;
    this.zLimit = 6;
    this.cooldownSeconds = 10;
    this.attackRate = 0.004;
    this.hosts = Array.from({ length: 40 }, (_, i) => `10.20.0.${i + 1}`);
    this.windows = new Map();
    this.lastAlert = new Map();
    this.clock = 0;
    this.queue = [];

    // Fit on benign history only, then discard it.
    const sizes = [];
    const durations = [];
    const volumes = [];
    for (let i = 0; i < 5000; i += 1) {
      const record = this.benign();
      sizes.push(record.size);
      durations.push(Math.log1p(record.duration));
      volumes.push(this.observe(record.ip, record.time));
    }
    this.sizeBaseline = new RobustBaseline(sizes);
    this.durationBaseline = new RobustBaseline(durations);
    const volumeMedian = median(volumes);
    this.volumeBaseline = new RobustBaseline(volumes, Math.sqrt(volumeMedian));
    this.windows.clear();
  }

  benign() {
    this.clock += exponential(1 / this.lineRate);
    const control = Math.random() < 0.35;
    return {
      time: this.clock,
      ip: this.hosts[Math.floor(Math.random() * this.hosts.length)],
      size: control
        ? uniform(40, 120)
        : Math.min(1500, Math.max(200, normal(1100, 280))),
      duration: Math.max(1, Math.exp(normal(4.5, 0.7))),
      flag: Math.random() < 0.002,
      mapped: true,
    };
  }

  attacker() {
    return `203.0.113.${1 + Math.floor(Math.random() * 254)}`;
  }

  injectAttack() {
    const kind = Math.floor(Math.random() * 3);
    if (kind === 0) {
      // A massive transfer from an unknown address.
      this.queue.push({
        time: this.clock,
        ip: this.attacker(),
        size: uniform(20000, 120000),
        duration: uniform(200, 3000),
        flag: Math.random() < 0.3,
        mapped: false,
      });
    } else if (kind === 1) {
      // A rapid run of tiny connections from one unknown address.
      const ip = this.attacker();
      const lines = 80 + Math.floor(Math.random() * 121);
      for (let i = 0; i < lines; i += 1) {
        this.clock += uniform(0.005, 0.02);
        this.queue.push({
          time: this.clock,
          ip,
          size: uniform(40, 120),
          duration: uniform(1, 10),
          flag: Math.random() < 0.3,
          mapped: false,
        });
      }
    } else {
      // A known host holding an unusually long session.
      this.queue.push({
        time: this.clock,
        ip: this.hosts[Math.floor(Math.random() * this.hosts.length)],
        size: uniform(600, 1400),
        duration: uniform(60000, 600000),
        flag: true,
        mapped: true,
      });
    }
  }

  observe(ip, time) {
    const events = this.windows.get(ip) ?? [];
    this.windows.set(ip, events);
    const horizon = time - this.windowSeconds;
    while (events.length && events[0] <= horizon) events.shift();
    events.push(time);
    return events.length;
  }

  screen(record) {
    const connections = this.observe(record.ip, record.time);
    const packetZ = this.sizeBaseline.z(record.size);
    const durationZ = this.durationBaseline.z(Math.log1p(record.duration));
    const volumeZ = this.volumeBaseline.z(connections);
    let category = null;
    if (packetZ >= this.zLimit) category = "packet burst";
    else if (volumeZ >= this.zLimit) category = "connection spike";
    else if (durationZ >= this.zLimit) category = "prolonged session";
    if (!category) return;

    const severity = record.mapped ? "MEDIUM" : "HIGH";
    const key = `${record.ip}|${severity}|${category}`;
    const previous = this.lastAlert.get(key);
    if (previous !== undefined && record.time - previous < this.cooldownSeconds) {
      return;
    }
    this.lastAlert.set(key, record.time);
    const origin = record.mapped ? "mapped" : "unmapped";
    addAlert(
      "NETWORK",
      severity,
      `Network anomaly (${category}) from ${origin} IP ${record.ip}: ` +
        `packet_size_bytes=${record.size.toFixed(0)}, ` +
        `connection_duration_ms=${record.duration.toFixed(1)}, ` +
        `unrecognized_port_flag=${record.flag ? 1 : 0}, ` +
        `connections_in_window=${connections}, ` +
        `packet_z=${packetZ.toFixed(1)}, duration_z=${durationZ.toFixed(1)}, ` +
        `volume_z=${volumeZ.toFixed(1)}`
    );
  }

  step(lines) {
    for (let i = 0; i < lines; i += 1) {
      if (!this.queue.length) {
        if (Math.random() < this.attackRate) this.injectAttack();
        else this.queue.push(this.benign());
      }
      this.screen(this.queue.shift());
    }
    // Forget quiet addresses, so memory stays bounded over a long session.
    if (this.windows.size > 4096) this.windows.clear();
    if (this.lastAlert.size > 4096) this.lastAlert.clear();
  }
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
const demoScene = await new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(new DemoScene(image));
  image.onerror = () => reject(new Error("The sample scene could not load."));
  image.src = "assets/scene.jpg";
});

await useSource(demoScene);
requestAnimationFrame(loop);

const network = new NetworkSimulator();
setValue("network-rate", network.lineRate, "lines/s");
const networkState = element("span");
networkState.dataset.base = "state";
setState(networkState, "simulated", "Simulated traffic");
byId("network-sub").replaceChildren(networkState);
setInterval(() => network.step(5), 100);
