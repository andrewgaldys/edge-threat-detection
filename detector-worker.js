// Runs the YOLO model in a background thread, so the page stays smooth while
// each frame is analysed. Pixels arrive from app.js and only boxes go back:
// this file never sends anything over the network.
import * as ort from "./vendor/ort.wasm.bundle.min.mjs";

const SIZE = 320;
const CLASS_COUNT = 80;

let session = null;
let confidenceLimit = 0.45;
let overlapLimit = 0.5;
let targetIds = new Set();

// GitHub Pages cannot send the headers that multi-threaded WebAssembly needs,
// so the model runs on a single thread here.
ort.env.wasm.numThreads = 1;
// Naming only the binary keeps the JavaScript glue that ships in this bundle.
ort.env.wasm.wasmPaths = {
  wasm: new URL("./vendor/ort-wasm-simd-threaded.wasm", import.meta.url).href,
};

// Downloads the model piece by piece, so the page can show progress.
async function download(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Model download failed (HTTP ${response.status})`);
  }
  const reader = response.body.getReader();
  const pieces = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    pieces.push(value);
    received += value.length;
    postMessage({ type: "progress", received });
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const piece of pieces) {
    bytes.set(piece, offset);
    offset += piece.length;
  }
  return bytes;
}

function overlap(a, b) {
  const width = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const height = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (width <= 0 || height <= 0) return 0;
  const shared = width * height;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return shared / (areaA + areaB - shared);
}

// Non-maximum suppression: keep the best box for each object, per class.
function suppress(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of boxes) {
    const duplicate = kept.some(
      (other) =>
        other.classId === box.classId && overlap(other, box) > overlapLimit
    );
    if (!duplicate) kept.push(box);
  }
  return kept;
}

async function analyse(pixels) {
  // RGBA pixels to the normalised red, green, blue planes the model expects.
  const area = SIZE * SIZE;
  const input = new Float32Array(3 * area);
  for (let pixel = 0, source = 0; pixel < area; pixel += 1, source += 4) {
    input[pixel] = pixels[source] / 255;
    input[pixel + area] = pixels[source + 1] / 255;
    input[pixel + 2 * area] = pixels[source + 2] / 255;
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]];
  const data = output.data;
  const count = output.dims[2];

  // Each column is one candidate: centre x, centre y, width, height, then a
  // score for every class. The best class decides what the object is, the
  // same way the Python pipeline's non-maximum suppression does.
  const boxes = [];
  for (let index = 0; index < count; index += 1) {
    let classId = -1;
    let score = 0;
    for (let cls = 0; cls < CLASS_COUNT; cls += 1) {
      const value = data[(4 + cls) * count + index];
      if (value > score) {
        score = value;
        classId = cls;
      }
    }
    if (score > confidenceLimit && targetIds.has(classId)) {
      const x = data[index];
      const y = data[count + index];
      const width = data[2 * count + index];
      const height = data[3 * count + index];
      boxes.push({
        classId,
        score,
        x1: x - width / 2,
        y1: y - height / 2,
        x2: x + width / 2,
        y2: y + height / 2,
      });
    }
  }
  return suppress(boxes);
}

async function load(settings) {
  confidenceLimit = settings.confidence;
  overlapLimit = settings.overlap;
  targetIds = new Set(settings.targetIds);
  const bytes = await download(new URL(settings.modelUrl, import.meta.url));
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  // One blank pass, so that the first real frame is not unusually slow.
  await analyse(new Uint8ClampedArray(SIZE * SIZE * 4));
  postMessage({ type: "ready" });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await load(data.settings);
    } else if (data.type === "frame") {
      const started = performance.now();
      const boxes = await analyse(new Uint8ClampedArray(data.pixels));
      postMessage({
        type: "result",
        generation: data.generation,
        boxes,
        milliseconds: performance.now() - started,
      });
    }
  } catch (error) {
    postMessage({ type: "error", message: String(error.message || error) });
  }
};
