"use strict";

const ORT_WASM_PATHS = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const VISION_LIB = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MODEL_URL = "models/mask_detector.onnx";

const INPUT_NAME = "input_1";
const OUTPUT_NAME = "dense_1";
const SIZE = 224;

let maskSession = null;
let detector = null;
let streaming = false;
let rafId = null;
let stream = null;

const chooseBox = document.getElementById("choose");
const webcamPanel = document.getElementById("webcamPanel");
const imagePanel = document.getElementById("imagePanel");
const video = document.getElementById("video");
const camCanvas = document.getElementById("camCanvas");
const camCtx = camCanvas.getContext("2d");
const camStatus = document.getElementById("camStatus");
const imgCanvas = document.getElementById("imgCanvas");
const imgCtx = imgCanvas.getContext("2d");
const imgStatus = document.getElementById("imgStatus");
const imgStage = document.getElementById("imgStage");
const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const notice = document.getElementById("notice");

function notify(msg) {
  notice.textContent = msg;
  notice.style.display = msg ? "block" : "none";
}

function showPanel(name) {
  chooseBox.style.display = "none";
  webcamPanel.style.display = name === "webcam" ? "block" : "none";
  imagePanel.style.display = name === "image" ? "block" : "none";
}

function backToMenu() {
  stopWebcam();
  chooseBox.style.display = "flex";
  webcamPanel.style.display = "none";
  imagePanel.style.display = "none";
}

/* ------------------------- model loading ------------------------- */

async function initModels() {
  notify("Loading AI models…");
  try {
    if (!maskSession) {
      ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
      ort.env.wasm.numThreads = 1;
      maskSession = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }
    if (!detector) {
      const vision = await import(VISION_LIB + "vision_bundle.mjs");
      const fileset = await vision.FilesetResolver.forVisionTasks(VISION_LIB + "wasm");
      let made = false;
      try {
        detector = await vision.FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.5,
        });
        made = true;
      } catch (e) {
        /* GPU not available, fall back to CPU */
      }
      if (!made) {
        detector = await vision.FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.5,
        });
      }
    }
    notify("");
    return true;
  } catch (err) {
    console.error(err);
    notify("Failed to load models: " + err.message + " — check your internet connection.");
    return false;
  }
}

/* ------------------------- preprocessing ------------------------- */

function cropFace(srcCanvas, box) {
  const det = new Float32Array(SIZE * SIZE * 3);
  const tmp = document.createElement("canvas");
  tmp.width = SIZE;
  tmp.height = SIZE;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });
  const pad = 0;
  const x = Math.max(0, box.originX - pad);
  const y = Math.max(0, box.originY - pad);
  const w = Math.min(box.width + pad * 2, srcCanvas.width - x);
  const h = Math.min(box.height + pad * 2, srcCanvas.height - y);
  tctx.drawImage(srcCanvas, x, y, w, h, 0, 0, SIZE, SIZE);
  const d = tctx.getImageData(0, 0, SIZE, SIZE).data;
  let j = 0;
  for (let i = 0; i < d.length; i += 4) {
    det[j++] = d[i] / 127.5 - 1.0;
    det[j++] = d[i + 1] / 127.5 - 1.0;
    det[j++] = d[i + 2] / 127.5 - 1.0;
  }
  return det;
}

async function classifyMask(srcCanvas, box) {
  const data = cropFace(srcCanvas, box);
  const feeds = {};
  feeds[INPUT_NAME] = new ort.Tensor("float32", data, [1, SIZE, SIZE, 3]);
  const out = await maskSession.run(feeds);
  const r = out[OUTPUT_NAME].data;
  const mask = r[0];
  const noMask = r[1];
  return {
    label: mask > noMask ? "MASK" : "NO MASK",
    conf: Math.max(mask, noMask),
  };
}

/* ------------------------- face detection ------------------------- */

function getFaceBoxes(src) {
  const ts = performance.now();
  const res = detector.detectForVideo(src, ts);
  return res.detections || [];
}

/* ------------------------- webcam ------------------------- */

async function startWebcam() {
  if (!(await initModels())) return;
  showPanel("webcam");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    notify("Camera blocked or unavailable. Enable camera access and try again.");
    backToMenu();
    return;
  }
  video.srcObject = stream;
  await video.play();
  camCanvas.width = 480;
  camCanvas.height = Math.round((480 * video.videoHeight) / video.videoWidth);
  camStatus.textContent = "Detecting…";
  streaming = true;
  tick();
}

function drawBox(ctx, box, text) {
  const color = text.startsWith("MASK") ? "#22c55e" : "#ef4444";
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(box.originX, box.originY, box.width, box.height);
  ctx.font = "bold 14px Inter, sans-serif";
  const tw = ctx.measureText(text).width + 12;
  const ty = Math.max(0, box.originY - 22);
  ctx.fillStyle = color;
  ctx.fillRect(box.originX, ty, tw, 22);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, box.originX + 6, ty + 16);
}

async function tick() {
  if (!streaming) return;
  camCtx.drawImage(video, 0, 0, camCanvas.width, camCanvas.height);

  let boxes = [];
  try {
    boxes = getFaceBoxes(camCanvas);
  } catch (e) {
    /* ignore occasional detection errors */
  }

  const results = [];
  for (const det of boxes) {
    const box = det.boundingBox;
    let cls = { label: "…", conf: 0 };
    try {
      cls = await classifyMask(camCanvas, box);
    } catch (e) {
      console.error(e);
    }
    results.push({ box, cls });
  }

  camCtx.clearRect(0, 0, camCanvas.width, camCanvas.height);
  camCtx.drawImage(video, 0, 0, camCanvas.width, camCanvas.height);
  for (const r of results) {
    drawBox(camCtx, r.box, r.cls.label + " " + (r.cls.conf * 100).toFixed(1) + "%");
  }
  camStatus.textContent =
    results.length > 0
      ? results.length + " face(s) found"
      : "No face detected — show your face to the camera";
  if (streaming) setTimeout(tick, 40);
}

function stopWebcam() {
  streaming = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (video.srcObject) video.srcObject = null;
}

/* ------------------------- image ------------------------- */

async function handleFile(file) {
  if (!file) return;
  if (!(await initModels())) return;
  const img = new Image();
  img.onload = async () => {
    const maxW = 800;
    const scale = img.width > maxW ? maxW / img.width : 1;
    imgCanvas.width = Math.round(img.width * scale);
    imgCanvas.height = Math.round(img.height * scale);
    imgCtx.drawImage(img, 0, 0, imgCanvas.width, imgCanvas.height);

    let boxes = [];
    try {
      boxes = getFaceBoxes(imgCanvas);
    } catch (e) {
      console.error(e);
    }

    if (boxes.length === 0) {
      imgStatus.textContent = "No face detected in this image";
      return;
    }
    let foundMask = 0;
    let foundNone = 0;
    for (const det of boxes) {
      const box = det.boundingBox;
      let cls = { label: "…", conf: 0 };
      try {
        cls = await classifyMask(imgCanvas, box);
      } catch (e) {
        console.error(e);
      }
      if (cls.label === "MASK") foundMask++;
      else foundNone++;
      const color = cls.label === "MASK" ? "#22c55e" : "#ef4444";
      imgCtx.strokeStyle = color;
      imgCtx.lineWidth = 3;
      imgCtx.strokeRect(box.originX, box.originY, box.width, box.height);
      const text = cls.label + " " + (cls.conf * 100).toFixed(1) + "%";
      imgCtx.font = "bold 14px Inter, sans-serif";
      const tw = imgCtx.measureText(text).width + 12;
      const ty = Math.max(0, box.originY - 22);
      imgCtx.fillStyle = color;
      imgCtx.fillRect(box.originX, ty, tw, 22);
      imgCtx.fillStyle = "#fff";
      imgCtx.fillText(text, box.originX + 6, ty + 16);
    }
    imgStatus.textContent =
      "Result: " + foundMask + " with mask, " + foundNone + " without mask";
    imgStage.classList.remove("hidden");
  };
  img.onerror = () => notify("Could not read that image file.");
  img.src = URL.createObjectURL(file);
}

/* ------------------------- wiring ------------------------- */

document.getElementById("optWebcam").addEventListener("click", startWebcam);
document.getElementById("optImage").addEventListener("click", () => {
  showPanel("image");
});
document.getElementById("stopCam").addEventListener("click", () => {
  stopWebcam();
  backToMenu();
});
document.getElementById("backWebcam").addEventListener("click", backToMenu);
document.getElementById("backImage").addEventListener("click", backToMenu);

document.getElementById("fileLabel").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  uploadBox.style.display = "none";
  imgStage.classList.remove("hidden");
  imgStatus.textContent = "Analyzing…";
  handleFile(fileInput.files[0]);
});