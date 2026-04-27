"""
backend/main.py  —  FastAPI Chest X-Ray Pneumonia Detection API
Phase 3: Real ONNXRuntime inference (no Triton yet — that comes in Phase 5)
Phase 4: LLM report via Anthropic Claude
"""
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tritonclient.http as httpclient
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = str(BASE_DIR / 'triton/model_repository/pneumonia_model/1/model.onnx')
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")  # Note: Triton HTTP port is usually 8000, but in docker-compose we map 8001 to 8000 locally. We will check 8001.
if TRITON_URL == "triton:8000": # if running locally outside docker, default to localhost:8001
    TRITON_URL = "localhost:8001"

# ── Globals (populated on startup) ───────────────────────────────────────────
ort_session: ort.InferenceSession = None
triton_client: httpclient.InferenceServerClient = None
use_triton = False


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session, triton_client, use_triton

    # 1. Try connecting to Triton Server first
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        if triton_client.is_server_live():
            print(f"[Backend] Connected to NVIDIA Triton Server at {TRITON_URL}")
            use_triton = True
    except Exception as e:
        print(f"[Backend] Triton Server not found at {TRITON_URL}: {e}")

    # 2. Fallback to Local ONNXRuntime
    if not use_triton:
        print(f"[Backend] Falling back to local ONNX model: {MODEL_PATH}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
        active = ort_session.get_providers()[0]
        print(f"[Backend] ONNXRuntime ready  |  Provider: {active}")

    yield
    print("[Backend] Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PneumoScan AI — Backend API",
    description="Chest X-Ray Pneumonia Detection powered by ResNet-50 ONNX + Claude AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173",
                   "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label:             str
    confidence:        float
    inference_time_ms: float
    report:            str | None = None


# ── Preprocessing ─────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Upload a valid JPEG or PNG.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)          # HWC -> CHW
    return np.expand_dims(img, 0)         # (1, 3, 224, 224)


# ── LLM report (Phase 4) ─────────────────────────────────────────────────────
async def generate_report(label: str, confidence: float) -> str | None:
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None   # API key not configured yet
    try:
        from openai import AsyncOpenAI
        # Defaulting to Groq base URL and their fast llama3 model
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        conf_pct = confidence * 100
        prompt = (
            f"A chest X-ray AI model classified this image as {label} "
            f"with {conf_pct:.1f}% confidence.\n\n"
            "Write a concise (3-4 sentences) structured diagnostic note "
            "as if you were a radiologist assistant. Include: "
            "(1) what the finding suggests, "
            "(2) recommended next steps, "
            "(3) a brief disclaimer that this is AI-assisted and not a final diagnosis. "
            "Use plain clinical English. Do not use markdown."
        )
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Monitoring"])
async def health():
    if use_triton:
        return {"status": "ok", "model": "ResNet50-Triton", "provider": "NVIDIA Triton Server"}
    else:
        provider = ort_session.get_providers()[0] if ort_session else "not loaded"
        return {"status": "ok", "model": "ResNet50-ONNX", "provider": provider}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Upload a chest X-ray (JPEG/PNG) → label + confidence + AI report."""
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 15 MB)")

    try:
        img = preprocess(file_bytes)
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Inference
    t0 = time.perf_counter()
    if use_triton:
        inputs = [httpclient.InferInput("input", img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img)
        outputs = [httpclient.InferRequestedOutput("output")]
        
        response = triton_client.infer("pneumonia_model", inputs, outputs=outputs)
        out_array = response.as_numpy("output")
        confidence = float(out_array[0][0])
    else:
        # Local ONNX inference
        outputs = ort_session.run(None, {"input": img})
        confidence = float(outputs[0][0][0])
        
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label = "PNEUMONIA" if confidence >= 0.5 else "NORMAL"

    # LLM report (Phase 4 — returns None if key not set)
    report = await generate_report(label, confidence)

    return PredictionResponse(
        label=label,
        confidence=round(confidence, 4),
        inference_time_ms=round(elapsed_ms, 2),
        report=report,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
