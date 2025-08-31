import io
import os
import threading
from typing import Optional
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from diffusers import StableDiffusionXLPipeline
import numpy as np
from PIL import Image

_arcface_model = None
_ip_adapter_loaded = False


class ModelHolder:
    def __init__(self, model_id: str, device: str) -> None:
        self.device: str = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipeline: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        ).to(self.device)
        self.lock: threading.Lock = threading.Lock()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_id() -> str:
    env_value: Optional[str] = os.getenv("BASE_MODEL")
    if env_value is not None and len(env_value) > 0:
        return env_value
    return "SG161222/RealVisXL_V3.0_Turbo"


app: FastAPI = FastAPI()
_holder: Optional[ModelHolder] = None
_loading: bool = False
_load_error: Optional[str] = None


def _load_model_async() -> None:
    global _holder, _loading, _load_error
    try:
        model_id: str = get_model_id()
        device: str = get_device()
        _holder = ModelHolder(model_id=model_id, device=device)
    except Exception as ex:
        _load_error = str(ex)
    finally:
        _loading = False

def _ensure_loader_started() -> None:
    global _loading
    if _holder is None and not _loading:
        _loading = True
        thread = threading.Thread(target=_load_model_async, daemon=True)
        thread.start()


def _lazy_load_arcface() -> None:
    global _arcface_model
    if _arcface_model is not None:
        return
    import insightface
    _arcface_model = insightface.app.FaceAnalysis(name="buffalo_m")
    _arcface_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)


def _compute_arcface_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        import cv2
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = _arcface_model.get(img)
        if not faces:
            return None
        return faces[0].normed_embedding
    except Exception:
        return None


@app.get("/health")
def health() -> JSONResponse:
    _ensure_loader_started()
    if _load_error is not None:
        return JSONResponse({"status": "error", "message": _load_error}, status_code=500)
    if _holder is None:
        return JSONResponse({"status": "loading"})
    return JSONResponse({"status": "ready", "device": _holder.device})


@app.post("/generate")
def generate(
    image: Optional[UploadFile] = File(default=None),
    id_ref: Optional[UploadFile] = File(default=None),
    id_scale: float = Form(default=1.0),
    prompt: str = Form(default="highly realistic studio headshot, cinematic, sharp focus"),
    negative_prompt: str = Form(default="plastic skin, deformed hands, extra fingers, blurry, watermark, text"),
    width: int = Form(default=1024),
    height: int = Form(default=1024),
    steps: int = Form(default=30),
    guidance_scale: float = Form(default=7.0),
    seed: Optional[int] = Form(default=None),
) -> Response:
    _ensure_loader_started()
    if _holder is None:
        return JSONResponse({"error": "model_not_ready"}, status_code=503)
    id_embedding: Optional[np.ndarray] = None
    if id_ref is not None:
        _lazy_load_arcface()
        id_bytes = id_ref.file.read()
        id_embedding = _compute_arcface_embedding(id_bytes)
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=_holder.device).manual_seed(int(seed))
    with _holder.lock, torch.inference_mode():
        result_images = _holder.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            height=int(height),
            width=int(width),
            generator=generator,
        ).images
    buffer: io.BytesIO = io.BytesIO()
    result_images[0].save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")

