import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional, Dict
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from diffusers.loaders import AttnProcsLayers
import numpy as np

DEFAULT_BASE_MODEL = "SG161222/RealVisXL_V3.0_Turbo"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0
DEFAULT_PROMPT = "highly realistic studio headshot, cinematic, sharp focus"
DEFAULT_NEGATIVE = "plastic skin, deformed hands, extra fingers, blurry, watermark, text"
IP_ADAPTER_REPO = "h94/IP-Adapter-FaceID-Plus"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = os.getenv("IP_ADAPTER_WEIGHT", "ip-adapter-faceid-plus_sdxl.bin")
LCM_LORA_ID = os.getenv("LCM_LORA_ID", "latent-consistency/lcm-lora-sdxl")
DEFAULT_LCM_STEPS = 4
DEFAULT_LCM_GUIDANCE = 6.5


class DeviceResolver:
    @staticmethod
    def resolve() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

class ModelHolder:
    def __init__(self, device: str, pipeline: StableDiffusionXLPipeline) -> None:
        self.device: str = device
        self.pipeline: StableDiffusionXLPipeline = pipeline
        self.lock: threading.Lock = threading.Lock()

    @staticmethod
    def create(model_id: str, device: str) -> "ModelHolder":
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True).to(device)
        return ModelHolder(device=device, pipeline=pipeline)


def get_device() -> str:
    return DeviceResolver.resolve()

def get_model_id() -> str:
    value: Optional[str] = os.getenv("BASE_MODEL")
    return value if value is not None and len(value) > 0 else DEFAULT_BASE_MODEL


@dataclass
class AppState:
    holder: Optional[ModelHolder]
    loading: bool
    load_error: Optional[str]
    arcface_model: Optional[Any]
    ip_adapter_loaded: bool
    lcm_loaded: bool

app: FastAPI = FastAPI()
_state = AppState(holder=None, loading=False, load_error=None, arcface_model=None, ip_adapter_loaded=False, lcm_loaded=False)

def _load_model_async() -> None:
    try:
        model_id: str = get_model_id()
        device: str = get_device()
        _state.holder = ModelHolder.create(model_id=model_id, device=device)
    except Exception as ex:
        _state.load_error = str(ex)
    finally:
        _state.loading = False

def _ensure_loader_started() -> None:
    if _state.holder is None and not _state.loading:
        _state.loading = True
        threading.Thread(target=_load_model_async, daemon=True).start()

def _lazy_load_arcface() -> None:
    if _state.arcface_model is not None:
        return
    import insightface
    model = insightface.app.FaceAnalysis(name="buffalo_m")
    model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    _state.arcface_model = model

def _compute_arcface_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        import cv2
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = _state.arcface_model.get(img) if _state.arcface_model is not None else []
        if not faces:
            return None
        return faces[0].normed_embedding
    except Exception:
        return None

def _ensure_ip_adapter_loaded() -> None:
    if _state.holder is None or _state.ip_adapter_loaded:
        return
    try:
        _state.holder.pipeline.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT,
        )
        _state.ip_adapter_loaded = True
    except Exception:
        _state.ip_adapter_loaded = False


def _ensure_lcm_loaded() -> None:
    if _state.holder is None or _state.lcm_loaded:
        return
    try:
        layers = AttnProcsLayers.from_pretrained(LCM_LORA_ID)
        _state.holder.pipeline.load_lora_weights(layers)
        _state.holder.pipeline.scheduler = LCMScheduler.from_config(_state.holder.pipeline.scheduler.config)
        _state.lcm_loaded = True
    except Exception:
        _state.lcm_loaded = False

@app.get("/health")
def health() -> JSONResponse:
    _ensure_loader_started()
    if _state.load_error is not None:
        return JSONResponse({"status": "error", "message": _state.load_error}, status_code=500)
    if _state.holder is None:
        return JSONResponse({"status": "loading"})
    return JSONResponse({"status": "ready", "device": _state.holder.device})

@app.post("/generate")
def generate(
    image: Optional[UploadFile] = File(default=None),
    id_ref: Optional[UploadFile] = File(default=None),
    id_scale: float = Form(default=1.0),
    prompt: str = Form(default=DEFAULT_PROMPT),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE),
    width: int = Form(default=DEFAULT_WIDTH),
    height: int = Form(default=DEFAULT_HEIGHT),
    steps: int = Form(default=DEFAULT_STEPS),
    guidance_scale: float = Form(default=DEFAULT_GUIDANCE),
    use_lcm: bool = Form(default=False),
    batch: int = Form(default=1),
    seed: Optional[int] = Form(default=None),
) -> Response:
    _ensure_loader_started()
    if _state.holder is None:
        return JSONResponse({"error": "model_not_ready"}, status_code=503)
    id_embedding: Optional[np.ndarray] = None
    if id_ref is not None:
        _lazy_load_arcface()
        id_bytes = id_ref.file.read()
        id_embedding = _compute_arcface_embedding(id_bytes)
        if id_embedding is not None:
            _ensure_ip_adapter_loaded()
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=_state.holder.device).manual_seed(int(seed))
    with _state.holder.lock, torch.inference_mode():
        if use_lcm:
            _ensure_lcm_loaded()
        ip_kwargs: Dict[str, Any] = {}
        if id_embedding is not None and _state.ip_adapter_loaded:
            emb_t = torch.tensor(id_embedding, dtype=torch.float32, device=_state.holder.device)
            ip_kwargs = {
                "ip_adapter_image": None,
                "ip_adapter_image_embeds": [emb_t, emb_t],
                "ip_adapter_scale": [float(id_scale), float(id_scale)],
            }
        effective_steps = int(steps if not use_lcm else DEFAULT_LCM_STEPS)
        effective_guidance = float(guidance_scale if not use_lcm else DEFAULT_LCM_GUIDANCE)
        result_images = _state.holder.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=effective_steps,
            guidance_scale=effective_guidance,
            height=int(height),
            width=int(width),
            generator=generator,
            num_images_per_prompt=int(batch),
            **ip_kwargs,
        ).images
    buffer: io.BytesIO = io.BytesIO()
    result_images[0].save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")

