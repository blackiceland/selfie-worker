import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional, Dict
import json
import time
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
IP_ADAPTER_LORA_WEIGHT = os.getenv("IP_ADAPTER_LORA_WEIGHT")
LCM_LORA_ID = os.getenv("LCM_LORA_ID", "latent-consistency/lcm-lora-sdxl")
DEFAULT_LCM_STEPS = 4
DEFAULT_LCM_GUIDANCE = 2.0


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


def _setup_logging() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _log(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    import logging
    payload: Dict[str, Any] = {"event": event}
    if data is not None:
        payload.update(data)
    logging.info(json.dumps(payload, ensure_ascii=False))

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
    model_name = os.getenv("ARCFACE_MODEL", "buffalo_l")
    model = insightface.app.FaceAnalysis(name=model_name)
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
        if IP_ADAPTER_LORA_WEIGHT is not None and len(IP_ADAPTER_LORA_WEIGHT) > 0:
            try:
                _state.holder.pipeline.load_lora_weights(
                    IP_ADAPTER_REPO,
                    weight_name=IP_ADAPTER_LORA_WEIGHT,
                )
                _log("ip_adapter_lora_loaded", {"weight": IP_ADAPTER_LORA_WEIGHT})
            except Exception as ex:
                _log("ip_adapter_lora_load_failed", {"error": str(ex)})
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
        _state.holder.pipeline.vae = _state.holder.pipeline.vae.to(dtype=torch.float32)
        _state.holder.pipeline.enable_vae_slicing()
        _state.holder.pipeline.enable_vae_tiling()
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
    _setup_logging()
    started_at = time.time()
    _log("request_received", {
        "use_lcm": use_lcm,
        "steps": steps,
        "guidance": guidance_scale,
        "id_scale": id_scale,
        "wh": [width, height],
        "batch": batch,
    })
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
            _log("id_ref_processed", {"embedding": True})
            try:
                import cv2
                from insightface.utils import face_align
                array = np.frombuffer(id_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
                faces = _state.arcface_model.get(img_bgr) if _state.arcface_model is not None else []
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    crop_bgr = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=224)
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    face_pil = Image.fromarray(crop_rgb)
                    if _state.holder is not None:
                        embeds = _state.holder.pipeline.prepare_ip_adapter_image_embeds(
                            ip_adapter_image=[face_pil],
                            device=_state.holder.device,
                            do_classifier_free_guidance=True,
                            num_images_per_prompt=int(batch),
                        )
                        clip_embeds = embeds[0] if isinstance(embeds, (list, tuple)) else embeds
                        proj = _state.holder.pipeline.unet.encoder_hid_proj.image_projection_layers[0]
                        proj.clip_embeds = clip_embeds.to(dtype=torch.float16 if _state.holder.device == "cuda" else torch.float32)
                        if hasattr(proj, "shortcut"):
                            setattr(proj, "shortcut", True)
                        _log("face_clip_embeds_set", {"shape": list(proj.clip_embeds.shape) if hasattr(proj.clip_embeds, 'shape') else []})
                else:
                    _log("face_align_skipped", {"reason": "no_face"})
            except Exception as ex:
                _log("face_clip_embed_error", {"error": str(ex)})
        else:
            _log("id_ref_processed", {"embedding": False})
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=_state.holder.device).manual_seed(int(seed))
    with _state.holder.lock, torch.inference_mode():
        if use_lcm:
            _ensure_lcm_loaded()
            _log("lcm_enabled", {"scheduler": True})
        ip_kwargs: Dict[str, Any] = {}
        if id_embedding is not None and _state.ip_adapter_loaded:
            emb_t = torch.tensor(id_embedding, dtype=torch.float32, device=_state.holder.device).unsqueeze(0).unsqueeze(0)
            neg = torch.zeros_like(emb_t)
            faceid_embeds = torch.cat([neg, emb_t], dim=0)
            if int(batch) > 1:
                faceid_embeds = faceid_embeds.repeat(1, int(batch), 1)
            ip_kwargs = {
                "ip_adapter_image": None,
                "ip_adapter_image_embeds": [faceid_embeds],
                "ip_adapter_scale": [float(id_scale)],
            }
            _log("ip_adapter_applied", {"scale": id_scale, "emb_shape": list(faceid_embeds.shape)})
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
    _log("request_completed", {"elapsed_ms": int((time.time() - started_at) * 1000)})
    return Response(content=buffer.getvalue(), media_type="image/png")

