"""Local inpainting backend for Chameleon (SDXL Inpainting via diffusers).

Runs natively on macOS so it can use the Apple GPU (MPS); Docker on macOS has no GPU access.
It speaks the same API as ImageEditor, so the Gateway switches to it by setting
  IMAGE_EDITOR_BASE_URL=http://host.docker.internal:8006
in Gateway/.env.

Unlike GPT Image models, this is true inpainting: only the masked region is regenerated, and the
original pixels outside the mask are pasted back so the guide image's background is kept exactly.

Mask convention (same as DALL-E 2 / MaskGenerator): fully transparent pixels = region to regenerate.
"""
import base64
import io
import os
import threading
import time

import numpy as np
import torch
from diffusers import AutoencoderKL, AutoPipelineForInpainting
from fastapi import FastAPI, UploadFile
from PIL import Image, ImageFilter

import resources

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _local_or_hub(repo: str) -> str:
    # prefer weights fetched by download_models.sh; otherwise let diffusers download from the Hub
    local = os.path.join(MODELS_DIR, repo)
    return local if os.path.isdir(local) else repo


MODEL_ID = os.getenv("INPAINT_MODEL", _local_or_hub("diffusers/stable-diffusion-xl-1.0-inpainting-0.1"))
# SDXL's VAE overflows in fp16 (black/NaN images on MPS); this is a numerically fixed drop-in replacement
VAE_ID = os.getenv("INPAINT_VAE", _local_or_hub("madebyollin/sdxl-vae-fp16-fix"))
WORK_SIZE = int(os.getenv("INPAINT_WORK_SIZE", 1024))  # SDXL's native resolution
STEPS = int(os.getenv("INPAINT_STEPS", 30))
GUIDANCE = float(os.getenv("INPAINT_GUIDANCE", 8.0))
STRENGTH = float(os.getenv("INPAINT_STRENGTH", 0.99))  # < 1.0 recommended for this model
NEGATIVE_PROMPT = os.getenv("INPAINT_NEGATIVE_PROMPT",
                            "cartoon, painting, illustration, drawing, deformed, disfigured, blurry, low quality")
# feather the paste-back edge so the mask boundary doesn't show as a seam
FEATHER = int(os.getenv("INPAINT_FEATHER", 4))

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.float16

app = FastAPI()
_pipe = None
_lock = threading.Lock()  # one generation at a time; the GPU is the bottleneck anyway


def get_pipe():
    global _pipe
    if _pipe is None:
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
        _pipe = AutoPipelineForInpainting.from_pretrained(MODEL_ID, vae=vae, torch_dtype=dtype, variant="fp16")
        _pipe.to(device)
        _pipe.enable_attention_slicing()
    return _pipe


@app.on_event("startup")
def load_model():
    t = time.time()
    get_pipe()
    print(f"loaded {MODEL_ID} on {device} in {time.time() - t:.0f}s")
    resources.start_sampler(device)


def new_usage() -> dict:
    # returned with every response; the Gateway writes it to the dataset's cost ledger
    return {"backend": "sdxl", "model": os.path.basename(MODEL_ID.rstrip("/")), "device": device,
            "settings": f"{WORK_SIZE}px, {STEPS} steps"}


def to_inpaint_mask(mask_rgba: Image.Image) -> Image.Image:
    # MaskGenerator/DALL-E convention: alpha == 0 means "regenerate". diffusers wants white = regenerate.
    alpha = np.array(mask_rgba.convert("RGBA"))[:, :, 3]
    return Image.fromarray(np.where(alpha < 128, 255, 0).astype(np.uint8), mode="L")


def inpaint(image: Image.Image, mask: Image.Image, prompt: str, seed: int | None, usage: dict) -> Image.Image:
    size = image.size
    work_image = image.resize((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    work_mask = mask.resize((WORK_SIZE, WORK_SIZE), Image.NEAREST)
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    with _lock, resources.measure(device, usage):
        out = get_pipe()(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, image=work_image, mask_image=work_mask,
                         width=WORK_SIZE, height=WORK_SIZE, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         strength=STRENGTH, generator=generator).images[0]
    out = out.resize(size, Image.LANCZOS)
    # the VAE round-trip shifts every pixel slightly; restore the original outside the mask
    blend = mask.filter(ImageFilter.GaussianBlur(FEATHER)) if FEATHER else mask
    return Image.composite(out, image, blend)


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/v1/images/edits")
def edit_image(image: UploadFile, prompt: str, mask: UploadFile = None, n: int = 1, size: str = None,
               seed: int = None):
    img = Image.open(image.file).convert("RGB")
    m = to_inpaint_mask(Image.open(mask.file)) if mask else Image.new("L", img.size, 255)
    data, usage = [], new_usage()
    for i in range(n):
        t = time.time()
        result = inpaint(img, m, prompt, None if seed is None else seed + i, usage)
        print(f"inpainted {img.size} in {time.time() - t:.1f}s: {prompt!r}")
        data.append({"b64_json": to_b64(result)})
    return {"data": data, "usage": usage}


@app.post("/v1/images/generations")
def generate_image(prompt: str, n: int = 1, size: str = None, seed: int = None):
    # the paper's "no guide" baseline: inpaint a blank canvas with a full mask
    canvas, full = Image.new("RGB", (WORK_SIZE, WORK_SIZE), (127, 127, 127)), Image.new("L", (WORK_SIZE, WORK_SIZE), 255)
    usage = new_usage()
    data = [{"b64_json": to_b64(inpaint(canvas, full, prompt, None if seed is None else seed + i, usage))}
            for i in range(n)]
    return {"data": data, "usage": usage}


@app.get("/health")
def health():
    return {"model": MODEL_ID, "device": device, "loaded": _pipe is not None}


@app.get("/resources")
def resource_summary():
    # totals since this server started; per-dataset numbers are in the Gateway's cost ledger
    return {"model": MODEL_ID, "device": device, **resources.summary()}
