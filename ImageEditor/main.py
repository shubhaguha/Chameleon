import base64
import io
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from openai import AsyncOpenAI

load_dotenv()
app = FastAPI()

_aclient = None


def get_client() -> AsyncOpenAI:
    # created lazily so the service (and the rest of the stack) starts without a key
    global _aclient
    if _aclient is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in ImageEditor/.env")
        _aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                               organization=os.getenv("OPENAI_ORGANIZATION_NAME") or None)
    return _aclient

# DALL-E 2 (used in the paper) was retired on 2026-05-12; GPT Image models replace it.
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low")
# "high" keeps more of the guide image outside the mask (GPT Image masks are only guidance); costs more input tokens.
# Leave empty for models that don't support it (e.g. gpt-image-1-mini).
INPUT_FIDELITY = os.getenv("OPENAI_INPUT_FIDELITY", "high")


def _is_dall_e(model: str) -> bool:
    return model.startswith("dall-e")


def _normalize(response) -> dict:
    # GPT Image models only return b64_json; DALL-E returned URLs. Always hand back b64_json.
    data = []
    for d in response.data:
        b64 = d.b64_json if d.b64_json else base64.b64encode(requests.get(d.url).content).decode("utf-8")
        data.append({"b64_json": b64})
    return {"data": data}


def _named(upload_bytes: bytes, name: str):
    f = io.BytesIO(upload_bytes)
    f.name = name  # the SDK infers the mime type from the file name
    return f


@app.post("/v1/images/edits")
async def edit_image(image: UploadFile, prompt: str, mask: UploadFile = None, n: int = 4, size: str = None):
    kwargs = dict(model=IMAGE_MODEL, image=_named(image.file.read(), "image.png"),
                  prompt=prompt, n=n, size=IMAGE_SIZE if not _is_dall_e(IMAGE_MODEL) else (size or "512x512"))
    if mask:
        kwargs["mask"] = _named(mask.file.read(), "mask.png")
    if not _is_dall_e(IMAGE_MODEL):
        kwargs["quality"] = IMAGE_QUALITY
        # without this, a mask with transparent pixels can yield a transparent background
        kwargs["background"] = "opaque"
        if INPUT_FIDELITY:
            kwargs["input_fidelity"] = INPUT_FIDELITY
    return _normalize(await get_client().images.edit(**kwargs))


@app.post("/v1/images/generations")
async def generate_image(prompt: str, n: int = 4, size: str = None):
    kwargs = dict(model=IMAGE_MODEL, prompt=prompt, n=n,
                  size=IMAGE_SIZE if not _is_dall_e(IMAGE_MODEL) else (size or "512x512"))
    if not _is_dall_e(IMAGE_MODEL):
        kwargs["quality"] = IMAGE_QUALITY
        kwargs["background"] = "opaque"
    return _normalize(await get_client().images.generate(**kwargs))
