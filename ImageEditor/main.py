import base64
import io
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from openai import AsyncOpenAI, BadRequestError

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

# DALL-E 2 (used in the paper) was retired on 2026-05-12 and gpt-image-1 on 2026-10-23.
# sunburst is OpenAI's recommendation "where editing precision matters most"; flare is faster at the same token price.
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low")
# "high" keeps more of the guide image outside the mask (GPT Image masks are only guidance); costs more input tokens.
# Dropped automatically if the model rejects it.
INPUT_FIDELITY = os.getenv("OPENAI_INPUT_FIDELITY", "high")
# Optional parameters that not every GPT Image model accepts; a 400 naming one of them is retried without it
# (rejected requests are not billed).
OPTIONAL_PARAMS = ("background", "input_fidelity")
# $ per 1M tokens, for the cost line in the log (gpt-image-2 / 2.5 standard rates)
PRICE_TEXT_IN = float(os.getenv("OPENAI_PRICE_TEXT_INPUT", 5.0))
PRICE_IMAGE_IN = float(os.getenv("OPENAI_PRICE_IMAGE_INPUT", 8.0))
PRICE_IMAGE_OUT = float(os.getenv("OPENAI_PRICE_IMAGE_OUTPUT", 30.0))


def _is_dall_e(model: str) -> bool:
    return model.startswith("dall-e")


def _normalize(response) -> dict:
    # GPT Image models only return b64_json; DALL-E returned URLs. Always hand back b64_json.
    data = []
    for d in response.data:
        b64 = d.b64_json if d.b64_json else base64.b64encode(requests.get(d.url).content).decode("utf-8")
        data.append({"b64_json": b64})
    return {"data": data}


async def _call(fn, kwargs: dict):
    while True:
        try:
            response = await fn(**kwargs)
        except BadRequestError as e:
            param = (getattr(e, "body", None) or {}).get("param") or ""
            dropped = next((p for p in OPTIONAL_PARAMS if p in kwargs and p in (param + str(e))), None)
            if dropped is None:
                raise
            print(f"{IMAGE_MODEL} rejected `{dropped}`; retrying without it")
            kwargs.pop(dropped)
            continue
        _log_usage(response)
        return response


def _log_usage(response):
    # GPT Image responses report token usage; log it so budgets can be checked against real numbers
    u = getattr(response, "usage", None)
    if not u:
        return
    details = getattr(u, "input_tokens_details", None)
    text_in = getattr(details, "text_tokens", 0) or 0
    image_in = getattr(details, "image_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cost = (text_in * PRICE_TEXT_IN + image_in * PRICE_IMAGE_IN + out * PRICE_IMAGE_OUT) / 1e6
    print(f"usage {IMAGE_MODEL} quality={IMAGE_QUALITY}: text_in={text_in} image_in={image_in} out={out} "
          f"~${cost:.4f}")


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
    return _normalize(await _call(get_client().images.edit, kwargs))


@app.post("/v1/images/generations")
async def generate_image(prompt: str, n: int = 4, size: str = None):
    kwargs = dict(model=IMAGE_MODEL, prompt=prompt, n=n,
                  size=IMAGE_SIZE if not _is_dall_e(IMAGE_MODEL) else (size or "512x512"))
    if not _is_dall_e(IMAGE_MODEL):
        kwargs["quality"] = IMAGE_QUALITY
        kwargs["background"] = "opaque"
    return _normalize(await _call(get_client().images.generate, kwargs))
