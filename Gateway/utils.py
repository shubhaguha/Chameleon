import io
import json
import os

from PIL import Image

from models import MUPEncoder


def store_png_files(dir, png_files_names, png_binaries):
    os.makedirs(dir, exist_ok=True)

    for name, bin in zip(png_files_names, png_binaries):
        with open(os.path.join(dir, name), "wb") as f:
            f.write(bin)


def store_mup(mup, filename, dir):
    os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, filename), "w") as f:
        f.write(json.dumps(mup, cls=MUPEncoder))


def resize_like(image: bytes, reference: bytes) -> bytes:
    # the paper maps generated images back to the data set's dimensions (§6.1); GPT Image models return 1024px
    ref_size = Image.open(io.BytesIO(reference)).size
    img = Image.open(io.BytesIO(image)).convert("RGB")
    if img.size != ref_size:
        img = img.resize(ref_size, Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def load_image(filename, parent_ds, is_generated: bool):
    # TODO: Duplicate logic with get_image API,
    #       needs to be merged with get_image API as a core functionality that provides image data

    path = os.getenv("RESOURCES_PATH") if not is_generated else os.getenv("GENERATION_PATH")
    with open(os.path.join(path, parent_ds, filename), "rb") as f:
        data = f.read()

    return data


def convert_list_to_dict(l: list) -> dict:
    q = {}
    if l is None:
        return q

    for i in l:
        k, v = i.split("=")
        q[k] = int(v)
    return q
