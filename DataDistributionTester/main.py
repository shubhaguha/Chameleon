import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

from models import OCSVMModel, ImageEmbedder
from utils import download_embedder, create_set_hash

load_dotenv()

# embeddings of a training set are written here, so a restart doesn't re-embed the whole dataset
CACHE_DIR = os.getenv("DDT_CACHE_DIR", "/cache")


def init_embedder():
    download_embedder()


app = FastAPI(on_startup=[init_embedder])

active_embeddings = {}
active_models = {}


def get_train_embeddings(training_paths: set[str]):
    hash_str = create_set_hash(training_paths)
    if hash_str not in active_embeddings:
        cache_file = os.path.join(CACHE_DIR, f"{hash_str}.npy")
        if os.path.exists(cache_file):
            active_embeddings[hash_str] = np.load(cache_file)
        else:
            print(f"embedding {len(training_paths)} training images, this happens once per training set")
            active_embeddings[hash_str] = np.array(ImageEmbedder()(train_paths=training_paths))
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(cache_file, active_embeddings[hash_str])
    return hash_str, active_embeddings[hash_str]


@app.post("/v1/predict/")
async def get_prediction(image: UploadFile, training_paths: list[str], nu: float = 0.3, kernel: str = "rbf"):
    training_paths = set(training_paths[0].split(","))
    hash_str, train_embeddings = get_train_embeddings(training_paths)

    # the one-class SVM depends only on the training set and its parameters, so fit it once
    model_key = (hash_str, nu, kernel)
    if model_key not in active_models:
        ocsvm = OCSVMModel(nu=nu, kernel=kernel)
        ocsvm.train(train_embeddings)
        active_models[model_key] = ocsvm

    test_embeddings = ImageEmbedder()(binary_image=image.file.read())
    predictions = active_models[model_key].predict(test_embeddings)
    return {"prediction": True if int(predictions[0]) == 1 else False}
