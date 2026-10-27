import json
import os
import pickle

from fastapi import FastAPI, Query
from ucb import LinUCB

app = FastAPI()

STATE_PATH = os.getenv("UCB_STATE_PATH", "/state/linucb.pkl")

with open("config.json", 'r') as f:
    config = json.load(f)

# one bandit per dataset, since arms (attributes) and contexts (combinations) are dataset-specific
bandits: dict[str, LinUCB] = {}


def save_state():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "wb") as file:
        pickle.dump(bandits, file)
    os.replace(tmp_path, STATE_PATH)


@app.on_event("startup")
def startup_event():
    global bandits
    try:
        with open(STATE_PATH, "rb") as file:
            bandits = pickle.load(file)
        print("Bandit state loaded:", {k: v.num_arms for k, v in bandits.items()})
    except FileNotFoundError:
        print("No saved bandit state, starting fresh.")


def get_bandit(dataset: str, num_arms: int) -> LinUCB:
    if dataset not in bandits or bandits[dataset].num_arms != num_arms:
        bandits[dataset] = LinUCB(num_arms, config.get("alpha", 1.0))
    return bandits[dataset]


@app.get("/v1/ucb/arms/")
def get_arm(dataset: str, context: str, num_arms: int, allowed_arms: list[int] = Query(None)):
    arm = get_bandit(dataset, num_arms).choose_arm(context, allowed_arms)
    print(f"chosen arm is {arm}")
    return {"arm": arm}


@app.post("/v1/ucb/arms/")
def update_arm(dataset: str, context: str, num_arms: int, arm: int, reward: float):
    get_bandit(dataset, num_arms).update(arm, context, reward)
    save_state()
    return {"success": True}
