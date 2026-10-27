"""Cost ledger: one JSON line per foundation-model query and per reviewed image.

The paper counts repair cost in foundation-model queries. Every query is paid for, including ones whose image is
later rejected by the distribution test or a reviewer, so the ledger records every attempt. Each query row carries
the `usage` dict the image editor returned: OpenAI token counts and `cost_usd` from ImageEditor, or compute
measurements from another backend. Numeric usage fields are summed generically, so a backend can add fields
without changes here.

Rows live in <GENERATION_PATH>/<dataset_id>/cost_ledger.jsonl:
  {"kind": "query", "ts", "pattern", "strategy", "accuracy", "image", "ok", "error", "ddt_pass", "seconds", "usage"}
  {"kind": "review", "ts", "image", "accepted"}
"""
import datetime
import json
import os
from collections import defaultdict

LEDGER_NAME = "cost_ledger.jsonl"


def ledger_path(dataset_id: str, generation_path: str | None = None) -> str:
    return os.path.join(generation_path or os.getenv("GENERATION_PATH"), dataset_id, LEDGER_NAME)


def record(dataset_id: str, kind: str, **fields):
    path = ledger_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"kind": kind, "ts": datetime.datetime.now().isoformat(timespec="seconds"), **fields}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def total_cost_usd(path: str) -> float:
    return sum((r.get("usage") or {}).get("cost_usd") or 0 for r in read_rows(path) if r["kind"] == "query")


def _add_usage(totals: dict, usage: dict):
    for k, v in (usage or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            totals[k] = totals.get(k, 0) + v


def _per(totals: dict, n: int) -> dict:
    return {k: v / n for k, v in totals.items()} if n else {}


def _new_group():
    return {"queries": 0, "failed": 0, "ddt_rejected": 0, "reviewed": 0, "accepted": 0, "seconds": 0.0,
            "usage": {}}


def summarize(rows: list[dict]) -> dict:
    """Totals overall and per pattern, plus cost per accepted image (the number that matters for a repair)."""
    overall, by_pattern = _new_group(), defaultdict(_new_group)
    image_pattern, backends = {}, set()
    for r in rows:
        if r["kind"] != "query":
            continue
        for g in (overall, by_pattern[r["pattern"]]):
            g["queries"] += 1
            g["failed"] += 0 if r["ok"] else 1
            g["ddt_rejected"] += 1 if r["ok"] and r.get("ddt_pass") is False else 0
            g["seconds"] += r.get("seconds") or 0
            _add_usage(g["usage"], r.get("usage"))
        if r.get("image"):
            image_pattern[r["image"]] = r["pattern"]
        usage = r.get("usage") or {}
        if usage.get("backend"):
            backends.add(f'{usage["backend"]}:{usage.get("model", "")}')
    for r in rows:
        if r["kind"] != "review" or r["image"] not in image_pattern:
            continue
        for g in (overall, by_pattern[image_pattern[r["image"]]]):
            g["reviewed"] += 1
            g["accepted"] += 1 if r["accepted"] else 0
    for g in [overall, *by_pattern.values()]:
        g["queries_per_accepted"] = g["queries"] / g["accepted"] if g["accepted"] else None
        g["usage_per_accepted"] = _per(g["usage"], g["accepted"])
    return {"backends": sorted(backends), **overall, "by_pattern": dict(by_pattern)}
