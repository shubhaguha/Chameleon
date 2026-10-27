"""Compute-resource tracking for the local backend.

Two views of the same run:
- per request, `measure()` returns a `usage` dict that the Gateway writes to the dataset's cost ledger
  (inference and CPU seconds, peak GPU/unified memory, optional energy and dollar estimates);
- for the whole run, a sampler thread appends a row every RESOURCE_SAMPLE_SECONDS to
  logs/resources_<start>.csv, and `summary()` (GET /resources) reports totals since the server started.

Energy is not measured: reading Apple GPU power needs `sudo powermetrics`. Set INPAINT_POWER_WATTS to the
machine's draw under load to get an estimate, and ELECTRICITY_USD_PER_KWH to turn it into cost_usd.
"""
import csv
import datetime
import os
import threading
import time
from contextlib import contextmanager

import psutil
import torch

SAMPLE_SECONDS = float(os.getenv("RESOURCE_SAMPLE_SECONDS", 5))
POWER_WATTS = float(os.getenv("INPAINT_POWER_WATTS")) if os.getenv("INPAINT_POWER_WATTS") else None
USD_PER_KWH = float(os.getenv("ELECTRICITY_USD_PER_KWH")) if os.getenv("ELECTRICITY_USD_PER_KWH") else None
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

_process = psutil.Process()
_state_lock = threading.Lock()
_busy = 0  # requests currently generating
_call_peak_gb = 0.0  # peak device memory seen during the current request
_started = time.time()
_totals = {"images": 0, "inference_seconds": 0.0, "cpu_seconds": 0.0, "peak_device_memory_gb": 0.0,
           "peak_rss_gb": 0.0, "busy_seconds": 0.0}


def device_memory_gb(device: str) -> float:
    # MPS shares unified memory with the CPU; driver_allocated_memory is what the Metal driver holds for torch
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _note_peak(gb: float):
    global _call_peak_gb
    with _state_lock:
        _call_peak_gb = max(_call_peak_gb, gb)
        _totals["peak_device_memory_gb"] = max(_totals["peak_device_memory_gb"], gb)


@contextmanager
def measure(device: str, usage: dict):
    """Wrap one generation; fills `usage` with what it consumed."""
    global _busy, _call_peak_gb
    with _state_lock:
        _busy += 1
        _call_peak_gb = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cpu0, t0 = _process.cpu_times(), time.monotonic()
    try:
        yield
    finally:
        seconds = time.monotonic() - t0
        cpu1 = _process.cpu_times()
        _note_peak(torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else device_memory_gb(device))
        rss = _process.memory_info().rss / 1e9
        with _state_lock:
            _busy -= 1
            usage["images"] = usage.get("images", 0) + 1
            usage["inference_seconds"] = round(usage.get("inference_seconds", 0) + seconds, 2)
            usage["cpu_seconds"] = round(usage.get("cpu_seconds", 0) +
                                         (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system), 2)
            usage["peak_device_memory_gb"] = round(max(usage.get("peak_device_memory_gb", 0), _call_peak_gb), 2)
            usage["peak_rss_gb"] = round(max(usage.get("peak_rss_gb", 0), rss), 2)
            if POWER_WATTS is not None:
                usage["energy_kwh_est"] = usage.get("energy_kwh_est", 0) + POWER_WATTS * seconds / 3.6e6
                if USD_PER_KWH is not None:
                    usage["cost_usd"] = usage["energy_kwh_est"] * USD_PER_KWH
            _totals["images"] += 1
            _totals["inference_seconds"] += seconds
            _totals["cpu_seconds"] += (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
            _totals["peak_rss_gb"] = max(_totals["peak_rss_gb"], rss)


def _sampler(device: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"resources_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    print(f"logging resource samples every {SAMPLE_SECONDS:g}s to {path}")
    _process.cpu_percent()  # first call only primes the counter
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "busy", "process_cpu_percent", "process_rss_gb", "device_memory_gb",
                         "system_memory_percent", "system_cpu_percent"])
        while True:
            time.sleep(SAMPLE_SECONDS)
            gb = device_memory_gb(device)
            _note_peak(gb)
            with _state_lock:
                busy = _busy > 0
                if busy:
                    _totals["busy_seconds"] += SAMPLE_SECONDS
            writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"), int(busy),
                             _process.cpu_percent(), round(_process.memory_info().rss / 1e9, 2), round(gb, 2),
                             psutil.virtual_memory().percent, psutil.cpu_percent()])
            f.flush()


def start_sampler(device: str):
    threading.Thread(target=_sampler, args=(device,), daemon=True).start()


def summary() -> dict:
    with _state_lock:
        t = dict(_totals)
    uptime = time.time() - _started
    t["uptime_seconds"] = round(uptime)
    # share of wall time spent generating, from the sampler (so resolution is RESOURCE_SAMPLE_SECONDS)
    t["utilization"] = round(t["busy_seconds"] / uptime, 3) if uptime else 0
    t["inference_seconds_per_image"] = round(t["inference_seconds"] / t["images"], 1) if t["images"] else None
    if POWER_WATTS is not None:
        t["energy_kwh_est"] = POWER_WATTS * t["inference_seconds"] / 3.6e6
        if USD_PER_KWH is not None:
            t["cost_usd"] = t["energy_kwh_est"] * USD_PER_KWH
    return t
