from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable

SPARK_BARS = "▁▂▃▄▅▆▇█"

def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _tflops_weight(gpu_name: str) -> float:
    """
    Very light, hardcoded TFLOPS reference to normalize prices.
    Conservative and extensible without external deps.
    Values approximate FP16 where relevant. Extend as needed.
    """
    name = gpu_name.lower()
    # Order matters: check more specific names first.
    table = [
        ("h100", 1000.0),
        ("h200", 1200.0),
        ("a100 80", 624.0),
        ("a100", 624.0),
        ("a10g", 125.0),
        ("a10", 125.0),
        ("l40s", 362.0),
        ("l40", 181.0),
        ("v100 32", 250.0),
        ("v100", 200.0),
        ("t4", 65.0),
        ("rtx 4090", 330.0),
        ("4090", 330.0),
        ("rtx 4080", 200.0),
        ("4080", 200.0),
        ("rtx 3090", 142.0),
        ("3090", 142.0),
        ("rtx 3080", 97.0),
        ("3080", 97.0),
        ("a6000", 160.0),
    ]
    for key, val in table:
        if key in name:
            return val
    # Fallback to a modest baseline to avoid division by zero.
    return 100.0

def compute_min_prices_by_gpu(prices: Iterable[Dict]) -> Dict[str, Tuple[float, str]]:
    """
    Input rows: {"gpu": str, "usd_per_hour": float, "source": str}
    Returns: {gpu_name: (min_price, provider)}
    """
    out: Dict[str, Tuple[float, str]] = {}
    for row in prices:
        try:
            gpu = str(row["gpu"]).strip()
            p = float(row["usd_per_hour"])
            src = str(row.get("source", "unknown")).strip() or "unknown"
        except Exception:
            continue
        if gpu not in out or p < out[gpu][0]:
            out[gpu] = (p, src)
    return out

def compute_dpi(min_prices: Dict[str, Tuple[float, str]]) -> float:
    """
    DeepTech GPU Price Index (DPI):
    Weighted harmonic mean of TFLOPS-normalized $/hr:
      normalized_cost_g = usd_per_hour / TFLOPS(g)
    DPI = 1 / mean(normalized_cost_g)  (units: TFLOPS per $/hr)
    Higher DPI => more TFLOPS per dollar (better market value).
    """
    vals: List[float] = []
    for gpu, (price, _) in min_prices.items():
        t = _tflops_weight(gpu)
        if price > 0 and t > 0:
            vals.append(price / t)
    if not vals:
        return 0.0
    mean_norm = sum(vals) / len(vals)
    if mean_norm <= 0:
        return 0.0
    dpi = 1.0 / mean_norm
    return round(dpi, 3)

def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(path: str, history: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def upsert_history(history: List[Dict], date_str: str, dpi: float) -> List[Dict]:
    found = False
    for row in history:
        if row.get("date") == date_str:
            row["dpi"] = dpi
            found = True
            break
    if not found:
        history.append({"date": date_str, "dpi": dpi})
    # Sort chronologically
    history.sort(key=lambda r: r.get("date", ""))
    # Trim to last ~180 days to keep file small
    return history[-180:]

def pct_change(series: List[float], lookback: int) -> float | None:
    if len(series) <= lookback:
        return None
    now = series[-1]
    past = series[-1 - lookback]
    if past == 0:
        return None
    return round(((now - past) / past) * 100.0, 2)

def make_sparkline(series: List[float], width: int = 45) -> str:
    if not series:
        return ""
    # Downsample to ~width points
    if len(series) > width:
        step = len(series) / width
        xs = []
        i = 0.0
        while len(xs) < width:
            xs.append(series[int(i)])
            i += step
    else:
        xs = series[:]
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return SPARK_BARS[0] * len(xs)
    out_chars = []
    for v in xs:
        idx = int((v - lo) / (hi - lo) * (len(SPARK_BARS) - 1))
        out_chars.append(SPARK_BARS[idx])
    return "".join(out_chars)

def cheapest_gpu(min_prices: Dict[str, Tuple[float, str]]) -> Tuple[str, float, str]:
    best_gpu, best_price, best_src = None, float("inf"), ""
    for gpu, (p, src) in min_prices.items():
        if p < best_price:
            best_gpu, best_price, best_src = gpu, p, src
    return best_gpu or "unknown", (best_price if best_price != float('inf') else 0.0), best_src or "unknown"
