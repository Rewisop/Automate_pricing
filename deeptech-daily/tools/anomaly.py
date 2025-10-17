from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Try optional YAML if available
try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None  # graceful fallback

def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _load_yaml(path: str) -> Any:
    if not os.path.exists(path) or yaml is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:  # noqa: BLE001
        return None


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0


def _mad(xs: List[float], med: Optional[float] = None) -> float:
    if not xs:
        return 0.0
    m = _median(xs) if med is None else med
    dev = [abs(x - m) for x in xs]
    return _median(dev) or 0.0


def robust_z(today_value: float, history_values: List[float]) -> Optional[float]:
    """Median/MAD-based z score; returns None if not enough data."""

    vals = [v for v in history_values if isinstance(v, (int, float))]
    if len(vals) < 8:  # minimum history for stability
        return None
    med = _median(vals)
    mad = _mad(vals, med)
    if mad == 0:
        return None
    # 1.4826 approximates MAD->sigma for normal
    return round((today_value - med) / (1.4826 * mad), 2)


def _count_items(node: Any) -> int:
    if node is None:
        return 0
    if isinstance(node, list):
        return len(node)
    if isinstance(node, dict):
        return len(node)
    return 0


def collect_today_metrics(base_dir: str) -> Dict[str, float]:
    """Derive a few scalar metrics from existing exports."""

    data_dir = os.path.join(base_dir, "data")
    out: Dict[str, float] = {}

    # 1) GPU min price per GPU
    prices = _load_json(os.path.join(data_dir, "gpu_prices.json")) or []
    min_by_gpu: Dict[str, float] = {}
    for row in prices:
        try:
            g = str(row.get("gpu", "")).strip()
            p = float(row.get("usd_per_hour"))
        except Exception:  # noqa: BLE001
            continue
        if not g or p <= 0:
            continue
        if g not in min_by_gpu or p < min_by_gpu[g]:
            min_by_gpu[g] = p
    for g, p in min_by_gpu.items():
        key = f"gpu_min_usd_per_hr::{g}"
        out[key] = float(p)

    # 2) Ecosystem volumes (best-effort if YAML available)
    # Hugging Face trending models
    hf = _load_yaml(os.path.join(data_dir, "hf_trending.yaml"))
    out["hf_trending_count"] = float(_count_items(hf))

    # Github trending repos (AI/LLM)
    gh = _load_yaml(os.path.join(data_dir, "github_trending.yaml"))
    out["github_trending_count"] = float(_count_items(gh))

    # arXiv new submissions
    arx = _load_yaml(os.path.join(data_dir, "arxiv.yaml"))
    out["arxiv_new_count"] = float(_count_items(arx))

    # Hacker News AI posts
    hn = _load_yaml(os.path.join(data_dir, "hn_ai.yaml"))
    out["hn_ai_count"] = float(_count_items(hn))

    # CVEs related to AI/ML
    cves = _load_yaml(os.path.join(data_dir, "cves.yaml"))
    out["cves_count"] = float(_count_items(cves))

    return out


def update_timeseries(ts_path: str, today: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Timeseries file schema:
    {
      "series": {
        "<metric_key>": [{"date": "YYYY-MM-DD", "value": number}, ...]
      },
      "last_updated": "YYYY-MM-DD"
    }
    """

    ts = _load_json(ts_path) or {"series": {}, "last_updated": None}
    series: Dict[str, List[Dict[str, Any]]] = ts.get("series", {})
    for k, v in metrics.items():
        arr = series.get(k, [])
        # upsert today's value
        updated = False
        for row in arr:
            if row.get("date") == today:
                row["value"] = float(v)
                updated = True
                break
        if not updated:
            arr.append({"date": today, "value": float(v)})
        arr.sort(key=lambda r: r.get("date", ""))
        # keep last ~180 days
        series[k] = arr[-180:]
    ts["series"] = series
    ts["last_updated"] = today
    _save_json(ts_path, ts)
    return ts


def assess_anomalies(ts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of alerts with severity and z-scores."""

    alerts: List[Dict[str, Any]] = []
    series: Dict[str, List[Dict[str, Any]]] = ts.get("series", {})
    for key, arr in series.items():
        if not arr:
            continue
        vals = [float(x.get("value", 0.0)) for x in arr]
        z = robust_z(vals[-1], vals[:-1])
        if z is None:
            continue
        sev = None
        if abs(z) >= 4.0:
            sev = "REGIME"
        elif abs(z) >= 3.0:
            sev = "ALERT"
        elif abs(z) >= 2.0:
            sev = "WATCH"
        if not sev:
            continue
        alerts.append({
            "metric": key,
            "today": vals[-1],
            "z": z,
            "severity": sev,
        })
    # Sort: REGIME > ALERT > WATCH, by |z| desc
    order = {"REGIME": 0, "ALERT": 1, "WATCH": 2}
    alerts.sort(key=lambda a: (order.get(a["severity"], 3), -abs(a["z"]), a["metric"]))
    return alerts


def _pretty_metric_name(key: str) -> str:
    if key.startswith("gpu_min_usd_per_hr::"):
        return f"Min $/hr – {key.split('::', 1)[1]}"
    mapping = {
        "hf_trending_count": "Hugging Face Trending (count)",
        "github_trending_count": "GitHub Trending (count)",
        "arxiv_new_count": "arXiv New (count)",
        "hn_ai_count": "HN AI Posts (count)",
        "cves_count": "AI/ML CVEs (count)",
    }
    return mapping.get(key, key)


def render_radar_md(alerts: List[Dict[str, Any]], today: str) -> str:
    # choose top 6 alerts for brevity
    top = alerts[:6]
    lines = ["<!-- RADAR:START -->", "## Anomaly & Regime-Shift Radar", ""]
    if not top:
        lines.append("_No significant anomalies today._")
        lines.append("<!-- RADAR:END -->")
        return "\n".join(lines) + "\n"
    for a in top:
        name = _pretty_metric_name(a["metric"])
        z = a["z"]
        sev = a["severity"]
        if sev == "REGIME":
            badge = "🧨"
        elif sev == "ALERT":
            badge = "🔴"
        else:
            badge = "🟡"
        lines.append(f"- {badge} **{name}** — z={z} on {today}")
    lines.append("")
    lines.append("<sub>Robust z-scores via median/MAD over ~180 days. 🧨=regime-shift candidate (|z|≥4), 🔴=alert (|z|≥3), 🟡=watch (2≤|z|<3).</sub>")
    lines.append("<!-- RADAR:END -->")
    return "\n".join(lines) + "\n"
