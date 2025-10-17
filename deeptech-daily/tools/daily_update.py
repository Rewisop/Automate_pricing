"""End-to-end automation for the DeepTech Daily report.

This module provides a single entry point that orchestrates fetching data from a
handful of public APIs (GPU pricing, arXiv, Hugging Face, GitHub, Papers with
Code, Hacker News, CVEs). Each `collect_*` function is responsible for a single
data source and returns both the machine readable payload and a Markdown
rendering. The `main()` function aggregates the results, writes structured files
to :mod:`deeptech-daily/data` and refreshes the Markdown sections in
``deeptech-daily/README.md``.

The script is intentionally dependency-light so that it can run inside scheduled
CI jobs as well as on local developer machines.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
import yaml
from huggingface_hub import HfApi
from tabulate import tabulate

GPU_NAME_ALIASES = {
    "H100": ["H100"],
    "A100": ["A100"],
    "A10": ["A10"],
    "A40": ["A40"],
    "A30": ["A30"],
    "A16": ["A16"],
    "A2": ["A2"],
    "L4": ["L4"],
    "L40": ["L40"],
    "L40S": ["L40S", "L40 S"],
    "T4": ["T4"],
    "V100": ["V100"],
    "P100": ["P100"],
    "P4": ["P4"],
    "P40": ["P40"],
    "K80": ["K80"],
    "M60": ["M60"],
    "RTX 4090": ["4090", "RTX4090", "RTX 4090"],
    "RTX 3090": ["3090", "RTX3090", "RTX 3090"],
    "RTX 6000": ["RTX 6000", "A6000", "6000"],
    "RTX 5000": ["RTX 5000", "5000"],
}

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
README_PATH = BASE_DIR / "README.md"
GPU_PROVIDERS_PATH = BASE_DIR / "providers" / "gpu_sources.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UTC = timezone.utc
NOW = datetime.now(UTC)
HEADERS = {
    "User-Agent": "deeptech-daily-bot/1.0",
    "Accept": "application/json",
}
HN_KEYWORDS = re.compile(r"\b(ai|llm|gpt|anthropic|deepmind|openai|model|inference)\b", re.IGNORECASE)
MAX_TEXT_LENGTH = 160
REQUEST_TIMEOUT = (5, 15)
SESSION = requests.Session()
SESSION.trust_env = False
LOG_LEVEL = os.getenv("DEEPTECH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("deeptech_daily")
HEX_PTR = re.compile(r"0x[0-9a-fA-F]+")


def instrumented(name: Optional[str] = None):
    """Log entry and exit for the wrapped callable."""

    def decorator(func):
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            LOGGER.info("Starting %s", label)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed %s: %s", label, exc)
                raise
            duration = time.perf_counter() - start
            LOGGER.info("Finished %s in %.2fs", label, duration)
            return result

        return wrapper

    return decorator


def retryable(*, attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: Tuple[type[Exception], ...] = (requests.RequestException, RuntimeError, ValueError)):
    """Retry the wrapped callable with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    if attempt == attempts:
                        LOGGER.error("Giving up after %s attempts calling %s", attempts, func.__name__)
                        raise
                    LOGGER.warning(
                        "Attempt %s/%s for %s failed (%s); retrying in %.1fs",
                        attempt,
                        attempts,
                        func.__name__,
                        format_exception(exc),
                        wait,
                    )
                    time.sleep(wait)
                    wait *= backoff

        return wrapper

    return decorator


@dataclass
class SectionResult:
    """Container returned by each section collector.

    Attributes
    ----------
    items:
        Machine-readable objects ready to be serialised to JSON/YAML.
    readme:
        Markdown representation injected between README markers.
    error:
        Optional description used to surface partial failures to the user.
    """

    items: List[Dict[str, Any]]
    readme: str
    error: Optional[str] = None


def truncate(text: str, limit: int = MAX_TEXT_LENGTH) -> str:
    """Collapse whitespace and shorten long text blocks for table output."""

    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def isoformat(dt: datetime) -> str:
    """Normalise datetimes to a Z-suffixed ISO 8601 string."""

    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text_if_changed(path: Path, content: str) -> bool:
    encoded = content.rstrip() + "\n"
    if path.exists():
        existing = path.read_text()
        if existing == encoded:
            return False
    path.write_text(encoded)
    return True


def canonical_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonical_hashable(v) for k, v in sorted(value.items()) if k not in {"generated_at", "fetched_at", "hash"}}
    if isinstance(value, list):
        return [canonical_hashable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(canonical_hashable(v) for v in value)
    return value


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    canonical = canonical_hashable(payload)
    serialised = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def enrich_with_provenance(payload: Dict[str, Any], source_url: Optional[Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched_hash = compute_payload_hash(enriched)
    if source_url is not None:
        enriched["source_url"] = source_url
    enriched["fetched_at"] = isoformat(NOW)
    enriched["hash"] = enriched_hash
    return enriched


def write_json_if_changed(path: Path, payload: Dict[str, Any], source_url: Optional[Any] = None) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = enrich_with_provenance(payload, source_url)
    payload = preserve_metadata(path, payload, json.loads)
    serialised = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return write_text_if_changed(path, serialised)


def write_yaml_if_changed(path: Path, payload: Dict[str, Any], source_url: Optional[Any] = None) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = enrich_with_provenance(payload, source_url)
    payload = preserve_metadata(path, payload, yaml.safe_load)
    serialised = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return write_text_if_changed(path, serialised)


@retryable()
def fetch_json(url: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: Tuple[int, int] = REQUEST_TIMEOUT) -> Any:
    response = SESSION.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def preserve_metadata(path: Path, payload: Dict[str, Any], loader) -> Dict[str, Any]:
    if not path.exists():
        return payload
    try:
        existing = loader(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return payload

    def strip_metadata(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: strip_metadata(v)
                for k, v in data.items()
                if k not in {"generated_at", "fetched_at", "hash"}
            }
        if isinstance(data, list):
            return [strip_metadata(item) for item in data]
        return data

    if strip_metadata(existing) == strip_metadata(payload):
        payload = dict(payload)
        for key in ("generated_at", "fetched_at"):
            if key in existing:
                payload[key] = existing[key]
    return payload


def format_exception(exc: Exception) -> str:
    text = str(exc)
    text = HEX_PTR.sub("0xXXXX", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_cached_payload(path: Path) -> Optional[Dict[str, Any]]:
    """Return the previously written dataset for ``path`` if it exists."""

    if not path.exists():
        return None
    try:
        raw = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to load cached payload for \"%s\"", path, exc_info=True)
        return None
    return data or {}


def cached_items(path: Path, item_key: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Fetch cached items and the timestamp they were last refreshed."""

    cached = load_cached_payload(path)
    if not cached:
        return [], None
    items = cached.get(item_key) or []
    if not isinstance(items, list):
        return [], cached.get("fetched_at") or cached.get("generated_at")
    timestamp = cached.get("fetched_at") or cached.get("generated_at")
    return items, timestamp


def stale_notice(message: str, timestamp: Optional[str]) -> str:
    """Render a short markdown notice describing stale content."""

    details = truncate(message, 120) if message else "upstream error"
    if timestamp:
        return f"_Showing cached data from {timestamp}. Refresh failed: {details}._"
    return f"_Feed unavailable. Refresh failed: {details}._"


def update_readme_sections(sections: Dict[str, str]) -> bool:
    if not README_PATH.exists():
        raise FileNotFoundError(f"README not found at {README_PATH}")
    original = README_PATH.read_text()
    updated = original
    for marker, content in sections.items():
        block = f"<!--{marker}:START-->"
        end_block = f"<!--{marker}:END-->"
        if block not in updated or end_block not in updated:
            raise ValueError(f"Missing markers for {marker}")
        replacement = f"{block}\n{content}\n{end_block}"
        pattern = re.compile(rf"{re.escape(block)}.*?{re.escape(end_block)}", re.DOTALL)
        updated = pattern.sub(replacement, updated)
    if updated == original:
        return False
    README_PATH.write_text(updated)
    return True


def load_gpu_sources() -> List[Dict[str, Any]]:
    if not GPU_PROVIDERS_PATH.exists():
        return []
    config = yaml.safe_load(GPU_PROVIDERS_PATH.read_text()) or {}
    return config.get("sources", [])


def parse_vastai(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = source.get("base_url")
    params = {"limit": 200, "skip": 0}
    data = fetch_json(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    offers = data.get("offers", [])
    parsed: List[Dict[str, Any]] = []
    for offer in offers:
        gpu_name = offer.get("gpu_name") or offer.get("gpu_name_id") or "Unknown"
        price = offer.get("dph_total") or offer.get("dph") or offer.get("usd_per_hour")
        if price is None:
            continue
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            continue
        parsed.append({
            "gpu": gpu_name,
            "usd_per_hour": round(price_val, 4),
            "source": source.get("name", "Vast.ai"),
        })
    return parsed


@instrumented("GPU pricing")
def collect_gpu_prices() -> SectionResult:
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    sources = load_gpu_sources()
    for source in sources:
        if not source.get("enabled"):
            continue
        parser_name = source.get("parser")
        parser = globals().get(parser_name)
        if not parser:
            errors.append(f"Missing parser {parser_name}")
            continue
        try:
            entries.extend(parser(source))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source.get('name')}: {format_exception(exc)}")
    aggregated: Dict[str, Dict[str, Any]] = {}
    for item in entries:
        gpu = item.get("gpu", "Unknown")
        price = item.get("usd_per_hour")
        if price is None:
            continue
        current = aggregated.get(gpu)
        if not current or price < current["usd_per_hour"]:
            aggregated[gpu] = {"gpu": gpu, "usd_per_hour": round(float(price), 4), "source": item.get("source", "Vast.ai")}
    sorted_items = sorted(aggregated.values(), key=lambda x: (x["gpu"].lower(), x["usd_per_hour"]))
    cache_path = DATA_DIR / "gpu_prices.json"
    cached_gpus, cached_ts = cached_items(cache_path, "gpus")
    fallback_used = False
    if not sorted_items and cached_gpus:
        sorted_items = sorted(cached_gpus, key=lambda x: (x.get("gpu", "").lower(), x.get("usd_per_hour", float("inf"))))
        fallback_used = True
    table = "No GPU pricing data available."
    if sorted_items:
        rows = [(item.get("gpu", "Unknown"), f"${item.get('usd_per_hour', 0):.4f}") for item in sorted_items]
        table = tabulate(rows, headers=["GPU", "Min USD/hr"], tablefmt="github")
    error_text = "; ".join(errors) if errors else None
    if fallback_used:
        notice = stale_notice(error_text or "no live GPU providers responded", cached_ts)
        table = f"{table}\n\n{notice}" if table else notice
    elif error_text:
        table += f"\n\n_Warnings: {truncate(error_text, 120)}_"
    if not fallback_used:
        payload = {
            "generated_at": isoformat(NOW),
            "sources": [src for src in sources],
            "gpus": sorted_items,
            "errors": errors,
        }
        source_urls: List[str] = [src.get("base_url") for src in sources if src.get("base_url")]
        write_json_if_changed(cache_path, payload, source_urls or None)
    return SectionResult(sorted_items, table, error_text)

from deeptech_daily.tools.metrics import (
    compute_min_prices_by_gpu,
    compute_dpi,
    load_history,
    save_history,
    upsert_history,
    pct_change,
    make_sparkline,
    utc_date,
    cheapest_gpu,
)

DPI_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dpi_history.json")
README_PATH = Path(os.path.join(os.path.dirname(__file__), "..", "README.md"))


def _safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def update_dpi_section():
    prices = _safe_read_json(os.path.join(os.path.dirname(__file__), "..", "data", "gpu_prices.json")) or []
    min_prices = compute_min_prices_by_gpu(prices)
    dpi_value = compute_dpi(min_prices)
    today = utc_date()

    history = load_history(DPI_HISTORY_PATH)
    history = upsert_history(history, today, dpi_value)
    save_history(DPI_HISTORY_PATH, history)

    series = [row["dpi"] for row in history]
    spark = make_sparkline(series, width=45)
    wow = pct_change(series, 7)
    mo30 = pct_change(series, 30)
    cheap_gpu, cheap_price, cheap_src = cheapest_gpu(min_prices)

    # Prepare markdown fragment
    def fmt_pct(pct):
        if pct is None:
            return "n/a"
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct}%"

    md = []
    md.append("<!-- DPI:START -->")
    md.append("## DeepTech GPU Price Index (DPI)")
    md.append("")
    md.append(f"**Today:** `{dpi_value}`  |  **7d:** `{fmt_pct(wow)}`  |  **30d:** `{fmt_pct(mo30)}`  ")
    md.append("")
    md.append(f"`{spark}`")
    md.append("")
    md.append(f"**Cheapest now:** **{cheap_gpu}** at `${cheap_price:.3f}/hr` via **{cheap_src}**.")
    md.append("")
    md.append("<sub>DPI is TFLOPS-per-$/hr (higher is better). Computed from daily minimum observed prices per GPU.</sub>")
    md.append("<!-- DPI:END -->")
    fragment = "\n".join(md).strip() + "\n"

    # Inject or replace in README
    readme = ""
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme = f.read()
    except FileNotFoundError:
        readme = ""

    start_tag = "<!-- DPI:START -->"
    end_tag = "<!-- DPI:END -->"
    if start_tag in readme and end_tag in readme:
        pre, rest = readme.split(start_tag, 1)
        _, post = rest.split(end_tag, 1)
        new_readme = pre + fragment + post
    else:
        # Insert near top under H1 if possible
        if "# " in readme:
            parts = readme.split("\n", 2)
            # keep first line (title) and insert after
            if len(parts) >= 2:
                new_readme = parts[0] + "\n" + parts[1] + "\n\n" + fragment + (parts[2] if len(parts) == 3 else "")
            else:
                new_readme = readme + "\n\n" + fragment
        else:
            new_readme = fragment + "\n" + readme

    if new_readme != readme:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(new_readme)


def parse_feed_entry(entry: Any) -> Tuple[str, datetime]:
    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if published_parsed:
        dt = datetime(*published_parsed[:6], tzinfo=UTC)
    else:
        dt = NOW
    return entry.get("id") or entry.get("link"), dt


@instrumented("arXiv digest")
def collect_arxiv() -> SectionResult:
    feeds = [
        "https://export.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/rss/cs.CL",
        "https://export.arxiv.org/rss/cs.LG",
    ]
    items: Dict[str, Dict[str, Any]] = {}
    cache_path = DATA_DIR / "arxiv.yaml"
    parsed_any = False
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:  # noqa: BLE001
            continue
        entries = getattr(parsed, "entries", []) or []
        if entries:
            parsed_any = True
        for entry in entries:
            entry_id, published = parse_feed_entry(entry)
            if not entry_id:
                continue
            published_iso = isoformat(published)
            if entry_id in items and items[entry_id]["published"] >= published_iso:
                continue
            items[entry_id] = {
                "id": entry_id,
                "title": truncate(entry.get("title", "Untitled")),
                "summary": truncate(entry.get("summary", "")),
                "link": entry.get("link"),
                "published": published_iso,
            }
    sorted_items = sorted(items.values(), key=lambda x: x["published"], reverse=True)[:15]
    bullets = [f"- [{item['title']}]({item['link']}) — {item['summary']}" for item in sorted_items]
    section = "\n".join(bullets) if bullets else "No recent arXiv updates."
    if not sorted_items:
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        message = "no arXiv entries retrieved"
        if cached_items_list:
            fallback_bullets = [
                f"- [{item['title']}]({item['link']}) — {item['summary']}" for item in cached_items_list
            ]
            fallback_section = "\n".join(fallback_bullets) if fallback_bullets else section
            notice = stale_notice(message, cached_ts)
            if fallback_section:
                fallback_section = f"{fallback_section}\n\n{notice}"
            else:
                fallback_section = notice
            return SectionResult(cached_items_list, fallback_section, message)
        if not parsed_any:
            return SectionResult([], stale_notice(message, cached_ts), message)
    payload = {
        "generated_at": isoformat(NOW),
        "items": sorted_items,
    }
    write_yaml_if_changed(cache_path, payload, feeds)
    return SectionResult(sorted_items, section)


def hf_api_client(token: Optional[str] = None) -> HfApi:
    if token is None:
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    return HfApi(token=token)


@instrumented("HF trending models")
def collect_hf_models() -> SectionResult:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    @retryable()
    def fetch(current_token: Optional[str]):
        api_client = hf_api_client(current_token)
        return list(api_client.list_models(sort="downloads", direction=-1, limit=15))

    cache_path = DATA_DIR / "hf_trending.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared: List[Dict[str, Any]] = []
        for model in items:
            prepared.append(
                {
                    "model_id": model.get("model_id"),
                    "downloads": model.get("downloads"),
                    "likes": model.get("likes"),
                    "tags": sorted(model.get("tags", []) or []),
                    "last_modified": model.get("last_modified"),
                }
            )
        prepared.sort(key=lambda x: (-(x.get("downloads") or 0), x.get("model_id") or ""))
        rows = [
            [item.get("model_id"), item.get("downloads", 0), item.get("likes", 0)] for item in prepared
        ]
        table = tabulate(rows, headers=["Model", "Downloads", "Likes"], tablefmt="github") if rows else "No trending models."
        if notice:
            table = f"{table}\n\n{notice}"
        return prepared, table

    try:
        models = fetch(token)
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        if token:
            try:
                models = fetch(None)
            except Exception as fallback_exc:  # noqa: BLE001
                message = format_exception(fallback_exc)
        else:
            models = None
        if models is None:
            cached_items_list, cached_ts = cached_items(cache_path, "items")
            if cached_items_list:
                prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
                return SectionResult(prepared, table, message)
            payload = {
                "generated_at": isoformat(NOW),
                "items": [],
                "error": message,
            }
            write_yaml_if_changed(cache_path, payload, "https://huggingface.co/api/models")
            return SectionResult([], stale_notice(message, cached_ts), message)

    if not isinstance(models, list):
        models = list(models)
    items: List[Dict[str, Any]] = []
    for model in models:
        model_id = getattr(model, "modelId", None) or getattr(model, "id", None)
        last_modified = getattr(model, "lastModified", None) or getattr(model, "last_modified", None)
        if isinstance(last_modified, datetime):
            lm = isoformat(last_modified)
        elif isinstance(last_modified, str):
            try:
                lm = isoformat(datetime.fromisoformat(last_modified.replace("Z", "+00:00")))
            except ValueError:
                lm = last_modified
        else:
            lm = None
        items.append({
            "model_id": model_id,
            "downloads": getattr(model, "downloads", None),
            "likes": getattr(model, "likes", None),
            "tags": sorted(getattr(model, "tags", []) or []),
            "last_modified": lm,
        })
    items, table = render(items)
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(cache_path, payload, "https://huggingface.co/api/models")
    return SectionResult(items, table)


def github_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github+json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@instrumented("GitHub trending repos")
def collect_github_trending() -> SectionResult:
    since = (NOW - timedelta(days=7)).date().isoformat()
    query = "(LLM OR \"large language model\" OR genai) created:>" + since
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 20,
    }
    cache_path = DATA_DIR / "github_trending.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared: List[Dict[str, Any]] = []
        for repo in items:
            prepared.append(
                {
                    "full_name": repo.get("full_name"),
                    "html_url": repo.get("html_url"),
                    "description": truncate(repo.get("description") or ""),
                    "stargazers_count": repo.get("stargazers_count", 0),
                    "created_at": repo.get("created_at"),
                }
            )
        prepared.sort(key=lambda x: (-x.get("stargazers_count", 0), x.get("full_name") or ""))
        top_items = prepared[:20]
        rows = [
            [item.get("full_name"), item.get("stargazers_count", 0), item.get("description", "")] for item in top_items
        ]
        table = tabulate(rows, headers=["Repository", "Stars", "Description"], tablefmt="github") if rows else "No trending repositories."
        if notice:
            table = f"{table}\n\n{notice}"
        return top_items, table

    @retryable()
    def fetch() -> Dict[str, Any]:
        response = SESSION.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=github_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in {403, 429}:
            raise RuntimeError(f"GitHub API limit {response.status_code}")
        response.raise_for_status()
        return response.json()

    try:
        data = fetch()
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        if cached_items_list:
            prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, table, message)
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": message,
        }
        write_yaml_if_changed(
            cache_path,
            payload,
            "https://api.github.com/search/repositories",
        )
        return SectionResult([], stale_notice(message, cached_ts), message)
    repos = data.get("items", [])
    items: List[Dict[str, Any]] = []
    for repo in repos:
        items.append({
            "full_name": repo.get("full_name"),
            "html_url": repo.get("html_url"),
            "description": truncate(repo.get("description") or ""),
            "stargazers_count": repo.get("stargazers_count", 0),
            "created_at": repo.get("created_at"),
        })
    items, table = render(items)
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(
        cache_path,
        payload,
        "https://api.github.com/search/repositories",
    )
    return SectionResult(items, table)


@instrumented("Papers with Code feed")
def collect_papers_with_code() -> SectionResult:
    url = "https://paperswithcode.com/api/v1/papers/"
    params = {
        "q": "LLM OR \"large language model\"",
        "ordering": "-published",
        "items_per_page": 20,
    }
    cache_path = DATA_DIR / "pwc_llm.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared = sorted(items, key=lambda x: x.get("published", ""), reverse=True)
        bullets = [
            f"- [{item['title']}]({item['paper_url']}) — {truncate((item.get('authors') or '')[:120])}"
            for item in prepared
        ]
        section = "\n".join(bullets) if bullets else "No recent papers found."
        if notice:
            section = f"{section}\n\n{notice}" if section else notice
        return prepared, section

    try:
        data = fetch_json(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        if cached_items_list:
            prepared, section = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, section, message)
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": message,
        }
        write_yaml_if_changed(
            cache_path,
            payload,
            "https://paperswithcode.com/api/v1/papers/",
        )
        return SectionResult([], stale_notice(message, cached_ts), message)
    now_minus_7 = NOW - timedelta(days=7)
    items: List[Dict[str, Any]] = []
    for paper in data.get("results", []):
        published_raw = paper.get("published")
        try:
            published_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00")) if published_raw else None
        except Exception:  # noqa: BLE001
            published_dt = None
        if published_dt and published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=UTC)
        if published_dt and published_dt < now_minus_7:
            continue
        items.append({
            "title": truncate(paper.get("title") or "Untitled"),
            "paper_url": paper.get("url_abs") or paper.get("url_pdf"),
            "published": isoformat(published_dt or NOW),
            "authors": paper.get("authors"),
        })
    items, section = render(items)
    if not items:
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        message = "no recent Papers with Code results"
        if cached_items_list:
            prepared, fallback_section = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, fallback_section, message)
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(
        cache_path,
        payload,
        "https://paperswithcode.com/api/v1/papers/",
    )
    return SectionResult(items, section)


@instrumented("Hacker News highlights")
def collect_hn_ai() -> SectionResult:
    cache_path = DATA_DIR / "hn_ai.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared = sorted(items, key=lambda x: (-x.get("score", 0), x.get("time", ""), x.get("id")))
        bullets = [f"- [{item['title']}]({item['url']}) — {item['score']} points" for item in prepared[:15]]
        section = "\n".join(bullets) if bullets else "No AI-related stories in the last 24h."
        if notice:
            section = f"{section}\n\n{notice}" if section else notice
        return prepared, section

    try:
        top_ids = fetch_json(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        if cached_items_list:
            prepared, section = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, section, message)
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": message,
        }
        write_yaml_if_changed(
            cache_path,
            payload,
            "https://hacker-news.firebaseio.com/",
        )
        return SectionResult([], stale_notice(message, cached_ts), message)
    if not isinstance(top_ids, list):
        top_ids = []
    window_start = NOW - timedelta(hours=24)
    items: List[Dict[str, Any]] = []
    for story_id in top_ids[:200]:
        try:
            item = fetch_json(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception:
            continue
        if not item or item.get("type") != "story":
            continue
        title = item.get("title") or ""
        if not HN_KEYWORDS.search(title):
            continue
        ts = item.get("time")
        if ts is None:
            continue
        published = datetime.fromtimestamp(ts, UTC)
        if published < window_start:
            continue
        items.append({
            "id": item.get("id"),
            "title": truncate(title),
            "url": item.get("url") or f"https://news.ycombinator.com/item?id={item.get('id')}",
            "score": item.get("score", 0),
            "time": isoformat(published),
        })
        if len(items) >= 30:
            break
    prepared, section = render(items)
    if not prepared:
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        message = "no recent Hacker News stories matched filters"
        if cached_items_list:
            prepared, fallback_section = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, fallback_section, message)
    payload = {
        "generated_at": isoformat(NOW),
        "items": prepared,
    }
    write_yaml_if_changed(
        cache_path,
        payload,
        "https://hacker-news.firebaseio.com/",
    )
    return SectionResult(prepared, section)


@instrumented("CVE feed")
def collect_cves() -> SectionResult:
    cache_path = DATA_DIR / "cves.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared = sorted(items, key=lambda x: (-x.get("cvss", 0.0), x.get("published", ""), x.get("id") or ""))[:10]
        rows = [(item.get("id"), f"{item.get('cvss', 0):.1f}", item.get("summary", "")) for item in prepared]
        table = tabulate(rows, headers=["CVE", "CVSS", "Summary"], tablefmt="github") if rows else "No recent CVEs."
        if notice:
            table = f"{table}\n\n{notice}" if table else notice
        return prepared, table

    try:
        data = fetch_json(
            "https://cve.circl.lu/api/last",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        if cached_items_list:
            prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, table, message)
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": message,
        }
        write_yaml_if_changed(
            cache_path,
            payload,
            "https://cve.circl.lu/api/last",
        )
        return SectionResult([], stale_notice(message, cached_ts), message)
    items: List[Dict[str, Any]] = []
    for cve in data:
        cvss3 = cve.get("cvss3") or cve.get("cvss3_score")
        cvss = cve.get("cvss")
        try:
            severity = float(cvss3 or cvss or 0)
        except (TypeError, ValueError):
            severity = 0.0
        published_raw = cve.get("Published") or cve.get("published")
        try:
            published = datetime.fromisoformat(published_raw.replace("Z", "+00:00")) if published_raw else NOW
        except Exception:  # noqa: BLE001
            published = NOW
        items.append({
            "id": cve.get("id") or cve.get("cve") or cve.get("cve_id"),
            "summary": truncate(cve.get("summary") or cve.get("description") or ""),
            "cvss": severity,
            "published": isoformat(published),
        })
    prepared, table = render(items)
    if not prepared:
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        message = "no CVE entries returned"
        if cached_items_list:
            prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, table, message)
    payload = {
        "generated_at": isoformat(NOW),
        "items": prepared,
    }
    write_yaml_if_changed(
        cache_path,
        payload,
        "https://cve.circl.lu/api/last",
    )
    return SectionResult(prepared, table)


@instrumented("HF trending datasets")
def collect_hf_datasets() -> SectionResult:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    @retryable()
    def fetch(current_token: Optional[str]):
        api_client = hf_api_client(current_token)
        return list(api_client.list_datasets(sort="downloads", direction=-1, limit=15))

    cache_path = DATA_DIR / "hf_datasets.yaml"

    def render(items: List[Dict[str, Any]], notice: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
        prepared: List[Dict[str, Any]] = []
        for dataset in items:
            prepared.append(
                {
                    "dataset_id": dataset.get("dataset_id") or dataset.get("id"),
                    "downloads": dataset.get("downloads"),
                    "likes": dataset.get("likes"),
                    "tags": sorted(dataset.get("tags", []) or []),
                    "last_modified": dataset.get("last_modified"),
                }
            )
        prepared.sort(key=lambda x: (-(x.get("downloads") or 0), x.get("dataset_id") or ""))
        rows = [
            [item.get("dataset_id"), item.get("downloads", 0), item.get("likes", 0)] for item in prepared
        ]
        table = tabulate(rows, headers=["Dataset", "Downloads", "Likes"], tablefmt="github") if rows else "No trending datasets."
        if notice:
            table = f"{table}\n\n{notice}"
        return prepared, table

    try:
        datasets = fetch(token)
    except Exception as exc:  # noqa: BLE001
        message = format_exception(exc)
        if token:
            try:
                datasets = fetch(None)
            except Exception as fallback_exc:  # noqa: BLE001
                message = format_exception(fallback_exc)
                datasets = None
        else:
            datasets = None
        if datasets is None:
            cached_items_list, cached_ts = cached_items(cache_path, "items")
            if cached_items_list:
                prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
                return SectionResult(prepared, table, message)
            payload = {
                "generated_at": isoformat(NOW),
                "items": [],
                "error": message,
            }
            write_yaml_if_changed(cache_path, payload, "https://huggingface.co/api/datasets")
            return SectionResult([], stale_notice(message, cached_ts), message)

    if not isinstance(datasets, list):
        datasets = list(datasets)
    items: List[Dict[str, Any]] = []
    for dataset in datasets:
        ds_id = getattr(dataset, "id", None) or getattr(dataset, "datasetId", None) or getattr(dataset, "path", None)
        last_modified = getattr(dataset, "lastModified", None) or getattr(dataset, "last_modified", None)
        if isinstance(last_modified, datetime):
            lm = isoformat(last_modified)
        elif isinstance(last_modified, str):
            try:
                lm = isoformat(datetime.fromisoformat(last_modified.replace("Z", "+00:00")))
            except ValueError:
                lm = last_modified
        else:
            lm = None
        items.append({
            "dataset_id": ds_id,
            "downloads": getattr(dataset, "downloads", None),
            "likes": getattr(dataset, "likes", None),
            "tags": sorted(getattr(dataset, "tags", []) or []),
            "last_modified": lm,
        })
    items, table = render(items)
    if not items:
        cached_items_list, cached_ts = cached_items(cache_path, "items")
        message = "no trending Hugging Face datasets returned"
        if cached_items_list:
            prepared, table = render(cached_items_list, stale_notice(message, cached_ts))
            return SectionResult(prepared, table, message)
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(cache_path, payload, "https://huggingface.co/api/datasets")
    return SectionResult(items, table)



def render_dashboard(results: Dict[str, SectionResult]) -> bool:
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = {
        "GPU": DATA_DIR / "gpu_prices.json",
        "ARXIV": DATA_DIR / "arxiv.yaml",
        "HF": DATA_DIR / "hf_trending.yaml",
        "GHTREND": DATA_DIR / "github_trending.yaml",
        "PWC": DATA_DIR / "pwc_llm.yaml",
        "HN": DATA_DIR / "hn_ai.yaml",
        "CVE": DATA_DIR / "cves.yaml",
        "HFDATA": DATA_DIR / "hf_datasets.yaml",
    }

    def load_metadata(marker: str) -> Dict[str, Any]:
        path = dataset_paths.get(marker)
        if not path or not path.exists():
            return {}
        try:
            if path.suffix == ".json":
                return json.loads(path.read_text())
            return yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001
            return {}

    def render_metadata(marker: str) -> str:
        meta = load_metadata(marker)
        if not meta:
            return ""
        parts: List[str] = []
        fetched_at = meta.get("fetched_at")
        if fetched_at:
            parts.append(f"Fetched at {escape(str(fetched_at))}")
        source = meta.get("source_url")
        links: List[str] = []
        if isinstance(source, str):
            links = [source]
        elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
            links = [str(item) for item in source if item]
        if links:
            link_html = ", ".join(
                f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(url)}</a>'
                for url in links
            )
            parts.append(f"Source: {link_html}")
        digest = meta.get("hash")
        if digest:
            parts.append(f"Hash: <code>{escape(str(digest))}</code>")
        if not parts:
            return ""
        return "<p class=\"metadata\">" + " • ".join(parts) + "</p>"

    def make_link(text: Optional[str], url: Optional[str]) -> str:
        text = text or "Unknown"
        if not url:
            return escape(text)
        return f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(text)}</a>'

    def indent_block(text: str, spaces: int = 2) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line if line else line for line in text.splitlines())

    def render_table(headers: List[str], rows: List[List[str]]) -> str:
        if not rows:
            return '<p class="empty">No data available.</p>'
        header_cells = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        header_line = f"    <tr>{header_cells}</tr>"
        body_lines = [
            "    <tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        ]
        table_lines = [
            "<table>",
            "  <thead>",
            header_line,
            "  </thead>",
            "  <tbody>",
            *body_lines,
            "  </tbody>",
            "</table>",
        ]
        return "\n".join(table_lines)

    def gpu_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            price = item.get("usd_per_hour")
            price_str = f"${price:.4f}" if isinstance(price, (int, float)) else "—"
            rows.append(
                [
                    escape(item.get("gpu", "Unknown")),
                    price_str,
                    escape(item.get("source", "")),
                ]
            )
        return ["GPU", "Min USD/hr", "Source"], rows

    def arxiv_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            rows.append(
                [
                    make_link(item.get("title"), item.get("link")),
                    escape(item.get("summary", "")),
                    escape(item.get("published", "")),
                ]
            )
        return ["Title", "Summary", "Published"], rows

    def hf_model_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            model_id = item.get("model_id") or ""
            rows.append(
                [
                    make_link(model_id, f"https://huggingface.co/{model_id}" if model_id else None),
                    escape(str(item.get("downloads", 0))),
                    escape(str(item.get("likes", 0))),
                ]
            )
        return ["Model", "Downloads", "Likes"], rows

    def github_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            rows.append(
                [
                    make_link(item.get("full_name"), item.get("html_url")),
                    escape(str(item.get("stargazers_count", 0))),
                    escape(item.get("description", "")),
                ]
            )
        return ["Repository", "Stars", "Description"], rows

    def pwc_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            rows.append(
                [
                    make_link(item.get("title"), item.get("paper_url")),
                    escape(item.get("published", "")),
                    escape(", ".join(item.get("authors", []) or [])[:120]),
                ]
            )
        return ["Title", "Published", "Authors"], rows

    def hn_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            rows.append(
                [
                    make_link(item.get("title"), item.get("url")),
                    escape(str(item.get("score", 0))),
                    escape(item.get("time", "")),
                ]
            )
        return ["Story", "Score", "Published"], rows

    def cve_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            cve_id = item.get("id")
            link = f"https://www.cve.org/CVERecord?id={cve_id}" if cve_id else None
            rows.append(
                [
                    make_link(cve_id or "N/A", link),
                    escape(f"{item.get('cvss', 0):.1f}" if item.get("cvss") is not None else "0"),
                    escape(item.get("summary", "")),
                ]
            )
        return ["CVE", "CVSS", "Summary"], rows

    def hf_dataset_rows(result: SectionResult) -> Tuple[List[str], List[List[str]]]:
        rows: List[List[str]] = []
        for item in result.items:
            dataset_id = item.get("dataset_id") or ""
            rows.append(
                [
                    make_link(dataset_id, f"https://huggingface.co/datasets/{dataset_id}" if dataset_id else None),
                    escape(str(item.get("downloads", 0))),
                    escape(str(item.get("likes", 0))),
                ]
            )
        return ["Dataset", "Downloads", "Likes"], rows

    section_builders = [
        ("GPU", "GPU Pricing Snapshot", gpu_rows),
        ("ARXIV", "arXiv Digest", arxiv_rows),
        ("HF", "Hugging Face Trending Models", hf_model_rows),
        ("GHTREND", "GitHub Trending Repositories", github_rows),
        ("PWC", "Papers with Code", pwc_rows),
        ("HN", "Hacker News Highlights", hn_rows),
        ("CVE", "Latest CVEs", cve_rows),
        ("HFDATA", "Hugging Face Trending Datasets", hf_dataset_rows),
    ]

    sections_html: List[str] = []
    for marker, title, row_builder in section_builders:
        result = results.get(marker) or SectionResult([], "", None)
        headers, rows = row_builder(result)
        table_html = render_table(headers, rows)
        metadata_html = render_metadata(marker)
        warning_html = (
            f'<p class="warning">{escape(result.error)}</p>' if result.error else ""
        )
        table_block = indent_block(table_html)
        section_lines = [f"<section>", f"  <h2>{escape(title)}</h2>"]
        if metadata_html:
            section_lines.append("  " + metadata_html)
        if warning_html:
            section_lines.append("  " + warning_html)
        section_lines.append(table_block)
        section_lines.append("</section>")
        sections_html.append("\n".join(section_lines))

    data_links = "".join(
        f'<li><a href="{escape(os.path.relpath(path, docs_dir))}">{escape(path.name)}</a></li>'
        for path in dataset_paths.values()
        if path.exists()
    )

    style = """
body { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
header { padding: 2.5rem 1.5rem; text-align: center; background: #1e293b; }
header h1 { margin: 0 0 0.5rem; font-size: 2.5rem; }
header p { margin: 0; color: #94a3b8; font-size: 1rem; }
main { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
section { background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 12px 24px rgba(15, 23, 42, 0.4); }
section h2 { margin-top: 0; color: #f8fafc; font-size: 1.5rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { text-align: left; padding: 0.65rem 0.75rem; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }
th { background: rgba(148, 163, 184, 0.1); color: #e2e8f0; font-weight: 600; }
tr:hover td { background: rgba(59, 130, 246, 0.08); }
a { color: #38bdf8; text-decoration: none; }
a:hover { text-decoration: underline; }
.warning { margin: 0.75rem 0; color: #fbbf24; }
.empty { margin: 1rem 0 0; color: #94a3b8; font-style: italic; }
.metadata { margin: 0.35rem 0 0; color: #94a3b8; font-size: 0.9rem; }
footer { text-align: center; padding: 2rem 1.5rem 3rem; color: #64748b; font-size: 0.9rem; }
footer ul { list-style: none; padding: 0; margin: 1rem 0 0; display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: center; }
footer li { background: rgba(148, 163, 184, 0.1); border-radius: 6px; padding: 0.4rem 0.75rem; }
"""

    last_updated = isoformat(NOW)
    sections_rendered = "".join(sections_html)
    footer_html = (
        f"<footer><p>Datasets updated as part of the DeepTech Daily automation run.</p>"
        f"<p>Last refreshed at {escape(last_updated)}.</p>"
        f"<ul>{data_links}</ul></footer>"
    )
    html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>DeepTech Daily Dashboard</title>
  <style>{style}</style>
</head>
<body>
  <header>
    <h1>DeepTech Daily Dashboard</h1>
    <p>A curated snapshot of GPUs, research, software, and security intel powered by the DeepTech Daily datasets.</p>
  </header>
  <main>
    {sections_rendered}
  </main>
  {footer_html}
</body>
</html>
"""

    return write_text_if_changed(docs_dir / "index.html", html_doc)


SECTION_BUILDERS = {
    "GPU": collect_gpu_prices,
    "ARXIV": collect_arxiv,
    "HF": collect_hf_models,
    "GHTREND": collect_github_trending,
    "PWC": collect_papers_with_code,
    "HN": collect_hn_ai,
    "CVE": collect_cves,
    "HFDATA": collect_hf_datasets,
}

from deeptech_daily.tools.anomaly import (
    assess_anomalies,
    collect_today_metrics,
    render_radar_md,
    update_timeseries,
    utc_date as anomaly_utc_date,
)

ANOMALY_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ANOMALY_TS_PATH = os.path.join(ANOMALY_BASE_DIR, "data", "metrics_timeseries.json")
ANOMALY_README_PATH = os.path.join(ANOMALY_BASE_DIR, "README.md")


def _inject_radar_section() -> None:
    today = anomaly_utc_date()
    metrics = collect_today_metrics(ANOMALY_BASE_DIR)
    ts = update_timeseries(ANOMALY_TS_PATH, today, metrics)
    alerts = assess_anomalies(ts)
    fragment = render_radar_md(alerts, today)

    try:
        with open(ANOMALY_README_PATH, "r", encoding="utf-8") as f:
            readme = f.read()
    except FileNotFoundError:
        readme = ""

    start_tag = "<!-- RADAR:START -->"
    end_tag = "<!-- RADAR:END -->"
    if start_tag in readme and end_tag in readme:
        pre, rest = readme.split(start_tag, 1)
        _, post = rest.split(end_tag, 1)
        new_readme = pre + fragment + post
    else:
        if "# " in readme:
            parts = readme.split("\n", 2)
            if len(parts) >= 2:
                new_readme = (
                    parts[0]
                    + "\n"
                    + parts[1]
                    + "\n\n"
                    + fragment
                    + (parts[2] if len(parts) == 3 else "")
                )
            else:
                new_readme = readme + "\n\n" + fragment
        else:
            new_readme = fragment + "\n" + readme

    if new_readme != readme:
        with open(ANOMALY_README_PATH, "w", encoding="utf-8") as f:
            f.write(new_readme)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    readme_updates: Dict[str, str] = {}
    section_results: Dict[str, SectionResult] = {}
    for marker, builder in SECTION_BUILDERS.items():
        result = builder()
        section_results[marker] = result
        readme_updates[marker] = result.readme
    readme_changed = update_readme_sections(readme_updates)
    dashboard_changed = render_dashboard(section_results)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        changed = bool(status.stdout.strip())
    except Exception:
        changed = readme_changed or dashboard_changed
    if not changed and (readme_changed or dashboard_changed):
        changed = True
    print(f"changed: {str(changed)}")
    auto_commit_enabled = os.getenv("DEEPTECH_AUTO_COMMIT", "1").lower() not in {"0", "false", "no"}
    if auto_commit_enabled:
        command = os.getenv(
            "DEEPTECH_AUTO_COMMIT_CMD",
            'git diff --quiet || git commit -am "daily refresh" && git push',
        )
        try:
            subprocess.run(command, shell=True, check=True, cwd=str(REPO_ROOT))
            LOGGER.info("Auto-commit command executed: %s", command)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Auto-commit command failed with exit code %s", exc.returncode)
    update_dpi_section()
    _inject_radar_section()
    return 0


def _run_import_test() -> None:
    """Run a lightweight smoke test to validate module imports."""

    try:
        for marker, builder in SECTION_BUILDERS.items():
            if not callable(builder):
                raise TypeError(f"Section builder for {marker} is not callable")
    except Exception:  # noqa: BLE001
        LOGGER.exception("Daily update script import test failed")
        raise
    LOGGER.info("✅ Daily update script import test passed")


def _request_with_retries(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    """Perform a GET request with up to two retries and a 5s timeout."""

    attempts = 3
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code >= 500:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                time.sleep(0.5 * attempt)
                continue
            return response
        except requests.RequestException as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5 * attempt)
    if last_error:
        LOGGER.warning(
            "Request to %s failed after %s attempts: %s",
            url,
            attempts,
            format_exception(last_error),
        )
    return None


def _normalise_gpu_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    search_space = raw.upper()
    for canonical, patterns in GPU_NAME_ALIASES.items():
        for pattern in patterns:
            if pattern.upper() in search_space:
                return canonical
    match = re.search(r"(RTX\s*)?(\d{3,4})", search_space)
    if match:
        digits = match.group(2)
        return f"RTX {digits}" if len(digits) == 4 else digits
    return raw.strip()


def _current_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_azure_gpu_pricing() -> List[Dict[str, Any]]:
    url = (
        "https://prices.azure.com/api/retail/prices?$filter="
        "serviceName eq 'Virtual Machines' and (contains(skuName,'NC') or contains(skuName,'ND') or contains(skuName,'NV'))"
    )
    records: List[Dict[str, Any]] = []
    next_url: Optional[str] = url
    while next_url:
        response = _request_with_retries(next_url)
        if response is None:
            print("parse_azure_gpu_pricing fetched 0 entries")
            return []
        try:
            payload = response.json()
        except ValueError:
            print("parse_azure_gpu_pricing fetched 0 entries")
            return []
        for item in payload.get("Items", []):
            if item.get("currencyCode") != "USD":
                continue
            unit = (item.get("unitOfMeasure") or "").lower()
            if "hour" not in unit:
                continue
            price = item.get("unitPrice") or item.get("retailPrice")
            try:
                usd_per_hour = float(price)
            except (TypeError, ValueError):
                continue
            sku_name = item.get("skuName") or ""
            gpu_name = _normalise_gpu_name(sku_name)
            record = {
                "gpu": gpu_name or sku_name,
                "usd_per_hour": usd_per_hour,
                "source": "azure",
                "region": item.get("armRegionName"),
                "sku": item.get("skuId") or sku_name,
                "notes": item.get("productName"),
                "fetched_at": _current_timestamp(),
            }
            records.append(record)
        next_url = payload.get("NextPageLink")
        if not next_url:
            break
    print(f"parse_azure_gpu_pricing fetched {len(records)} entries")
    return records


def parse_aws_gpu_pricing() -> List[Dict[str, Any]]:
    url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json"
    response = _request_with_retries(url)
    if response is None:
        print("parse_aws_gpu_pricing fetched 0 entries")
        return []
    try:
        data = response.json()
    except ValueError:
        print("parse_aws_gpu_pricing fetched 0 entries")
        return []
    products = data.get("products", {})
    terms = data.get("terms", {}).get("OnDemand", {})
    records: List[Dict[str, Any]] = []
    allowed_tokens = ("p3", "p4d", "p5", "g5", "g6")
    for sku, product in products.items():
        attributes = product.get("attributes", {})
        instance_type = (attributes.get("instanceType") or "").lower()
        if not any(token in instance_type for token in allowed_tokens):
            continue
        ondemand = None
        for term in terms.values():
            if term.get("sku") == sku:
                ondemand = term
                break
        if not ondemand:
            continue
        price_dimensions = ondemand.get("priceDimensions", {})
        dimension = next(iter(price_dimensions.values()), None)
        if not dimension:
            continue
        price_per_unit = dimension.get("pricePerUnit", {}).get("USD")
        try:
            usd_per_hour = float(price_per_unit)
        except (TypeError, ValueError):
            continue
        gpu_name = _normalise_gpu_name(attributes.get("gpuModel") or attributes.get("instanceType"))
        record = {
            "gpu": gpu_name or (attributes.get("gpuModel") or attributes.get("instanceType")),
            "usd_per_hour": usd_per_hour,
            "source": "aws",
            "region": attributes.get("location"),
            "sku": sku,
            "notes": attributes.get("instanceType"),
            "fetched_at": _current_timestamp(),
        }
        records.append(record)
    print(f"parse_aws_gpu_pricing fetched {len(records)} entries")
    return records


def parse_gcp_gpu_pricing() -> List[Dict[str, Any]]:
    base_url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
    page_token: Optional[str] = None
    records: List[Dict[str, Any]] = []
    while True:
        params = {"pageSize": 500}
        if page_token:
            params["pageToken"] = page_token
        response = _request_with_retries(base_url, params=params)
        if response is None:
            print(f"parse_gcp_gpu_pricing fetched {len(records)} entries")
            return records
        try:
            payload = response.json()
        except ValueError:
            print(f"parse_gcp_gpu_pricing fetched {len(records)} entries")
            return records
        for sku in payload.get("skus", []):
            description = sku.get("description", "")
            if "GPU" not in description.upper():
                continue
            pricing_infos = sku.get("pricingInfo", [])
            usd_per_hour: Optional[float] = None
            for info in pricing_infos:
                expr = info.get("pricingExpression", {})
                unit = (expr.get("unit") or "").upper()
                rates = expr.get("tieredRates", [])
                if not rates:
                    continue
                rate = rates[0].get("unitPrice", {})
                units = float(rate.get("units", 0))
                nanos = float(rate.get("nanos", 0)) / 1_000_000_000
                price = units + nanos
                if unit == "HOUR":
                    usd_per_hour = price
                    break
                if unit == "SECOND":
                    usd_per_hour = price * 3600
                    break
                if unit == "MONTH":
                    usd_per_hour = price / (24 * 30)
                    break
            if usd_per_hour is None:
                continue
            gpu_name = _normalise_gpu_name(description)
            record = {
                "gpu": gpu_name or description,
                "usd_per_hour": usd_per_hour,
                "source": "gcp",
                "region": ",".join(sku.get("serviceRegions", [])) or None,
                "sku": sku.get("name"),
                "notes": description,
                "fetched_at": _current_timestamp(),
            }
            records.append(record)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    print(f"parse_gcp_gpu_pricing fetched {len(records)} entries")
    return records


def parse_vast_ai_offers() -> List[Dict[str, Any]]:
    url = "https://console.vast.ai/api/v0/bundles/search/"
    params = {"q": "verified=true rentable=true", "limit": 100}
    response = _request_with_retries(url, params=params)
    if response is None or response.status_code != 200:
        print("parse_vast_ai_offers fetched 0 entries")
        return []
    try:
        data = response.json()
    except ValueError:
        print("parse_vast_ai_offers fetched 0 entries")
        return []
    offers = data.get("offers") or []
    records: List[Dict[str, Any]] = []
    for offer in offers:
        gpu_name = _normalise_gpu_name(offer.get("gpu_name") or offer.get("gpu_name_brand"))
        price = offer.get("dph_total") or offer.get("price_gpu")
        try:
            usd_per_hour = float(price)
        except (TypeError, ValueError):
            continue
        record = {
            "gpu": gpu_name
            or (offer.get("gpu_name") or offer.get("gpu_name_brand") or "Unknown"),
            "usd_per_hour": usd_per_hour,
            "source": "vast.ai",
            "region": offer.get("geolocation"),
            "sku": str(offer.get("id")) if offer.get("id") is not None else None,
            "notes": offer.get("verification") or offer.get("hostname"),
            "fetched_at": _current_timestamp(),
        }
        records.append(record)
    print(f"parse_vast_ai_offers fetched {len(records)} entries")
    return records


def parse_hyperstack_pricing() -> List[Dict[str, Any]]:
    url = "https://www.hyperstack.cloud/api/pricing"
    response = _request_with_retries(url, headers={"accept": "application/json, text/plain, */*"})
    if response is None or response.status_code != 200:
        print("parse_hyperstack_pricing fetched 0 entries")
        return []
    try:
        data = response.json()
    except ValueError:
        print("parse_hyperstack_pricing fetched 0 entries")
        return []
    pricing_table = data.get("pricing") if isinstance(data, dict) else data
    records: List[Dict[str, Any]] = []
    if isinstance(pricing_table, dict):
        iterable = pricing_table.items()
    elif isinstance(pricing_table, list):
        iterable = enumerate(pricing_table)
    else:
        iterable = []
    for key, value in iterable:
        if not isinstance(value, dict):
            continue
        price = value.get("price") or value.get("usdPerHour") or value.get("usd_per_hour")
        try:
            usd_per_hour = float(price)
        except (TypeError, ValueError):
            continue
        gpu_name = _normalise_gpu_name(value.get("name") or str(key))
        record = {
            "gpu": gpu_name or (value.get("name") or str(key)),
            "usd_per_hour": usd_per_hour,
            "source": "hyperstack",
            "region": value.get("region"),
            "sku": value.get("sku")
            or value.get("id")
            or (str(key) if not isinstance(key, int) else None),
            "notes": value.get("notes") or value.get("description"),
            "fetched_at": _current_timestamp(),
        }
        records.append(record)
    print(f"parse_hyperstack_pricing fetched {len(records)} entries")
    return records


if __name__ == "__main__":
    try:
        _run_import_test()
    except Exception:  # noqa: BLE001
        sys.exit(1)
    sample_functions = [
        parse_azure_gpu_pricing,
        parse_aws_gpu_pricing,
        parse_gcp_gpu_pricing,
        parse_vast_ai_offers,
        parse_hyperstack_pricing,
    ]
    for func in sample_functions:
        try:
            results = func()
            preview = results[:2]
            print(
                f"Sample from {func.__name__}: "
                f"{json.dumps(preview, indent=2) if preview else '[]'}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error running {func.__name__}: {format_exception(exc)}")
    sys.exit(main())
