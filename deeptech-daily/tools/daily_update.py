from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import requests
import yaml
from huggingface_hub import HfApi
from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
README_PATH = BASE_DIR / "README.md"
GPU_PROVIDERS_PATH = BASE_DIR / "providers" / "gpu_sources.yaml"
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
HEX_PTR = re.compile(r"0x[0-9a-fA-F]+")


@dataclass
class SectionResult:
    items: List[Dict[str, Any]]
    readme: str
    error: Optional[str] = None


def truncate(text: str, limit: int = MAX_TEXT_LENGTH) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text_if_changed(path: Path, content: str) -> bool:
    encoded = content.rstrip() + "\n"
    if path.exists():
        existing = path.read_text()
        if existing == encoded:
            return False
    path.write_text(encoded)
    return True


def write_json_if_changed(path: Path, payload: Dict[str, Any]) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = preserve_generated_at(path, payload, json.loads)
    serialised = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return write_text_if_changed(path, serialised)


def write_yaml_if_changed(path: Path, payload: Dict[str, Any]) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = preserve_generated_at(path, payload, yaml.safe_load)
    serialised = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return write_text_if_changed(path, serialised)


def preserve_generated_at(path: Path, payload: Dict[str, Any], loader) -> Dict[str, Any]:
    if not path.exists():
        return payload
    try:
        existing = loader(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return payload

    def strip_generated_at(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: strip_generated_at(v) for k, v in data.items() if k != "generated_at"}
        if isinstance(data, list):
            return [strip_generated_at(item) for item in data]
        return data

    if strip_generated_at(existing) == strip_generated_at(payload):
        existing_ts = existing.get("generated_at")
        if existing_ts:
            payload = dict(payload)
            payload["generated_at"] = existing_ts
    return payload


def format_exception(exc: Exception) -> str:
    text = str(exc)
    text = HEX_PTR.sub("0xXXXX", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    response = SESSION.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
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


def collect_gpu_prices() -> SectionResult:
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    for source in load_gpu_sources():
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
    table = "No GPU pricing data available."
    if sorted_items:
        rows = [(item["gpu"], f"${item['usd_per_hour']:.4f}") for item in sorted_items]
        table = tabulate(rows, headers=["GPU", "Min USD/hr"], tablefmt="github")
    error_text = "; ".join(errors) if errors else None
    payload = {
        "generated_at": isoformat(NOW),
        "sources": [src for src in load_gpu_sources()],
        "gpus": sorted_items,
        "errors": errors,
    }
    write_json_if_changed(DATA_DIR / "gpu_prices.json", payload)
    if error_text and not sorted_items:
        table = f"Source unavailable ({truncate(error_text, 120)})."
    elif error_text:
        table += f"\n\n_Warnings: {truncate(error_text, 120)}_"
    return SectionResult(sorted_items, table, error_text)


def parse_feed_entry(entry: Any) -> Tuple[str, datetime]:
    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if published_parsed:
        dt = datetime(*published_parsed[:6], tzinfo=UTC)
    else:
        dt = NOW
    return entry.get("id") or entry.get("link"), dt


def collect_arxiv() -> SectionResult:
    feeds = [
        "https://export.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/rss/cs.CL",
        "https://export.arxiv.org/rss/cs.LG",
    ]
    items: Dict[str, Dict[str, Any]] = {}
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:  # noqa: BLE001
            continue
        for entry in parsed.entries:
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
    payload = {
        "generated_at": isoformat(NOW),
        "items": sorted_items,
    }
    write_yaml_if_changed(DATA_DIR / "arxiv.yaml", payload)
    return SectionResult(sorted_items, section)


def hf_api_client() -> HfApi:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    return HfApi(token=token)


def collect_hf_models() -> SectionResult:
    api = hf_api_client()
    try:
        models = api.list_models(sort="downloads", direction=-1, limit=15)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "hf_trending.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
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
    items.sort(key=lambda x: (-(x["downloads"] or 0), x["model_id"]))
    rows = [(item["model_id"], item["downloads"] or 0, item["likes"] or 0) for item in items]
    table = tabulate(rows, headers=["Model", "Downloads", "Likes"], tablefmt="github") if rows else "No trending models."
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(DATA_DIR / "hf_trending.yaml", payload)
    return SectionResult(items, table)


def github_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    headers = HEADERS.copy()
    headers["Accept"] = "application/vnd.github+json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def collect_github_trending() -> SectionResult:
    since = (NOW - timedelta(days=7)).date().isoformat()
    query = "(LLM OR \"large language model\" OR genai) created:>" + since
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 20,
    }
    try:
        response = SESSION.get("https://api.github.com/search/repositories", params=params, headers=github_headers(), timeout=REQUEST_TIMEOUT)
        if response.status_code in {403, 429}:
            raise RuntimeError(f"GitHub API limit {response.status_code}")
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "github_trending.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
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
    items.sort(key=lambda x: (-x["stargazers_count"], x["full_name"] or ""))
    top_items = items[:20]
    rows = [(item["full_name"], item["stargazers_count"], item["description"]) for item in top_items]
    table = tabulate(rows, headers=["Repository", "Stars", "Description"], tablefmt="github") if rows else "No trending repositories."
    payload = {
        "generated_at": isoformat(NOW),
        "items": top_items,
    }
    write_yaml_if_changed(DATA_DIR / "github_trending.yaml", payload)
    return SectionResult(top_items, table)


def collect_papers_with_code() -> SectionResult:
    url = "https://paperswithcode.com/api/v1/papers/"
    params = {
        "q": "LLM OR \"large language model\"",
        "ordering": "-published",
        "items_per_page": 20,
    }
    try:
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "pwc_llm.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
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
    items.sort(key=lambda x: x["published"], reverse=True)
    bullets = [f"- [{item['title']}]({item['paper_url']}) — {truncate((item.get('authors') or '')[:120])}" for item in items]
    section = "\n".join(bullets) if bullets else "No recent papers found."
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(DATA_DIR / "pwc_llm.yaml", payload)
    return SectionResult(items, section)


def collect_hn_ai() -> SectionResult:
    try:
        top_ids = SESSION.get("https://hacker-news.firebaseio.com/v0/topstories.json", headers=HEADERS, timeout=REQUEST_TIMEOUT).json()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "hn_ai.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
    if not isinstance(top_ids, list):
        top_ids = []
    window_start = NOW - timedelta(hours=24)
    items: List[Dict[str, Any]] = []
    for story_id in top_ids[:200]:
        try:
            item = SESSION.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", headers=HEADERS, timeout=REQUEST_TIMEOUT).json()
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
    items.sort(key=lambda x: (-x["score"], x["time"], x["id"]))
    bullets = [f"- [{item['title']}]({item['url']}) — {item['score']} points" for item in items[:15]]
    section = "\n".join(bullets) if bullets else "No AI-related stories in the last 24h."
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(DATA_DIR / "hn_ai.yaml", payload)
    return SectionResult(items, section)


def collect_cves() -> SectionResult:
    try:
        response = SESSION.get("https://cve.circl.lu/api/last", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "cves.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
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
    items.sort(key=lambda x: (-x["cvss"], x["published"], x["id"] or ""))
    top_items = items[:10]
    rows = [(item["id"], f"{item['cvss']:.1f}", item["summary"]) for item in top_items]
    table = tabulate(rows, headers=["CVE", "CVSS", "Summary"], tablefmt="github") if rows else "No recent CVEs."
    payload = {
        "generated_at": isoformat(NOW),
        "items": top_items,
    }
    write_yaml_if_changed(DATA_DIR / "cves.yaml", payload)
    return SectionResult(top_items, table)


def collect_hf_datasets() -> SectionResult:
    api = hf_api_client()
    try:
        datasets = api.list_datasets(sort="downloads", direction=-1, limit=15)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "generated_at": isoformat(NOW),
            "items": [],
            "error": format_exception(exc),
        }
        write_yaml_if_changed(DATA_DIR / "hf_datasets.yaml", payload)
        message = format_exception(exc)
        return SectionResult([], f"Source unavailable ({truncate(message, 120)}).", message)
    items: List[Dict[str, Any]] = []
    for dataset in datasets:
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
            "dataset_id": dataset.id,
            "downloads": getattr(dataset, "downloads", None),
            "likes": getattr(dataset, "likes", None),
            "tags": sorted(getattr(dataset, "tags", []) or []),
            "last_modified": lm,
        })
    items.sort(key=lambda x: (-(x["downloads"] or 0), x["dataset_id"]))
    rows = [(item["dataset_id"], item["downloads"] or 0, item["likes"] or 0) for item in items]
    table = tabulate(rows, headers=["Dataset", "Downloads", "Likes"], tablefmt="github") if rows else "No trending datasets."
    payload = {
        "generated_at": isoformat(NOW),
        "items": items,
    }
    write_yaml_if_changed(DATA_DIR / "hf_datasets.yaml", payload)
    return SectionResult(items, table)


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


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    readme_updates: Dict[str, str] = {}
    for marker, builder in SECTION_BUILDERS.items():
        result = builder()
        readme_updates[marker] = result.readme
    readme_changed = update_readme_sections(readme_updates)
    try:
        import subprocess

        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(BASE_DIR))
        changed = bool(result.stdout.strip())
    except Exception:
        changed = readme_changed
    if not changed and readme_changed:
        changed = True
    print(f"changed: {str(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
