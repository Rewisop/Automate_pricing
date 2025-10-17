# Automate_pricing

Automate_pricing contains a small automation stack for publishing a daily
snapshot of the GPU market together with a handful of other AI/ML intelligence
signals. Think of it as an autonomous research assistant: every execution pulls
fresh pricing data, trending repositories, noteworthy papers, security updates,
and community chatter, then compiles the results into Markdown and structured
data that can be consumed by humans or downstream tooling.

The core workflow lives in the [`deeptech-daily`](deeptech-daily) subdirectory
and is designed to run unattended in CI (for example via GitHub Actions) or on
a local cron job. When executed, the bot orchestrates a set of provider
parsers, fetches data from public APIs, writes canonical datasets to
`deeptech-daily/data/`, refreshes Markdown sections inside
`deeptech-daily/README.md`, and republishes a static dashboard inside `docs/`.

At its heart the project is intentionally lightweight—the automation is a
single Python script plus configuration—so adapting it to a different
publication cadence or new data sources only requires editing a few YAML files
or adding a parser function. Runs emit structured logs (configurable via
`DEEPTECH_LOG_LEVEL`) so scheduled executions are easy to observe and debug.

## Key capabilities

- **Daily GPU price intelligence:** Consolidates on-demand GPU rental prices
  from supported providers and tracks historical minimum rates for comparison.
- **AI/ML research radar:** Surfaces the latest arXiv submissions, trending
  Hugging Face models and datasets, and notable Papers with Code entries.
- **Ecosystem pulse checks:** Monitors fast-rising GitHub repositories,
  high-signal Hacker News discussions, and recent AI-related CVEs.
- **Shareable outputs:** Generates a public-facing Markdown report, machine
  readable data exports, and a static dashboard suitable for GitHub Pages or
  other hosting.

## Repository layout

```text
Automate_pricing/
├── README.md                 # You are here
└── deeptech-daily/
    ├── README.md             # Public-facing daily report (auto-updated)
    ├── data/                 # Structured data exported by the updater
    ├── providers/            # Configuration for GPU price providers
    ├── tools/
    │   └── daily_update.py   # Main automation entry-point
    └── requirements.txt      # Python dependencies for the toolchain
```

## Getting started

1. **Create a virtual environment** (Python 3.10+ recommended) and install the
   dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r deeptech-daily/requirements.txt
   ```

2. **Provide API credentials (optional but recommended).** Some data sources
   allow higher rate limits or access to private endpoints when authenticated.
   The updater looks for the following environment variables:

   | Variable | Purpose |
   | --- | --- |
   | `HUGGINGFACE_HUB_TOKEN` | Unlocks trending model/dataset feeds from the Hugging Face Hub. |
   | `GITHUB_TOKEN` | Raises rate limits for the GitHub Trending search. |

3. **Run the updater locally:**

   ```bash
   python deeptech-daily/tools/daily_update.py
   ```

   The command prints whether any files changed. Updated datasets live in
   `deeptech-daily/data/`, the Markdown report is rewritten in place, and a
   static dashboard is published to `docs/index.html` for GitHub Pages.

   By default the tool also executes `git diff --quiet || git commit -am
   "daily refresh" && git push` from the repository root so that fresh data is
   automatically committed. Disable this behaviour by setting
   `DEEPTECH_AUTO_COMMIT=0` or override the command with
   `DEEPTECH_AUTO_COMMIT_CMD`.

4. **Automate the run** by wiring the script into a scheduled workflow. The
   project includes a reusable GitHub Actions workflow in the original upstream
   project, but any scheduler that can execute the script is compatible.

## Customising GPU providers

GPU pricing is driven by `deeptech-daily/providers/gpu_sources.yaml`. Enable or
disable providers with the `enabled` flag and point to a parser function defined
in `daily_update.py`. You can add new providers by:

1. Defining a new parser function that returns a list of dictionaries in the
   format `{"gpu": str, "usd_per_hour": float, "source": str}`.
2. Registering the parser in `gpu_sources.yaml` and ensuring it is included in
   the `SECTION_BUILDERS` mapping if it powers a README section.

## Data exports

Every run writes machine-readable data that mirrors the Markdown sections in the
daily report. Files are timestamped and can be used for downstream analysis or
visualisation.

| File | Format | Description |
| --- | --- | --- |
| `gpu_prices.json` | JSON | Minimum observed price per GPU and provider metadata. |
| `arxiv.yaml` | YAML | Latest arXiv submissions for AI/ML categories. |
| `hf_trending.yaml` | YAML | Top Hugging Face models by downloads. |
| `github_trending.yaml` | YAML | AI/LLM repositories created within the last week, sorted by stars. |
| `pwc_llm.yaml` | YAML | Recent Papers with Code entries related to LLMs. |
| `hn_ai.yaml` | YAML | High-scoring AI-related posts from Hacker News. |
| `cves.yaml` | YAML | Notable CVEs touching AI/ML systems. |
| `hf_datasets.yaml` | YAML | Trending Hugging Face datasets. |

Each export now includes provenance metadata—`generated_at`, `fetched_at`,
`source_url`, and a deterministic content `hash`—so downstream consumers can
track refresh cadence and verify integrity.

## DeepTech Daily Dashboard

A static “DeepTech Daily Dashboard” is generated at `docs/index.html` using the
datasets above. Push the `docs/` directory to GitHub and enable GitHub Pages to
publish a live snapshot of the intel feed.

## Contributing

Contributions are welcome! Please open an issue or pull request describing the
change you would like to make. Enhancements that are especially helpful
include:

- Additional data sources or improvements to the resiliency of existing ones.
- Documentation improvements or examples for integrating the updater into
  different automation platforms.
- Tooling that surfaces the generated data (dashboards, visualisations, etc.).

Before submitting a change, run the updater locally to ensure the report and
datasets regenerate without errors.
