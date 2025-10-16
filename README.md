# Automate_pricing

Automate_pricing contains a small automation stack for publishing a daily
snapshot of the GPU market together with a handful of other AI/ML intelligence
signals. The core workflow lives in the [`deeptech-daily`](deeptech-daily)
subdirectory and is designed to run in CI (for example via GitHub Actions) or
on a local cron job. When executed, the bot fetches pricing and trend data from
several public APIs, writes the canonical datasets to `deeptech-daily/data/` and
refreshes Markdown sections inside `deeptech-daily/README.md`.

The repository is intentionally lightweight—the automation is a single Python
script and a few configuration files—so having clear documentation makes it
easy to adapt the project to a different publication cadence or data sources.

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
   `deeptech-daily/data/` and the Markdown report is rewritten in place.

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
