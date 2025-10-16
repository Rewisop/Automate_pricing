# GPU provider configuration

Providers listed in `gpu_sources.yaml` drive the "GPU Pricing Snapshot" section
of the daily report. Each entry must specify:

- `name` – human-readable source name displayed in the output table.
- `enabled` – set to `true` to include the provider during the next run.
- `parser` – function defined in [`tools/daily_update.py`](../tools/daily_update.py)
  responsible for fetching and normalising pricing data.
- `base_url` (and any additional metadata the parser expects).

To add a new provider:

1. Implement a parser function that returns a list of dictionaries with
   `gpu`, `usd_per_hour`, and optional `source` keys.
2. Register the parser name in `gpu_sources.yaml` and flip `enabled: true` when
   you are ready to ingest it.

Parsers can share helper utilities defined in `daily_update.py`. Keep API keys
out of this file—use environment variables or secret stores provided by your
automation platform.
