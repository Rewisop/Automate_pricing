# Tooling overview

The automation layer consists of a single Python entry point,
[`daily_update.py`](daily_update.py). The script is self-contained so it can be
invoked by GitHub Actions, cron jobs, or any other scheduler with Python 3.10+
available.

Key concepts:

- **Section collectors** (`collect_*` functions) encapsulate logic for each data
  source. They return a `SectionResult` that includes both structured data and a
  Markdown fragment for the public README.
- **Writers** (`write_json_if_changed`, `write_yaml_if_changed`) emit files only
  when their contents change, keeping Git history noise-free.
- **README markers** (`<!--XYZ:START-->` / `<!--XYZ:END-->`) define the regions
  that will be replaced when the updater runs.

Run the script locally to verify changes before committing:

```bash
python daily_update.py
```

The command prints whether any files changed and will surface API failures in
the console output as well as in the rendered README sections.
