"""Loading published and uploaded benchmark runs."""
import json

import pandas as pd

from .config import LEGACY_MODEL_NAMES, RAW_PREFIXES, ROOT
from .runners import NUMERIC_COLUMNS, RESULT_COLUMNS

BENCHMARKS_DIR = ROOT / "benchmarks"
NOT_RECORDED = "Not recorded"


def normalize_results(results):
    """Bring older or uploaded result files to the current columns.

    Files are never rewritten. Older runs called Jev Router plain "Jev";
    that label is mapped to "Jev Router" on load, never to Jev direct.
    """
    results = results.copy()
    if "served_by" not in results and "checkpoint" in results:
        results["served_by"] = results["checkpoint"]
    for column in RESULT_COLUMNS:
        if column not in results:
            results[column] = None if column in NUMERIC_COLUMNS else ""
    for column in NUMERIC_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")
    text_columns = [c for c in RESULT_COLUMNS if c not in NUMERIC_COLUMNS]
    results[text_columns] = results[text_columns].fillna("").astype(str)
    results["model"] = results["model"].replace(LEGACY_MODEL_NAMES)
    # Early runs used "error" for every failure.
    results["status"] = results["status"].replace({"error": "request_failed"})
    return results[RESULT_COLUMNS]


def normalize_metadata(metadata):
    models = metadata.get("models", {})
    metadata = dict(metadata, models={
        LEGACY_MODEL_NAMES.get(name, name): info for name, info in models.items()
    })
    return metadata


def list_published_runs():
    """Return [(run_dir, metadata)] newest first."""
    runs = []
    for run_dir in BENCHMARKS_DIR.glob("*/"):
        meta_path = run_dir / "metadata.json"
        if (run_dir / "results.csv").exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text("utf-8"))
            runs.append((run_dir, normalize_metadata(metadata)))
    return sorted(runs, key=lambda item: item[1].get("started_utc") or "",
                  reverse=True)


def load_run(run_dir):
    results = pd.read_csv(run_dir / "results.csv", dtype={"id": str},
                          keep_default_na=False)
    metadata = json.loads((run_dir / "metadata.json").read_text("utf-8"))
    return normalize_results(results), normalize_metadata(metadata)


def load_raw(run_dir, model, case_id):
    for prefix in RAW_PREFIXES.get(model, [model.lower()]):
        path = run_dir / "raw" / f"{prefix}_{case_id}.json"
        if path.exists():
            return json.loads(path.read_text("utf-8"))
    return None
