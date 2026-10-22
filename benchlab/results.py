"""Loading published and uploaded benchmark runs."""
import json

import pandas as pd

from .config import ROOT
from .runners import RESULT_COLUMNS

BENCHMARKS_DIR = ROOT / "benchmarks"
NOT_RECORDED = "Not recorded"


def normalize_results(results):
    """Bring older or uploaded result files to the current columns."""
    results = results.copy()
    if "served_by" not in results and "checkpoint" in results:
        results["served_by"] = results["checkpoint"]
    for column in RESULT_COLUMNS:
        if column not in results:
            results[column] = None if column in (
                "latency_ms", "cost_usd", "reported_confidence") else ""
    for column in ("latency_ms", "cost_usd", "reported_confidence"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    text_columns = [c for c in RESULT_COLUMNS if c not in (
        "latency_ms", "cost_usd", "reported_confidence")]
    results[text_columns] = results[text_columns].fillna("").astype(str)
    # Early runs used "error" for every failure.
    results["status"] = results["status"].replace({"error": "request_failed"})
    return results[RESULT_COLUMNS]


def list_published_runs():
    """Return [(run_dir, metadata)] newest first."""
    runs = []
    for run_dir in BENCHMARKS_DIR.glob("*/"):
        meta_path = run_dir / "metadata.json"
        if (run_dir / "results.csv").exists() and meta_path.exists():
            runs.append((run_dir, json.loads(meta_path.read_text("utf-8"))))
    return sorted(runs, key=lambda item: item[1].get("started_utc") or "",
                  reverse=True)


def load_run(run_dir):
    results = pd.read_csv(run_dir / "results.csv", dtype={"id": str},
                          keep_default_na=False)
    metadata = json.loads((run_dir / "metadata.json").read_text("utf-8"))
    return normalize_results(results), metadata


def load_raw(run_dir, model, case_id):
    path = run_dir / "raw" / f"{model.lower()}_{case_id}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None
