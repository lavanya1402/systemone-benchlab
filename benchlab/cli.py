"""Run the policy benchmark from the command line.

    python -m benchlab.cli --dataset v2-heldout --models Laya Jev
    python -m benchlab.cli --dataset v2-heldout --models Laya Jev --publish
"""
import argparse
import os
import shutil
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from .config import DATASETS, JEV_DEFAULT_MODEL, MODELS, ROOT
from .data import load_dataset, scored_cases
from .results import BENCHMARKS_DIR
from .runners import build_metadata, results_frame, run_cases, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS),
                        default="v2-heldout")
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=MODELS)
    parser.add_argument("--publish", action="store_true",
                        help="also copy the run into benchmarks/")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    jev_model = os.environ.get("JEV_MODEL", JEV_DEFAULT_MODEL)
    cases = load_dataset(args.dataset)
    scored = scored_cases(cases)
    started = datetime.now(timezone.utc)
    run_id = f"{args.dataset}_{started:%Y%m%dT%H%M%SZ}"

    rows, raws = [], {}
    for model_name in args.models:
        def report(index, row, model_name=model_name):
            print(f"{model_name} {index + 1}/{len(scored)} {row['id']}: "
                  f"{row['status']} {row['predicted_decision']} "
                  f"{row['error']}".rstrip(), flush=True)
        model_rows, model_raws = run_cases(
            model_name, scored, on_row=report,
            api_key=os.environ.get("OPENROUTER_API_KEY"), jev_model=jev_model,
        )
        rows += model_rows
        raws.update(model_raws)

    results = results_frame(rows)
    metadata = build_metadata(
        run_id=run_id, source="published" if args.publish else "local",
        dataset_version=args.dataset, results=results,
        started_utc=started.isoformat(timespec="seconds"),
        finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cases_run=len(scored), flagged_skipped=len(cases) - len(scored),
        jev_model=jev_model,
    )
    run_dir = ROOT / "results" / run_id
    save_run(run_dir, results, metadata, raws)
    print(f"Saved {run_dir}")
    if args.publish:
        shutil.copytree(run_dir, BENCHMARKS_DIR / run_id)
        print(f"Published {BENCHMARKS_DIR / run_id}")
    print(pd.crosstab([results["model"], results["expected_decision"]],
                      results["predicted_decision"].where(
                          results["status"].eq("ok"), "NO_DECISION")))


if __name__ == "__main__":
    main()
