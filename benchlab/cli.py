"""Run the policy benchmark from the command line.

    python -m benchlab.cli --dataset v2-heldout --models Laya "Jev Direct"
    python -m benchlab.cli --dataset v2-heldout --models "Jev Direct" --cases V2-003
    python -m benchlab.cli --models "Jev Direct" --cases V2-003 --dry-run
    python -m benchlab.cli --dataset v2-heldout --models Laya "Jev Direct" --publish
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from .config import (
    DATASETS, JEV_DEFAULT_MODEL, MODELS, ROOT, TYPESAFE_DEFAULT_MODEL,
    TYPESAFE_URL, build_request, model_label,
)
from .data import load_dataset, scored_cases
from .results import BENCHMARKS_DIR
from .runners import (
    build_metadata, estimated_cost_usd, missing_credential, results_frame,
    run_cases, save_run, typesafe_request_body,
)


def load_settings():
    load_dotenv(ROOT / ".env")
    return {
        "typesafe_api_key": os.environ.get("TYPESAFE_API_KEY"),
        "jev_direct_model": os.environ.get("TYPESAFE_MODEL",
                                           TYPESAFE_DEFAULT_MODEL),
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
        "jev_router_model": os.environ.get("JEV_MODEL", JEV_DEFAULT_MODEL),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS),
                        default="v2-heldout")
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["Laya", "Jev Direct"])
    parser.add_argument("--cases", nargs="+", metavar="ID",
                        help="run only these labelled case IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the request for each case; call nothing")
    parser.add_argument("--publish", action="store_true",
                        help="also copy the run into benchmarks/")
    args = parser.parse_args()

    settings = load_settings()
    cases = load_dataset(args.dataset)
    scored = scored_cases(cases)
    if args.cases:
        unknown = set(args.cases) - set(scored["id"])
        if unknown:
            sys.exit(f"Not labelled cases in {args.dataset}: "
                     f"{', '.join(sorted(unknown))}")
        scored = scored[scored["id"].isin(args.cases)]
    print(f"{len(scored)} labelled cases; {len(cases) - len(scored_cases(cases))} "
          "flagged cases excluded from scoring.")

    if args.dry_run:
        for case in scored.to_dict("records"):
            request = build_request(case["policy"], case["action"],
                                    case["context"])
            print(f"\n--- {case['id']} (ID and label are not sent)")
            if "Jev Direct" in args.models:
                print(f"POST {TYPESAFE_URL}\nAuthorization: Bearer <hidden>")
                print(json.dumps(typesafe_request_body(
                    request, settings["jev_direct_model"]), indent=2))
            else:
                print(request)
        return

    for model_name in args.models:
        missing = missing_credential(model_name, settings)
        if missing:
            sys.exit(f"{model_label(model_name)} needs {missing} in .env or "
                     "the environment.")

    started = datetime.now(timezone.utc)
    run_id = f"{args.dataset}_{started:%Y%m%dT%H%M%SZ}"
    rows, raws = [], {}
    for model_name in args.models:
        def report(index, row, model_name=model_name):
            print(f"{model_label(model_name)} {index + 1}/{len(scored)} "
                  f"{row['id']}: {row['status']} {row['predicted_decision']} "
                  f"answered_by={row['served_by'] or '-'} "
                  f"{row['latency_ms']:.0f} ms {row['error']}".rstrip(),
                  flush=True)
        model_rows, model_raws = run_cases(model_name, scored, on_row=report,
                                           settings=settings)
        rows += model_rows
        raws.update(model_raws)

    results = results_frame(rows)
    metadata = build_metadata(
        run_id=run_id, source="published" if args.publish else "local",
        dataset_version=args.dataset, results=results,
        started_utc=started.isoformat(timespec="seconds"),
        finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cases_run=len(scored),
        flagged_skipped=len(cases) - len(scored_cases(cases)),
        settings=settings,
    )
    if args.cases:
        metadata["dataset"]["case_subset"] = sorted(args.cases)
    run_dir = ROOT / "results" / run_id
    save_run(run_dir, results, metadata, raws)
    print(f"Saved {run_dir}")
    direct = results[results["model"].eq("Jev Direct")]
    if not direct.empty:
        cost = estimated_cost_usd(direct)
        tokens = pd.to_numeric(direct["input_tokens"], errors="coerce").sum()
        print("Jev direct input tokens:", int(tokens)
              if cost is not None else "unavailable",
              "| estimated cost:", f"${cost:.6f}" if cost is not None
              else "unavailable")
    if args.publish:
        shutil.copytree(run_dir, BENCHMARKS_DIR / run_id)
        print(f"Published {BENCHMARKS_DIR / run_id}")
    print(pd.crosstab([results["model"], results["expected_decision"]],
                      results["predicted_decision"].where(
                          results["status"].eq("ok"), "NO_DECISION")))


if __name__ == "__main__":
    main()
