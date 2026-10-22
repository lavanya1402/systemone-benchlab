"""Calls Laya and Jev on policy cases and records every outcome."""
import functools
import json
import subprocess
import time
import warnings
from importlib import metadata as importlib_metadata

import httpx
import pandas as pd

from .config import (
    DATASETS, DECISIONS, JEV_MAX_ATTEMPTS, JEV_TIMEOUT_S, OPENROUTER_URL,
    POLICY_QUESTIONS, PROMPT_VERSION, ROOT, build_request, jev_system_prompt,
    prompt_fingerprint,
)
from .data import dataset_sha256

STATUS_OK = "ok"
STATUS_REQUEST_FAILED = "request_failed"
STATUS_INVALID = "invalid_output"

RESULT_COLUMNS = [
    "id", "model", "model_id", "served_by", "provider", "expected_decision",
    "predicted_decision", "status", "latency_ms", "cost_usd",
    "reported_confidence", "explanation", "raw_output", "error", "warnings",
    "phase",
]
EXPLANATION_KEYS = ("explanation", "reason", "rationale")


def laya_version():
    try:
        return importlib_metadata.version("laya")
    except importlib_metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def laya_router():
    import laya
    return laya.Router(device="cpu", preload=False)


def _failure(status, error, raw_output=""):
    return {"status": status, "error": error, "raw_output": raw_output}


def call_laya(request):
    """Return (row fields, raw response) for one Laya policy decision."""
    try:
        result = laya_router().predict(
            {"request": request}, POLICY_QUESTIONS, model=None, lang="en",
        )
    except Exception as exc:
        return _failure(STATUS_REQUEST_FAILED, f"Laya error: {exc}"), None

    route = (result.get("routing") or {}).get("model")
    fields = {
        "served_by": " / ".join(filter(None, [result.get("model"), route])),
        "provider": "local CPU",
    }
    answer = (result.get("answers") or {}).get("decision")
    if not isinstance(answer, dict) or "choice" not in answer:
        fields.update(_failure(
            STATUS_INVALID, "Laya returned no decision field",
            json.dumps(result.get("answers"), default=str)[:500],
        ))
        return fields, result
    fields["reported_confidence"] = answer.get("answer_confidence")
    if answer["choice"] not in DECISIONS:
        fields.update(_failure(
            STATUS_INVALID, f"Decision {answer['choice']!r} is not a valid label",
            str(answer["choice"])[:500],
        ))
        return fields, result
    fields.update(status=STATUS_OK, predicted_decision=answer["choice"])
    return fields, result


def call_jev(request, api_key, model):
    """Return (row fields, raw response) for one Jev policy decision."""
    if not api_key:
        return _failure(STATUS_REQUEST_FAILED,
                        "OPENROUTER_API_KEY is not configured"), None
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": jev_system_prompt()},
            {"role": "user", "content": request},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "policy_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": DECISIONS},
                    },
                    "required": ["decision"],
                    "additionalProperties": False,
                },
            },
        },
    }
    try:
        for attempt in range(JEV_MAX_ATTEMPTS):
            response = httpx.post(
                OPENROUTER_URL, json=body, timeout=JEV_TIMEOUT_S,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            # OpenRouter answers 429 while admission control checks credits.
            if response.status_code != 429 or attempt == JEV_MAX_ATTEMPTS - 1:
                break
            time.sleep(int(response.headers.get("retry-after") or 10) + 1)
    except httpx.TimeoutException:
        return _failure(STATUS_REQUEST_FAILED,
                        f"Timed out after {JEV_TIMEOUT_S} s"), None
    except httpx.HTTPError as exc:
        return _failure(STATUS_REQUEST_FAILED, f"Network error: {exc}"), None

    try:
        result = response.json()
    except ValueError:
        status = (STATUS_INVALID if response.status_code == 200
                  else STATUS_REQUEST_FAILED)
        return _failure(status, f"HTTP {response.status_code}: response was "
                        "not JSON", response.text[:500]), None
    if not isinstance(result, dict):
        return _failure(STATUS_INVALID, "Response JSON was not an object",
                        response.text[:500]), None
    if response.status_code != 200 or "error" in result:
        error = result.get("error")
        message = (error.get("message") if isinstance(error, dict)
                   else str(error or "unknown error"))
        return _failure(STATUS_REQUEST_FAILED,
                        f"HTTP {response.status_code}: {message}"), result

    fields = {
        "served_by": result.get("model", ""),
        "provider": result.get("provider", ""),
        "cost_usd": (result.get("usage") or {}).get("cost"),
    }
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        fields.update(_failure(STATUS_INVALID,
                               "Response had no message content"))
        return fields, result
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        fields.update(_failure(STATUS_INVALID,
                               "Model output was not valid JSON",
                               str(content)[:500]))
        return fields, result
    decision = parsed.get("decision") if isinstance(parsed, dict) else None
    if decision not in DECISIONS:
        fields.update(_failure(
            STATUS_INVALID, f"Decision {decision!r} is not a valid label",
            str(content)[:500],
        ))
        return fields, result
    explanation = next((parsed[key] for key in EXPLANATION_KEYS
                        if isinstance(parsed.get(key), str)), "")
    fields.update(status=STATUS_OK, predicted_decision=decision,
                  explanation=explanation)
    return fields, result


def run_case(model_name, case, *, phase, api_key=None, jev_model=None):
    """Run one case and return (result row, raw response or None)."""
    row = dict.fromkeys(RESULT_COLUMNS, "")
    row.update(
        id=case["id"], model=model_name, phase=phase,
        expected_decision=case.get("expected_decision", ""),
        model_id=(jev_model if model_name == "Jev"
                  else f"laya {laya_version() or ''}".strip()),
        latency_ms=None, cost_usd=None, reported_confidence=None,
    )
    request = build_request(case["policy"], case["action"], case["context"])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if model_name == "Laya":
            fields, raw = call_laya(request)
        else:
            fields, raw = call_jev(request, api_key, jev_model)
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    row.update(fields)
    row["warnings"] = " | ".join(str(item.message) for item in captured)
    return row, raw


def run_cases(model_name, cases, *, on_row=None, **kwargs):
    rows, raws = [], {}
    for index, case in enumerate(cases.to_dict("records")):
        row, raw = run_case(
            model_name, case,
            phase="first_request" if index == 0 else "subsequent", **kwargs,
        )
        rows.append(row)
        raws[f"{model_name.lower()}_{case['id']}"] = raw
        if on_row:
            on_row(index, row)
    return rows, raws


def git_commit():
    """Short commit hash, marked when the code differs from that commit."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=5)
    try:
        commit = git("rev-parse", "--short", "HEAD").stdout.strip()
        if not commit:
            return None
        dirty = git("status", "--porcelain", "--untracked-files=normal").stdout
        return f"{commit} + uncommitted changes" if dirty.strip() else commit
    except Exception:
        return None


def build_metadata(*, run_id, source, dataset_version, results, started_utc,
                   finished_utc, cases_run, flagged_skipped, jev_model):
    models = {}
    for model_name in results["model"].unique():
        rows = results[results["model"].eq(model_name)]
        served = sorted(set(rows["served_by"]) - {""})
        if model_name == "Laya":
            models["Laya"] = {
                "system": "Laya (local checkpoint)",
                "package_version": laya_version(),
                "served_by": served,
                "settings": {
                    "language": "en", "device": "cpu",
                    "checkpoint_selection": "automatic (model=None)",
                },
            }
        else:
            models["Jev"] = {
                "system": "Jev Router (routed system, not a single model)",
                "requested_model": jev_model,
                "served_by": served,
                "providers": sorted(set(rows["provider"]) - {""}),
                "settings": {
                    "temperature": 0,
                    "response_format": "strict JSON schema; decision enum",
                    "timeout_s": JEV_TIMEOUT_S,
                    "retries": f"up to {JEV_MAX_ATTEMPTS - 1} on HTTP 429",
                },
            }
    return {
        "run_id": run_id,
        "source": source,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "dataset": {
            "version": dataset_version,
            "file": DATASETS[dataset_version]["file"],
            "sha256": dataset_sha256(dataset_version),
            "cases_run": cases_run,
            "flagged_for_review_skipped": flagged_skipped,
        },
        "prompt": {"version": PROMPT_VERSION,
                   "fingerprint": prompt_fingerprint()},
        "models": models,
        "app_commit": git_commit(),
    }


def save_run(run_dir, results, metadata, raws):
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results[RESULT_COLUMNS].to_csv(run_dir / "results.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    for name, raw in raws.items():
        if raw is not None:
            (raw_dir / f"{name}.json").write_text(
                json.dumps(raw, indent=2, default=str), encoding="utf-8")


def results_frame(rows):
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
