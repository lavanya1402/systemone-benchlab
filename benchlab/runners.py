"""Calls Laya, Jev and Jev Router on policy cases and records every outcome.

Every model receives only the request text built from policy, proposed
action and context, plus the same decision question. Case IDs, expected
labels, scenarios and review notes are never sent.
"""
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
    POLICY_QUESTIONS, PROMPT_VERSION, RAW_PREFIXES, ROOT,
    TYPESAFE_MAX_ATTEMPTS, TYPESAFE_TIMEOUT_S, TYPESAFE_URL,
    TYPESAFE_USD_PER_INPUT_TOKEN, build_request, jev_system_prompt,
    prompt_fingerprint,
)
from .data import dataset_sha256

STATUS_OK = "ok"
STATUS_REQUEST_FAILED = "request_failed"
STATUS_INVALID = "invalid_output"

RESULT_COLUMNS = [
    "id", "model", "model_id", "served_by", "provider", "expected_decision",
    "predicted_decision", "status", "latency_ms", "cost_usd",
    "input_tokens", "output_tokens", "reported_confidence", "explanation",
    "raw_output", "error", "warnings", "phase",
]
NUMERIC_COLUMNS = ["latency_ms", "cost_usd", "input_tokens", "output_tokens",
                   "reported_confidence"]
EXPLANATION_KEYS = ("explanation", "reason", "rationale")
CREDENTIALS = {
    "Jev Direct": ("typesafe_api_key", "TYPESAFE_API_KEY"),
    "Jev Router": ("openrouter_api_key", "OPENROUTER_API_KEY"),
}


def missing_credential(model_name, settings):
    """Name of the missing environment variable, or None."""
    key, env_name = CREDENTIALS.get(model_name, (None, None))
    return env_name if key and not settings.get(key) else None


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


def _choice_fields(answer, model_name):
    """Validate a native choice answer shared by Laya and Jev direct."""
    if not isinstance(answer, dict) or "choice" not in answer:
        return _failure(STATUS_INVALID,
                        f"{model_name} returned no decision field",
                        json.dumps(answer, default=str)[:500])
    fields = {"reported_confidence": answer.get("answer_confidence",
                                                answer.get("confidence"))}
    if answer["choice"] not in DECISIONS:
        fields.update(_failure(
            STATUS_INVALID, f"Decision {answer['choice']!r} is not a valid label",
            str(answer["choice"])[:500],
        ))
        return fields
    fields.update(status=STATUS_OK, predicted_decision=answer["choice"])
    return fields


def call_laya(request):
    """Return (row fields, raw response) for one Laya policy decision."""
    try:
        result = laya_router().predict(
            {"request": request}, POLICY_QUESTIONS, model=None, lang="en",
        )
    except Exception as exc:
        return _failure(STATUS_REQUEST_FAILED, f"Laya error: {exc}"), None

    route = (result.get("routing") or {}).get("model")
    usage = result.get("usage") or {}
    fields = {
        "served_by": " / ".join(filter(None, [result.get("model"), route])),
        "provider": "local CPU",
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    fields.update(_choice_fields((result.get("answers") or {}).get("decision"),
                                 "Laya"))
    return fields, result


def typesafe_request_body(request, model):
    """Native System One request: the same choice question Laya answers."""
    return {"model": model, "state": {"request": request},
            "questions": POLICY_QUESTIONS}


def _post_with_retries(url, body, api_key, timeout, attempts, retry_statuses):
    """POST, retrying the given statuses with backoff. Raises httpx errors."""
    for attempt in range(attempts):
        response = httpx.post(
            url, json=body, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code not in retry_statuses or attempt == attempts - 1:
            return response
        retry_after = response.headers.get("retry-after")
        time.sleep(int(retry_after) + 1 if retry_after and retry_after.isdigit()
                   else 2 ** attempt)
    return response


def _error_message(response):
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or "no response body"
    if isinstance(body, dict):
        error = body.get("error", body.get("detail"))
        if isinstance(error, dict):
            return str(error.get("message", error))[:300]
        if error is not None:
            return json.dumps(error, default=str)[:300]
    return json.dumps(body, default=str)[:300]


def call_jev_direct(request, api_key, model):
    """Return (row fields, raw response) from TypeSafe's Jev decision API."""
    if not api_key:
        return _failure(STATUS_REQUEST_FAILED,
                        "TYPESAFE_API_KEY is not configured"), None
    try:
        # 429 rate limit and 529 overloaded are retryable per the docs.
        response = _post_with_retries(
            TYPESAFE_URL, typesafe_request_body(request, model), api_key,
            TYPESAFE_TIMEOUT_S, TYPESAFE_MAX_ATTEMPTS, {429, 529},
        )
    except httpx.TimeoutException:
        return _failure(STATUS_REQUEST_FAILED,
                        f"Timed out after {TYPESAFE_TIMEOUT_S} s"), None
    except httpx.HTTPError as exc:
        return _failure(STATUS_REQUEST_FAILED, f"Network error: {exc}"), None

    if response.status_code != 200:
        return _failure(STATUS_REQUEST_FAILED,
                        f"HTTP {response.status_code}: "
                        f"{_error_message(response)}"), None
    try:
        result = response.json()
    except ValueError:
        return _failure(STATUS_INVALID, "Response was not JSON",
                        response.text[:500]), None
    if not isinstance(result, dict):
        return _failure(STATUS_INVALID, "Response JSON was not an object",
                        response.text[:500]), None

    usage = result.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    fields = {
        # The versioned model ID that answered, as reported by TypeSafe.
        "served_by": result.get("model", ""),
        "provider": "TypeSafe API",
        "input_tokens": input_tokens,
        "output_tokens": usage.get("output_tokens"),
    }
    fields.update(_choice_fields((result.get("answers") or {}).get("decision"),
                                 "Jev"))
    return fields, result


def call_jev_router(request, api_key, model):
    """Return (row fields, raw response) for one Jev Router decision."""
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
        # OpenRouter answers 429 while admission control checks credits.
        response = _post_with_retries(OPENROUTER_URL, body, api_key,
                                      JEV_TIMEOUT_S, JEV_MAX_ATTEMPTS, {429})
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
        return _failure(STATUS_REQUEST_FAILED,
                        f"HTTP {response.status_code}: "
                        f"{_error_message(response)}"), result

    usage = result.get("usage") or {}
    fields = {
        "served_by": result.get("model", ""),
        "provider": result.get("provider", ""),
        "cost_usd": usage.get("cost"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
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


def model_id_for(model_name, settings):
    if model_name == "Jev Direct":
        return settings.get("jev_direct_model", "")
    if model_name == "Jev Router":
        return settings.get("jev_router_model", "")
    return f"laya {laya_version() or ''}".strip()


def run_case(model_name, case, *, phase, settings):
    """Run one case and return (result row, raw response or None)."""
    row = dict.fromkeys(RESULT_COLUMNS, "")
    row.update(
        id=case["id"], model=model_name, phase=phase,
        expected_decision=case.get("expected_decision", ""),
        model_id=model_id_for(model_name, settings),
        **dict.fromkeys(NUMERIC_COLUMNS),
    )
    # Only these three fields reach any model.
    request = build_request(case["policy"], case["action"], case["context"])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if model_name == "Laya":
            fields, raw = call_laya(request)
        elif model_name == "Jev Direct":
            fields, raw = call_jev_direct(
                request, settings.get("typesafe_api_key"),
                settings.get("jev_direct_model"))
        elif model_name == "Jev Router":
            fields, raw = call_jev_router(
                request, settings.get("openrouter_api_key"),
                settings.get("jev_router_model"))
        else:
            raise ValueError(f"Unknown model {model_name!r}")
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    row.update(fields)
    row["warnings"] = " | ".join(str(item.message) for item in captured)
    return row, raw


def run_cases(model_name, cases, *, settings, on_row=None):
    rows, raws = [], {}
    for index, case in enumerate(cases.to_dict("records")):
        row, raw = run_case(
            model_name, case, settings=settings,
            phase="first_request" if index == 0 else "subsequent",
        )
        rows.append(row)
        raws[f"{RAW_PREFIXES[model_name][0]}_{case['id']}"] = raw
        if on_row:
            on_row(index, row)
    return rows, raws


def estimated_cost_usd(rows):
    """Jev direct cost from input tokens at the documented price, or None."""
    tokens = pd.to_numeric(rows["input_tokens"], errors="coerce")
    if rows.empty or tokens.isna().any():
        return None
    return float(tokens.sum()) * TYPESAFE_USD_PER_INPUT_TOKEN


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
                   finished_utc, cases_run, flagged_skipped, settings):
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
        elif model_name == "Jev Direct":
            cost = estimated_cost_usd(rows)
            models["Jev Direct"] = {
                "system": "Jev — direct decision model (TypeSafe System One API)",
                "requested_model": settings.get("jev_direct_model"),
                "served_by": served,
                "endpoint": TYPESAFE_URL,
                "settings": {
                    "question": "native choice question, identical to Laya's",
                    "timeout_s": TYPESAFE_TIMEOUT_S,
                    "retries": f"up to {TYPESAFE_MAX_ATTEMPTS - 1} on HTTP 429/529",
                },
                "cost": (
                    "Not returned by the API. Estimated at "
                    f"${cost:.6f} from input tokens at the documented "
                    "$0.042 per million input tokens (output free)."
                    if cost is not None else "Unavailable"
                ),
            }
        else:
            models["Jev Router"] = {
                "system": "Jev Router via OpenRouter (routed system, not "
                          "a single model)",
                "requested_model": settings.get("jev_router_model"),
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
