import hashlib
import importlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# Dependency order: later modules import from earlier ones.
BENCHLAB_MODULES = ("benchlab.config", "benchlab.data", "benchlab.runners",
                    "benchlab.metrics", "benchlab.results")


def source_digest(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def reload_changed_benchlab_modules():
    """Reload benchlab modules whose source changed since they were imported.

    Streamlit Cloud reruns app.py after a git pull but keeps modules it has
    already imported, so new names would otherwise raise ImportError.
    """
    for name in BENCHLAB_MODULES:
        module = sys.modules.get(name)
        if module is not None and getattr(
                module, "_source_digest", None) != source_digest(module):
            importlib.reload(module)


reload_changed_benchlab_modules()

from benchlab import metrics  # noqa: E402
from benchlab.config import (
    DATASETS, DECISION_INFO, DECISIONS, JEV_DEFAULT_MODEL, MODEL_COLORS,
    MODELS, NO_DECISION, PROMPT_VERSION,
    RAW_PREFIXES, ROOT, TYPESAFE_DEFAULT_MODEL, TYPESAFE_MAX_ATTEMPTS,
    TYPESAFE_TIMEOUT_S, model_label, prompt_fingerprint,
)
from benchlab.data import dataset_sha256, load_dataset, scored_cases
from benchlab.results import (
    NOT_RECORDED, list_published_runs, load_raw, load_run, normalize_results,
)
from benchlab.runners import (
    build_metadata, estimated_cost_usd, laya_router, laya_version,
    missing_credential, results_frame, run_case, run_cases, save_run,
)

# Record what was loaded so the next rerun only reloads changed modules.
for _name in BENCHLAB_MODULES:
    sys.modules[_name]._source_digest = source_digest(sys.modules[_name])

st.set_page_config(
    page_title="SystemOne BenchLab",
    page_icon=":material/balance:",
    layout="wide",
)

load_dotenv(ROOT / ".env")


def secret(name, default=None):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets file locally
        return default


# Keys are read from .env locally or Streamlit secrets when deployed. They
# are passed to the runners only and never displayed.
SETTINGS = {
    "typesafe_api_key": secret("TYPESAFE_API_KEY"),
    "jev_direct_model": secret("TYPESAFE_MODEL", TYPESAFE_DEFAULT_MODEL),
    "openrouter_api_key": secret("OPENROUTER_API_KEY"),
    "jev_router_model": secret("JEV_MODEL", JEV_DEFAULT_MODEL),
}
# The app compares two models. Earlier Laya vs Jev Router runs stay in
# benchmarks/ and the README, and an uploaded file may still contain them,
# but they are not shown or offered here.
APP_MODELS = ["Laya", "Jev Direct"]
DEFAULT_RUN_MODELS = APP_MODELS
MODEL_BY_LABEL = {model_label(m): m for m in MODELS}
RUNNABLE_BY_LABEL = {model_label(m): m for m in APP_MODELS}


def credential_warning(model_name):
    missing = missing_credential(model_name, SETTINGS)
    if missing:
        st.warning(f"{model_label(model_name)} needs `{missing}` in `.env` "
                   "(local) or Streamlit secrets (deployed), so it will be "
                   "skipped.", icon=":material/key_off:")
    return missing
DATASET_BY_TITLE = {spec["title"]: version
                    for version, spec in DATASETS.items()}
DISCLAIMER = ("Results apply to this dataset and run; they do not establish "
              "universal reliability.")
CASE_COLUMNS = ["id", "business_area", "scenario", "policy", "action",
                "context", "label_status"]
METRIC_HELP = {
    "accuracy": ("Share of scored cases with the correct decision. Failed "
                 "requests and invalid outputs count as incorrect."),
    "macro_f1": ("Average of the F1 scores for ALLOW, REQUIRE_APPROVAL and "
                 "BLOCK, so each decision type counts equally. Ranges 0 to 1."),
    "latency": ("Median time per successful request, excluding each model's "
                "first request, which can include loading. Laya runs on local "
                "CPU; Jev includes the network round trip."),
    "failed": ("Requests that timed out or returned an API or network error. "
               "Recorded, never converted into a decision."),
    "invalid": ("Responses that could not be parsed into one of the three "
                "decisions. Recorded, never converted into a decision."),
    "cost": "Cost reported by OpenRouter for this run.",
    "estimated_cost": ("TypeSafe does not return a cost. Estimated from the "
                       "reported input tokens at the documented $0.042 per "
                       "million input tokens; output tokens are free."),
}


@st.cache_data(max_entries=8)
def cached_dataset(version):
    return load_dataset(version)


@st.cache_data(ttl=300)
def cached_published_runs():
    return [(str(path), meta) for path, meta in list_published_runs()]


@st.cache_data(max_entries=16)
def cached_run(path):
    return load_run(Path(path))


def show(value):
    if value is None or value == "" or value == []:
        return NOT_RECORDED
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def confidence_note(model_name, value):
    note = (f"{model_label(model_name)}-reported confidence for its answer: "
            f"{value:.2f}. This is the model's own estimate, not a measure of "
            "accuracy.")
    if model_name == "Laya":
        note += (" Laya's checkpoint warns about invalid temperatures, so "
                 "treat it as uncalibrated.")
    return note


def decision_badge(label):
    info = DECISION_INFO.get(label)
    if info:
        st.badge(label, icon=info["icon"], color=info["color"])
    else:
        st.badge(label or NO_DECISION, icon=":material/error:", color="gray")


def run_label(meta):
    dataset = DATASETS.get(meta.get("dataset", {}).get("version"), {})
    started = (meta.get("started_utc") or "")[:16].replace("T", " ")
    return f"{dataset.get('title', 'Unknown dataset')} · {started} UTC"


def join_cases(results, dataset_version):
    results = results.copy()
    if dataset_version in DATASETS:
        cases = cached_dataset(dataset_version)[CASE_COLUMNS]
        results = results.merge(cases, on="id", how="left")
    for column in CASE_COLUMNS[1:]:
        if column not in results:
            results[column] = ""
    results[CASE_COLUMNS[1:]] = results[CASE_COLUMNS[1:]].fillna("")
    results["actual_decision"] = metrics.scored_prediction(results)
    results["error_type"] = results.apply(metrics.error_type, axis=1)
    return results


def model_identity(model_name, rows, meta):
    served = sorted(set(rows["served_by"]) - {""})
    requested = (meta.get("models", {}).get(model_name, {})
                 .get("requested_model")
                 or show(sorted(set(rows["model_id"]) - {""})))
    if model_name == "Jev Router":
        return (f"Routed system via OpenRouter · requested `{requested}` · "
                f"answered by {len(served) or 'an unrecorded number of'} "
                f"underlying model{'s' if len(served) != 1 else ''} on this run")
    if model_name == "Jev Direct":
        return (f"TypeSafe decision API · requested `{requested}` · answered "
                f"by {show(served)}")
    return f"Local checkpoint · served by {show(served)}"


def headline_metrics(results, meta, compact=False):
    models = [m for m in MODELS if m in set(results["model"])]
    for column, model_name in zip(st.columns(len(models)), models):
        rows = results[results["model"].eq(model_name)]
        summary = metrics.summary(rows)
        with column.container(border=True):
            st.markdown(f"#### {model_label(model_name)}")
            st.caption(model_identity(model_name, rows, meta))
            left, right = st.columns(2)
            left.metric("Accuracy", f"{summary['accuracy']:.1%}",
                        help=METRIC_HELP["accuracy"])
            left.caption(f"{summary['correct']} of {summary['cases']} correct")
            right.metric("Macro F1", f"{summary['macro_f1']:.3f}",
                         help=METRIC_HELP["macro_f1"])
            if compact:
                missed = metrics.critical_errors(rows).iloc[0]
                st.markdown(
                    f"Approval required → ALLOW: **{missed['count']} of "
                    f"{missed['denominator']}** approval cases"
                )
                continue
            latency = summary["median_latency_ms"]
            left.metric("Median latency",
                        "n/a" if pd.isna(latency) else f"{latency:,.0f} ms",
                        help=METRIC_HELP["latency"])
            right.metric("Failed requests", summary["request_failed"],
                         help=METRIC_HELP["failed"])
            left.metric("Invalid outputs", summary["invalid_output"],
                        help=METRIC_HELP["invalid"])
            if summary["total_cost_usd"] is not None:
                right.metric("Cost", f"${summary['total_cost_usd']:.6f}",
                             help=METRIC_HELP["cost"])
            elif model_name == "Jev Direct":
                estimate = estimated_cost_usd(rows)
                right.metric("Estimated cost",
                             "Unavailable" if estimate is None
                             else f"${estimate:.6f}",
                             help=METRIC_HELP["estimated_cost"])


def metadata_table(meta):
    dataset = meta.get("dataset", {})
    prompt = meta.get("prompt", {})
    rows = [
        ("Run ID", meta.get("run_id")),
        ("Started (UTC)", meta.get("started_utc")),
        ("Finished (UTC)", meta.get("finished_utc")),
        ("Dataset version", dataset.get("version")),
        ("Dataset SHA-256", dataset.get("sha256")),
        ("Cases run", dataset.get("cases_run")),
        ("Flagged cases skipped", dataset.get("flagged_for_review_skipped")),
        ("Prompt version", prompt.get("version")),
        ("Prompt fingerprint", prompt.get("fingerprint")),
    ]
    for model_name, info in meta.get("models", {}).items():
        name = model_label(model_name)
        rows += [
            (f"{name}: system", info.get("system")),
            (f"{name}: requested model", info.get("requested_model")),
            (f"{name}: package version", info.get("package_version")),
            (f"{name}: answered by", info.get("served_by")),
            (f"{name}: endpoint", info.get("endpoint")),
            (f"{name}: providers", info.get("providers")),
            (f"{name}: settings", info.get("settings")),
            (f"{name}: cost", info.get("cost")),
        ]
    rows.append(("App commit", meta.get("app_commit")))
    # Fields that do not apply to a system are omitted, not "Not recorded".
    optional = ("requested model", "package version", "endpoint",
                "providers", "cost")
    table = pd.DataFrame(
        [(field, show(value)) for field, value in rows
         if not (value is None and ": " in field
                 and field.rsplit(": ", 1)[1] in optional)],
        columns=["Field", "Value"],
    )
    st.table(table, hide_index=True, alt="Run metadata")
    for note in meta.get("notes", []):
        st.caption(f"Note: {note}")


def recall_chart(results):
    rows = []
    present = [m for m in MODELS if m in set(results["model"])]
    for model_name in present:
        table = metrics.per_class(results[results["model"].eq(model_name)])
        table["model"] = model_label(model_name)
        rows.append(table)
    data = pd.concat(rows)
    data["label"] = data["recall"].map("{:.0%}".format)
    domain = [model_label(m) for m in present]
    colors = [MODEL_COLORS[m] for m in present]
    base = alt.Chart(data).encode(
        x=alt.X("decision:N", sort=DECISIONS, title="Expected decision",
                axis=alt.Axis(labelAngle=0, labelLimit=0)),
        xOffset=alt.XOffset("model:N", sort=domain),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("decision:N", title="Expected decision"),
            alt.Tooltip("recall:Q", title="Recall", format=".0%"),
            alt.Tooltip("support:Q", title="Cases"),
        ],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("recall:Q", title="Recall",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", title="Model", scale=alt.Scale(
            domain=domain, range=colors),
            legend=alt.Legend(orient="bottom", labelLimit=0)),
    )
    labels = base.mark_text(dy=-8, color="#0F1E3D").encode(
        y="recall:Q", text="label:N")
    return (bars + labels).properties(height=320)


def confusion_chart(rows):
    data = metrics.confusion(rows)
    columns = DECISIONS + [NO_DECISION]
    peak = max(int(data["count"].max()), 1)
    base = alt.Chart(data).encode(
        x=alt.X("predicted:N", sort=columns, title="Predicted decision",
                axis=alt.Axis(labelAngle=-30, orient="top", labelLimit=0,
                              labelOverlap=False)),
        y=alt.Y("expected:N", sort=DECISIONS, title="Expected decision",
                axis=alt.Axis(labelLimit=0)),
    )
    cells = base.mark_rect(stroke="#FFFFFF", strokeWidth=2).encode(
        color=alt.Color("count:Q", title="Cases",
                        scale=alt.Scale(scheme="blues", domain=[0, peak]),
                        legend=alt.Legend(gradientLength=140)),
        tooltip=[alt.Tooltip("expected:N", title="Expected"),
                 alt.Tooltip("predicted:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Cases")],
    )
    text = base.mark_text(fontSize=14, fontWeight=600).encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > peak / 2,
                            alt.value("#FFFFFF"), alt.value("#0F1E3D")),
    )
    return (cells + text).properties(height=240)


def render_results(results, meta, raw_lookup):
    source = meta.get("source")
    if source == "published":
        st.badge("Published result", icon=":material/verified:", color="blue")
    elif source == "live":
        st.badge("Live run · not published", icon=":material/bolt:",
                 color="orange")
    else:
        st.badge("Uploaded file · metadata not recorded",
                 icon=":material/upload_file:", color="gray")
    st.info(DISCLAIMER, icon=":material/info:")

    dataset = meta.get("dataset", {})
    facts = st.columns(4)
    facts[0].metric("Run started (UTC)",
                    show((meta.get("started_utc") or "")[:16].replace("T", " ")))
    facts[1].metric("Dataset", show(dataset.get("version")))
    facts[2].metric("Scored cases", show(dataset.get("cases_run")))
    facts[3].metric("Prompt version", show(meta.get("prompt", {}).get("version")))
    with st.expander("Full run metadata", icon=":material/description:"):
        metadata_table(meta)

    joined = join_cases(results, dataset.get("version"))
    models = [m for m in MODELS if m in set(joined["model"])]

    st.subheader("Headline metrics")
    headline_metrics(joined, meta)
    with st.expander("How these metrics are calculated",
                     icon=":material/calculate:"):
        st.markdown(
            "- **Accuracy**: correct decisions ÷ all scored cases. A failed "
            "request or invalid output counts as incorrect.\n"
            "- **Precision** (per decision): of the cases where the model "
            "gave this decision, the share that were right. Failed and "
            "invalid responses are not counted as any decision, so they do "
            "not affect precision.\n"
            "- **Recall** (per decision): of the cases whose correct answer "
            "is this decision, the share the model got right. Failed and "
            "invalid responses count as misses.\n"
            "- **F1**: the balance of precision and recall. **Macro F1** "
            "averages F1 over the three decisions equally.\n"
            "- **Median latency**: successful requests only, excluding each "
            "model's first request.\n"
            "- **Not scored**: cases flagged for review (no single defensible "
            "answer) are never run or scored.\n"
            "- Model-reported confidence is shown in case details only. It "
            "is not a measure of accuracy and is not used in any metric."
        )

    st.subheader("Critical errors")
    st.caption("Each count is out of the cases with that correct answer.")
    meanings = {
        metrics.MISSED_APPROVAL: "Would act without the sign-off the policy requires.",
        metrics.MISSED_BLOCK: "Would perform an action the policy prohibits.",
        metrics.OVER_BLOCK: "Would refuse an action the policy permits.",
    }
    critical = pd.DataFrame({"Error": list(meanings),
                             "What it means": list(meanings.values())})
    for model_name in models:
        table = metrics.critical_errors(joined[joined["model"].eq(model_name)])
        critical[model_label(model_name)] = [
            f"{r['count']} of {r['denominator']}"
            + (f" ({r['rate']:.0%})" if r["rate"] is not None else "")
            for r in table.to_dict("records")
        ]
    st.table(critical, hide_index=True,
             alt="Critical error counts with denominators for each model")

    st.subheader("Recall by expected decision")
    st.caption("Share of cases with each correct answer that the model "
               "answered correctly.")
    st.altair_chart(recall_chart(joined),
                    alt="Recall for each expected decision, by model")

    st.subheader("Per-class precision, recall and F1")
    per_class = []
    for model_name in models:
        table = metrics.per_class(joined[joined["model"].eq(model_name)])
        table.insert(0, "model", model_label(model_name))
        per_class.append(table)
    st.dataframe(
        pd.concat(per_class), hide_index=True,
        alt="Precision, recall, F1 and support per decision and model",
        column_config={
            "model": "Model", "decision": "Decision",
            "precision": st.column_config.NumberColumn("Precision", format="%.2f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.2f"),
            "f1": st.column_config.NumberColumn("F1", format="%.2f"),
            "support": st.column_config.NumberColumn("Support (cases)"),
        },
    )

    st.subheader("Confusion matrices")
    st.caption("Rows are the correct answer; columns are the model's answer. "
               "Diagonal cells are correct. NO_DECISION means the request "
               "failed or the output was invalid.")
    for column, model_name in zip(st.columns(len(models)), models):
        with column:
            st.markdown(f"**{model_label(model_name)}**")
            st.altair_chart(
                confusion_chart(joined[joined["model"].eq(model_name)]),
                alt=f"{model_label(model_name)} confusion matrix of expected "
                    "versus predicted decisions",
            )

    if "Jev Router" in models:
        st.subheader("Where Jev Router routed requests")
        st.caption("Jev Router (typesafe/jev-router on OpenRouter) is a "
                   "routed system, not the direct Jev decision model. Each "
                   "request can be answered by a different underlying model, "
                   "so results describe the routed system on this run.")
        jev = joined[joined["model"].eq("Jev Router")].copy()
        jev["served_by"] = jev["served_by"].replace("", NOT_RECORDED)
        jev["provider"] = jev["provider"].replace("", NOT_RECORDED)
        jev["correct"] = jev["error_type"].eq(metrics.CORRECT)
        routing = (jev.groupby(["served_by", "provider"])
                   .agg(requests=("id", "count"), correct=("correct", "sum"))
                   .reset_index().sort_values("requests", ascending=False))
        st.dataframe(routing, hide_index=True,
                     alt="Number of Jev Router requests answered by each model",
                     column_config={"served_by": "Answered by",
                                    "provider": "Provider",
                                    "requests": "Requests",
                                    "correct": "Correct"})

    st.subheader("Case explorer")
    filters = st.columns(5)
    areas = sorted(set(joined["business_area"]) - {""})
    joined["system"] = joined["model"].map(model_label)
    labels = [model_label(m) for m in models]
    pick_models = filters[0].multiselect("Model", labels, default=labels)
    pick_expected = filters[1].multiselect("Expected decision", DECISIONS)
    pick_actual = filters[2].multiselect("Predicted decision",
                                         DECISIONS + [NO_DECISION])
    present_errors = [e for e in metrics.ERROR_TYPES
                      if e in set(joined["error_type"])]
    pick_errors = filters[3].multiselect(
        "Error type", present_errors,
        default=[e for e in present_errors if e != metrics.CORRECT]
        or present_errors,
    )
    pick_areas = filters[4].multiselect("Business area", areas)
    view = joined[joined["system"].isin(pick_models)]
    if pick_expected:
        view = view[view["expected_decision"].isin(pick_expected)]
    if pick_actual:
        view = view[view["actual_decision"].isin(pick_actual)]
    if pick_errors:
        view = view[view["error_type"].isin(pick_errors)]
    if pick_areas:
        view = view[view["business_area"].isin(pick_areas)]

    columns = ["id", "system", "business_area", "policy", "action", "context",
               "expected_decision", "actual_decision", "error_type",
               "served_by"]
    # Explanations are shown only when a model actually returned them.
    if joined["explanation"].ne("").any():
        columns.append("explanation")
    event = st.dataframe(
        view[columns], hide_index=True, key="case_table",
        on_select="rerun", selection_mode="single-row",
        alt="Benchmark cases with policy, action, context and both answers",
        column_config={
            "id": "Case", "system": "Model", "business_area": "Area",
            "policy": st.column_config.TextColumn("Policy", width="large"),
            "action": st.column_config.TextColumn("Proposed action", width="medium"),
            "context": st.column_config.TextColumn("Context", width="medium"),
            "expected_decision": "Expected", "actual_decision": "Model answer",
            "error_type": "Result", "served_by": "Answered by",
            "explanation": "Model explanation",
        },
    )
    st.caption(f"Showing {len(view)} of {len(joined)} results. Select a row "
               "to see the full case and the model's raw response.")
    if event.selection.rows and event.selection.rows[0] < len(view):
        row = view.iloc[event.selection.rows[0]]
        with st.container(border=True):
            st.markdown(f"**Case {row['id']} · {row['system']}** · "
                        f"{row['business_area'] or 'No area recorded'}")
            st.markdown(f"**Policy:** {row['policy'] or NOT_RECORDED}")
            st.markdown(f"**Proposed action:** {row['action'] or NOT_RECORDED}")
            st.markdown(f"**Context:** {row['context'] or NOT_RECORDED}")
            left, right = st.columns(2)
            with left:
                st.markdown("Expected")
                decision_badge(row["expected_decision"])
            with right:
                st.markdown("Model answer")
                decision_badge(row["actual_decision"])
            if row["explanation"]:
                st.markdown(f"**Model explanation:** {row['explanation']}")
            if row["error"]:
                st.error(f"{row['status']}: {row['error']}")
            with st.expander("Technical details", icon=":material/code:"):
                tokens = row["input_tokens"]
                st.markdown(f"Status: `{row['status']}` · Model ID: "
                            f"{show(row['model_id'])} · Answered by: "
                            f"{show(row['served_by'])} · Provider: "
                            f"{show(row['provider'])} · Latency: "
                            f"{show(row['latency_ms'])} ms · Input tokens: "
                            f"{NOT_RECORDED if pd.isna(tokens) else int(tokens)}")
                if pd.notna(row["reported_confidence"]):
                    st.caption(confidence_note(row["model"],
                                               row["reported_confidence"]))
                if row["warnings"]:
                    st.warning(row["warnings"])
                if row["raw_output"]:
                    st.code(row["raw_output"], language=None)
                raw = raw_lookup(row["model"], row["id"]) if raw_lookup else None
                if raw is not None:
                    st.json(raw, expanded=False)
                else:
                    st.caption(f"Raw response: {NOT_RECORDED}")

    st.subheader("Downloads")
    export = joined.drop(columns=["label_status"])
    with st.container(horizontal=True):
        st.download_button(
            "Case-level results (CSV)", export.to_csv(index=False).encode("utf-8"),
            file_name=f"{meta.get('run_id') or 'results'}.csv", mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Results and run metadata (JSON)",
            json.dumps({"metadata": meta,
                        "results": export.to_dict("records")},
                       indent=2, default=str).encode("utf-8"),
            file_name=f"{meta.get('run_id') or 'results'}.json",
            mime="application/json", icon=":material/download:",
        )
    with st.expander("Technical details: all result rows",
                     icon=":material/table:"):
        st.dataframe(results, hide_index=True, alt="All raw result rows")


# ---------------------------------------------------------------- page

st.title("SystemOne BenchLab", icon=":material/balance:")
st.markdown(":yellow-badge[Lavanya AI Consultancy] Jev — direct decision "
            "model vs Laya · "
            "Agent action governance")

overview_tab, dataset_tab, run_tab, results_tab = st.tabs([
    ":material/home: Overview",
    ":material/table_view: Policy dataset",
    ":material/play_arrow: Run benchmark",
    ":material/analytics: Results & error analysis",
])

# Only runs of the two app models are shown; Jev Router runs stay on disk.
published_runs = [(path, meta) for path, meta in cached_published_runs()
                  if set(meta.get("models", {})) <= set(APP_MODELS)]

with overview_tab:
    st.header("Evaluate whether AI systems follow business policies before "
              "taking action.")
    st.markdown(
        "Before an AI agent issues a refund, deploys code or shares data, a "
        "governance model should check the proposed action against company "
        "policy and return one of three decisions:"
    )
    for column, label in zip(st.columns(3), DECISIONS):
        with column.container(border=True):
            decision_badge(label)
            st.markdown(DECISION_INFO[label]["meaning"])

    st.subheader("Two different kinds of testing")
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**Business policy benchmark** :blue-badge[Main evaluation]")
        st.markdown(
            "Each model receives a written policy, a proposed action and its "
            "context, and must return ALLOW, REQUIRE_APPROVAL or BLOCK. "
            "Answers are compared with a labelled correct decision. Every "
            "system receives identical instructions."
        )
    with right.container(border=True, height="stretch"):
        st.markdown("**General safety check** :gray-badge[Laya only]")
        st.markdown(
            "Laya's built-in safety categories: jailbreak, prompt injection, "
            "sensitive data, harm severity and topic. It does **not** read "
            "or apply a business policy, so these scores are not a measure "
            "of policy compliance."
        )

    st.subheader("The systems compared")
    systems = st.columns(2)
    with systems[0].container(border=True, height="stretch"):
        st.markdown(f"**{model_label('Laya')}**")
        st.markdown("Open-source local checkpoint, run on CPU. Answers the "
                    "decision as a native choice question.")
    with systems[1].container(border=True, height="stretch"):
        st.markdown(f"**{model_label('Jev Direct')}**")
        st.markdown("TypeSafe's own Jev model through its System One API, "
                    "pinned to `jev-1.13.0`. Answers the identical choice "
                    "question; no LLM is involved.")
    st.caption("Earlier runs compared Laya with Jev Router, an OpenRouter "
               "model that routes requests to general LLMs. They are kept in "
               "the GitHub repository and README but are not shown here.")

    headline = next((item for item in published_runs
                     if item[1]["dataset"]["version"] == "v2-heldout"),
                    published_runs[0] if published_runs else None)
    if headline:
        st.subheader("Latest published result")
        results, meta = cached_run(headline[0])
        st.markdown(f":blue-badge[Published result] {run_label(meta)}")
        headline_metrics(results, meta, compact=True)
        st.caption(DISCLAIMER + " See **Results & error analysis** for the "
                   "full breakdown, including every case.")

    st.subheader("How the comparison is kept fair")
    st.markdown(
        f"- **Same task:** every system gets the same instructions and "
        f"decision criteria (prompt `{PROMPT_VERSION}`, fingerprint "
        f"`{prompt_fingerprint()}`).\n"
        "- **Only the case text is sent:** policy, proposed action and "
        "context. Case IDs, expected answers, scenarios and review notes "
        "never reach a model.\n"
        "- **Identical question:** Laya and Jev answer the same native "
        "choice question with exactly three options.\n"
        "- **Held-out data:** the v2 set was written after the prompt was "
        "frozen, and the prompt was not changed afterwards.\n"
        "- **No hidden failures:** timeouts, API errors and unparseable "
        "outputs are recorded and counted as incorrect.\n"
        "- **Ambiguous cases are flagged, not guessed:** cases without one "
        "defensible answer are excluded from scoring.\n"
        "- **What answered is recorded:** the model version that answered "
        "each request is stored.\n"
        "- **Nothing simulated:** every number comes from a real run."
    )
    with st.expander("Limitations", icon=":material/warning:"):
        st.markdown(
            "- Cases are synthetic and hand-written; real policies are "
            "messier.\n"
            "- The sample is small (60 scored cases), so differences of a few "
            "cases are not meaningful.\n"
            "- Laya was used out of the box, without fine-tuning for this "
            "task.\n"
            "- Laya's checkpoint warns that some of its temperatures are "
            "invalid, so its reported confidence is uncalibrated.\n"
            "- Latency is indicative only: Laya runs on CPU here, while Jev "
            "includes the network round trip."
        )

with dataset_tab:
    version = st.segmented_control(
        "Dataset version", list(DATASETS), default="v2-heldout",
        format_func=lambda v: DATASETS[v]["title"], required=True,
        key="dataset_version",
    )
    cases = cached_dataset(version)
    scored = scored_cases(cases)
    flagged = cases[cases["label_status"].ne("confirmed")]
    st.caption(DATASETS[version]["role"])
    counts = st.columns(4)
    counts[0].metric("Cases", len(cases), border=True)
    counts[1].metric("Scored (confirmed label)", len(scored), border=True)
    counts[2].metric("Flagged for review", len(flagged), border=True)
    counts[3].metric("Business areas", cases["business_area"].nunique(),
                     border=True)
    st.caption(f"Dataset fingerprint (SHA-256): `{dataset_sha256(version)}`")

    left, right = st.columns(2)
    with left:
        by_decision = (scored["expected_decision"].value_counts()
                       .reindex(DECISIONS, fill_value=0)
                       .rename_axis("Expected decision")
                       .reset_index(name="Cases"))
        st.bar_chart(by_decision, x="Expected decision", y="Cases",
                     color=MODEL_COLORS["Laya"], sort=False,
                     alt="Number of scored cases for each expected decision")
    with right:
        by_area = (cases["business_area"].value_counts()
                   .rename_axis("Business area").reset_index(name="Cases"))
        st.bar_chart(by_area, x="Business area", y="Cases", horizontal=True,
                     color=MODEL_COLORS["Laya"], sort="-Cases",
                     alt="Number of cases in each business area")

    pick = st.columns(2)
    areas = pick[0].multiselect("Business area",
                                sorted(cases["business_area"].unique()),
                                key="dataset_areas")
    decisions = pick[1].multiselect("Expected decision", DECISIONS,
                                    key="dataset_decisions")
    view = scored
    if areas:
        view = view[view["business_area"].isin(areas)]
    if decisions:
        view = view[view["expected_decision"].isin(decisions)]
    st.dataframe(
        view.drop(columns=["label_status", "review_note"]), hide_index=True,
        alt="Scored policy cases",
        column_config={
            "id": "Case", "business_area": "Area", "scenario": "Scenario",
            "policy": st.column_config.TextColumn("Policy", width="large"),
            "action": st.column_config.TextColumn("Proposed action", width="medium"),
            "context": st.column_config.TextColumn("Context", width="medium"),
            "expected_decision": "Expected decision",
        },
    )
    if len(flagged):
        st.subheader("Flagged for review")
        st.caption("These cases have no single defensible answer under the "
                   "policy as written, so no ground-truth label is assigned "
                   "and they are never scored.")
        st.dataframe(
            flagged[["id", "business_area", "policy", "action", "context",
                     "review_note"]],
            hide_index=True, alt="Cases flagged for label review",
            column_config={
                "id": "Case", "business_area": "Area",
                "policy": st.column_config.TextColumn("Policy", width="large"),
                "action": "Proposed action", "context": "Context",
                "review_note": st.column_config.TextColumn("Why it is flagged",
                                                           width="large"),
            },
        )
    st.download_button(
        "Download this dataset (CSV)",
        cases.to_csv(index=False).encode("utf-8"),
        file_name=f"{version}.csv", mime="text/csv",
        icon=":material/download:",
    )

with run_tab:
    st.header("Business policy benchmark")
    st.caption("Runs ALLOW / REQUIRE_APPROVAL / BLOCK classification on the "
               "labelled cases of a dataset. Flagged cases are skipped.")
    with st.container(border=True):
        left, right = st.columns(2)
        run_version = DATASET_BY_TITLE[left.selectbox(
            "Dataset", list(DATASET_BY_TITLE), key="run_dataset")]
        run_models = [MODEL_BY_LABEL[label] for label in right.pills(
            "Models", list(RUNNABLE_BY_LABEL), selection_mode="multi",
            default=[model_label(m) for m in DEFAULT_RUN_MODELS],
            key="run_models")]
        for model_name in run_models:
            credential_warning(model_name)
        with st.expander("Run settings", icon=":material/tune:"):
            st.table(pd.DataFrame([
                ("Prompt version", f"{PROMPT_VERSION} "
                                   f"(fingerprint {prompt_fingerprint()})"),
                ("Sent to every model", "Policy, proposed action and context "
                                        "only"),
                (model_label("Laya"),
                 f"laya {show(laya_version())} · local CPU · language en · "
                 "checkpoint chosen automatically"),
                (model_label("Jev Direct"),
                 f"{SETTINGS['jev_direct_model']} · TypeSafe System One API · "
                 f"native choice question · timeout {TYPESAFE_TIMEOUT_S} s · "
                 f"up to {TYPESAFE_MAX_ATTEMPTS - 1} retries on HTTP 429/529 · "
                 "$0.042 per million input tokens (documented)"),
            ], columns=["Setting", "Value"]), hide_index=True,
                alt="Settings used for a live run")
            st.caption("Laya's first request downloads and loads its "
                       "checkpoint (about 800 MB), which can take a few "
                       "minutes on a fresh server.")
        start = st.button("Run benchmark", type="primary",
                          icon=":material/play_arrow:", disabled=not run_models)

    if start:
        run_cases_df = scored_cases(cached_dataset(run_version))
        total = len(cached_dataset(run_version))
        to_run = [m for m in run_models
                  if not missing_credential(m, SETTINGS)]
        started = datetime.now(timezone.utc)
        run_id = f"{run_version}_{started:%Y%m%dT%H%M%SZ}"
        rows, raws = [], {}
        with st.status(f"Running {len(run_cases_df)} cases",
                       expanded=True) as status:
            for model_name in to_run:
                name = model_label(model_name)
                bar = st.progress(0.0, text=f"{name}: starting")

                def on_row(index, row, bar=bar, name=name):
                    bar.progress(
                        (index + 1) / len(run_cases_df),
                        text=f"{name}: case {index + 1} of "
                             f"{len(run_cases_df)} · {row['id']} · "
                             f"{row['status']}",
                    )

                model_rows, model_raws = run_cases(
                    model_name, run_cases_df, on_row=on_row,
                    settings=SETTINGS,
                )
                rows += model_rows
                raws.update(model_raws)
            live_results = results_frame(rows)
            live_meta = build_metadata(
                run_id=run_id, source="live", dataset_version=run_version,
                results=live_results,
                started_utc=started.isoformat(timespec="seconds"),
                finished_utc=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                cases_run=len(run_cases_df),
                flagged_skipped=total - len(run_cases_df),
                settings=SETTINGS,
            )
            save_run(ROOT / "results" / run_id, live_results, live_meta, raws)
            status.update(label="Run complete", state="complete",
                          expanded=False)
        st.session_state["live_run"] = {
            "results": normalize_results(live_results), "metadata": live_meta,
            "raws": raws,
        }
        problems = int(live_results["status"].ne("ok").sum())
        st.success("Run complete. Open **Results & error analysis** and "
                   "choose the live run.", icon=":material/check_circle:")
        if problems:
            st.warning(f"{problems} requests failed or returned invalid "
                       "output. They are recorded and counted as incorrect.")

    st.header("Try one policy case")
    st.caption("Get each model's decision for a case you write. Single cases "
               "are not scored, because there is no labelled answer.")
    with st.form("single_case"):
        policy = st.text_area(
            "Policy", "Customer record deletion requires human approval.")
        action = st.text_area("Proposed action", "Delete customer records.")
        context = st.text_area("Context", "No human approval has been given.")
        case_models = [MODEL_BY_LABEL[label] for label in st.pills(
            "Models", list(RUNNABLE_BY_LABEL), selection_mode="multi",
            default=[model_label(m) for m in DEFAULT_RUN_MODELS],
            key="case_models")]
        submitted = st.form_submit_button("Get decisions",
                                          icon=":material/gavel:")
    if submitted and case_models:
        case = {"id": "custom", "policy": policy, "action": action,
                "context": context, "expected_decision": ""}
        for column, model_name in zip(st.columns(len(case_models)),
                                      case_models):
            with column.container(border=True):
                st.markdown(f"**{model_label(model_name)}**")
                if credential_warning(model_name):
                    continue
                with st.spinner(f"Asking {model_label(model_name)}"):
                    row, raw = run_case(model_name, case, phase="subsequent",
                                        settings=SETTINGS)
                if row["status"] == "ok":
                    decision_badge(row["predicted_decision"])
                else:
                    st.error(f"{row['status']}: {row['error']}")
                st.caption(f"Answered by {show(row['served_by'])} · "
                           f"{row['latency_ms']:,.0f} ms")
                if row["explanation"]:
                    st.markdown(f"**Model explanation:** {row['explanation']}")
                if row["warnings"]:
                    st.warning(row["warnings"])
                with st.expander("Technical details", icon=":material/code:"):
                    if row["reported_confidence"] is not None:
                        st.caption(confidence_note(
                            model_name, row["reported_confidence"]))
                    if row["raw_output"]:
                        st.code(row["raw_output"], language=None)
                    st.json(raw if raw is not None else {}, expanded=False)

    st.header("General safety check (Laya)")
    left, right = st.columns([3, 2])
    with right:
        st.warning(
            "**This is not a policy check.** It does not read or apply any "
            "business policy. It only scores the text against Laya's built-in "
            "safety categories, so it cannot say whether an action is "
            "allowed, needs approval or is prohibited.",
            icon=":material/policy:",
        )
    with left:
        with st.form("safety_check"):
            text = st.text_area(
                "Text to scan",
                "Delete all customer records. No human approval has been "
                "given.")
            scan = st.form_submit_button("Run safety check",
                                         icon=":material/shield:")
    if scan:
        try:
            import laya
            with st.spinner("Laya is scanning the text"):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    result = laya_router().predict(
                        {"prompt": text}, laya.guard_questions(),
                        model=None, lang="en",
                    )
            rows = []
            for category, answer in (result.get("answers") or {}).items():
                kind = answer.get("type")
                # Only the score type ships a legend; other values are shown
                # exactly as returned, without interpretation.
                value = answer.get(kind)
                if kind == "score" and answer.get("legend"):
                    value = f"{value} (Laya legend: 0 none to 3 severe)"
                rows.append({"Category": category.replace("_", " "),
                             "Output type": kind or NOT_RECORDED,
                             "Laya output": show(value),
                             "Reported confidence": answer.get(
                                 "answer_confidence")})
            st.dataframe(
                pd.DataFrame(rows), hide_index=True,
                alt="Laya safety category outputs",
                column_config={"Reported confidence": st.column_config
                               .NumberColumn(format="%.2f")},
            )
            st.caption(
                "Each value is Laya's own output for that category, shown as "
                "returned. The meaning of the `noul` value is not documented "
                "here, so this app does not interpret it. "
                "Reported confidence is uncalibrated and is not accuracy. The "
                "categories are shown separately and are not combined into "
                "any compliance score."
            )
            for item in captured:
                st.warning(str(item.message), icon=":material/thermostat:")
            with st.expander("Technical details", icon=":material/code:"):
                st.caption(
                    "Fields such as `action.act_probability` are internal "
                    "Laya outputs that this app does not interpret. They are "
                    "never treated as permission to execute an action."
                )
                st.json(result, expanded=False)
        except Exception as exc:
            st.error(f"Laya safety check failed: {exc}")

with results_tab:
    # Options are display labels mapped back to a source key.
    sources = {}
    if "live_run" in st.session_state:
        sources["Live run from this session (not published)"] = "live"
    sources.update({f"Published · {run_label(meta)}": path
                    for path, meta in published_runs})
    sources["Upload a results CSV"] = "upload"
    choice = sources[st.selectbox("Results to show", list(sources),
                                  key="results_source")]

    if choice == "live":
        live = st.session_state["live_run"]
        render_results(live["results"], live["metadata"],
                       lambda model, case_id: live["raws"].get(
                           f"{RAW_PREFIXES[model][0]}_{case_id}"))
    elif choice == "upload":
        uploaded = st.file_uploader("Results CSV", type=["csv"])
        st.caption("Required columns: id, model, expected_decision, "
                   "predicted_decision, status. Other columns are optional; "
                   "anything missing is shown as Not recorded.")
        refers_to = DATASET_BY_TITLE.get(st.selectbox(
            "Dataset these results refer to (used to show case text)",
            ["Unknown"] + list(DATASET_BY_TITLE), key="upload_dataset"))
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded, dtype={"id": str},
                                    keep_default_na=False)
                missing = {"id", "model", "expected_decision",
                           "predicted_decision", "status"} - set(frame.columns)
                if missing:
                    st.error("Missing columns: " + ", ".join(sorted(missing)))
                else:
                    render_results(
                        normalize_results(frame),
                        {"source": "uploaded",
                         "dataset": {"version": refers_to}},
                        None,
                    )
            except Exception as exc:
                st.error(f"Cannot read results: {exc}")
    else:
        results, meta = cached_run(choice)
        render_results(results, meta,
                       lambda model, case_id, path=choice: load_raw(
                           Path(path), model, case_id))
