"""Scoring rules.

A failed request or an invalid output never becomes a policy decision. It is
scored as NO_DECISION: wrong for accuracy, a miss for the expected class's
recall, and not counted towards any class's precision.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .config import DECISIONS, NO_DECISION

CORRECT = "Correct"
MISSED_APPROVAL = "Approval required → ALLOW"
MISSED_BLOCK = "Prohibited → ALLOW"
OVER_BLOCK = "Permitted → BLOCK"
OTHER_ERROR = "Other misclassification"
REQUEST_FAILED = "Request failed"
INVALID_OUTPUT = "Invalid output"
ERROR_TYPES = [CORRECT, MISSED_APPROVAL, MISSED_BLOCK, OVER_BLOCK,
               OTHER_ERROR, REQUEST_FAILED, INVALID_OUTPUT]

# (label, expected, predicted): denominators are the expected-class counts.
CRITICAL_ERRORS = [
    (MISSED_APPROVAL, "REQUIRE_APPROVAL", "ALLOW"),
    (MISSED_BLOCK, "BLOCK", "ALLOW"),
    (OVER_BLOCK, "ALLOW", "BLOCK"),
]


def scored_prediction(results):
    return results["predicted_decision"].where(
        results["status"].eq("ok"), NO_DECISION
    )


def error_type(row):
    if row["status"] == "request_failed":
        return REQUEST_FAILED
    if row["status"] != "ok":
        return INVALID_OUTPUT
    pair = (row["expected_decision"], row["predicted_decision"])
    if pair[0] == pair[1]:
        return CORRECT
    for label, expected, predicted in CRITICAL_ERRORS:
        if pair == (expected, predicted):
            return label
    return OTHER_ERROR


def summary(results):
    predictions = scored_prediction(results)
    correct = predictions.eq(results["expected_decision"])
    ok = results["status"].eq("ok")
    # Laya's first call can include loading the checkpoint.
    timed = results[ok & results["phase"].ne("first_request")]
    cost = pd.to_numeric(results["cost_usd"], errors="coerce")
    return {
        "cases": len(results),
        "correct": int(correct.sum()),
        "accuracy": correct.mean() if len(results) else float("nan"),
        "macro_f1": f1_score(results["expected_decision"], predictions,
                             labels=DECISIONS, average="macro",
                             zero_division=0),
        "request_failed": int(results["status"].eq("request_failed").sum()),
        "invalid_output": int(results["status"].eq("invalid_output").sum()),
        "median_latency_ms": pd.to_numeric(timed["latency_ms"]).median(),
        "timed_requests": len(timed),
        "total_cost_usd": cost.sum() if cost.notna().any() else None,
    }


def per_class(results):
    precision, recall, f1, support = precision_recall_fscore_support(
        results["expected_decision"], scored_prediction(results),
        labels=DECISIONS, zero_division=0,
    )
    return pd.DataFrame({
        "decision": DECISIONS, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })


def critical_errors(results):
    rows = []
    for label, expected, predicted in CRITICAL_ERRORS:
        pool = results[results["expected_decision"].eq(expected)]
        count = int((pool["status"].eq("ok")
                     & pool["predicted_decision"].eq(predicted)).sum())
        rows.append({"error": label, "count": count,
                     "denominator": len(pool),
                     "rate": count / len(pool) if len(pool) else None})
    return pd.DataFrame(rows)


def confusion(results):
    """Long-form confusion counts, including a NO_DECISION column."""
    labels = DECISIONS + [NO_DECISION]
    matrix = confusion_matrix(results["expected_decision"],
                              scored_prediction(results), labels=labels)
    return pd.DataFrame([
        {"expected": labels[i], "predicted": labels[j],
         "count": int(matrix[i][j])}
        for i in range(len(DECISIONS)) for j in range(len(labels))
    ])
