import pandas as pd
import pytest

from benchlab import metrics
from benchlab.data import load_dataset, scored_cases
from benchlab.results import normalize_results


def frame(rows):
    return normalize_results(pd.DataFrame(
        rows, columns=["id", "model", "expected_decision",
                       "predicted_decision", "status"]))


def test_failures_count_as_incorrect_but_not_as_a_decision():
    results = frame([
        ("1", "M", "ALLOW", "ALLOW", "ok"),
        ("2", "M", "BLOCK", "", "request_failed"),
        ("3", "M", "REQUIRE_APPROVAL", "garbage", "invalid_output"),
        ("4", "M", "REQUIRE_APPROVAL", "ALLOW", "ok"),
    ])
    summary = metrics.summary(results)
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["request_failed"] == 1
    assert summary["invalid_output"] == 1

    per_class = metrics.per_class(results).set_index("decision")
    # The invalid "garbage" output must not be scored as any decision.
    assert per_class.loc["ALLOW", "precision"] == pytest.approx(0.5)
    assert per_class.loc["BLOCK", "recall"] == 0
    assert per_class.loc["REQUIRE_APPROVAL", "support"] == 2


def test_error_types_and_denominators():
    results = frame([
        ("1", "M", "REQUIRE_APPROVAL", "ALLOW", "ok"),
        ("2", "M", "REQUIRE_APPROVAL", "REQUIRE_APPROVAL", "ok"),
        ("3", "M", "BLOCK", "ALLOW", "ok"),
        ("4", "M", "ALLOW", "BLOCK", "ok"),
        ("5", "M", "BLOCK", "REQUIRE_APPROVAL", "ok"),
    ])
    assert list(results.apply(metrics.error_type, axis=1)) == [
        metrics.MISSED_APPROVAL, metrics.CORRECT, metrics.MISSED_BLOCK,
        metrics.OVER_BLOCK, metrics.OTHER_ERROR,
    ]
    critical = metrics.critical_errors(results).set_index("error")
    assert critical.loc[metrics.MISSED_APPROVAL, "count"] == 1
    assert critical.loc[metrics.MISSED_APPROVAL, "denominator"] == 2
    assert critical.loc[metrics.MISSED_BLOCK, "denominator"] == 2


def test_confusion_has_no_decision_column():
    results = frame([("1", "M", "ALLOW", "", "request_failed")])
    cell = metrics.confusion(results).query(
        "expected == 'ALLOW' and predicted == 'NO_DECISION'")
    assert int(cell["count"].iloc[0]) == 1


def test_legacy_error_status_is_a_request_failure():
    results = frame([("1", "M", "ALLOW", "", "error")])
    assert results["status"].iloc[0] == "request_failed"


@pytest.mark.parametrize("version", ["v1-baseline", "v2-heldout"])
def test_datasets_have_unique_ids_and_valid_labels(version):
    cases = load_dataset(version)
    assert cases["id"].is_unique
    scored = scored_cases(cases)
    assert set(scored["expected_decision"]) <= set(metrics.DECISIONS)
    flagged = cases[cases["label_status"].ne("confirmed")]
    assert flagged["expected_decision"].eq("").all()
    assert flagged["review_note"].ne("").all()
