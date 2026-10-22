"""Dataset loading and fingerprinting."""
import hashlib

import pandas as pd

from .config import DATASETS, ROOT

COLUMNS = [
    "id", "business_area", "scenario", "policy", "action", "context",
    "expected_decision", "label_status", "review_note",
]


def _read(path):
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")


def load_dataset(version):
    spec = DATASETS[version]
    cases = _read(spec["file"])
    if "annotations" in spec:
        # v1 stays byte-for-byte unchanged; tags live in a side file.
        cases = cases.merge(_read(spec["annotations"]), on="id", how="left")
    for column in COLUMNS:
        if column not in cases:
            cases[column] = "confirmed" if column == "label_status" else ""
    return cases[COLUMNS]


def scored_cases(cases):
    """Cases with a confirmed ground-truth label."""
    return cases[cases["label_status"].eq("confirmed")]


def dataset_sha256(version):
    # Normalise line endings and BOM so the hash matches on every OS.
    raw = (ROOT / DATASETS[version]["file"]).read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
