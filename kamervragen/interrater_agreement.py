#!/usr/bin/env python3
"""
interrater_agreement.py — Inter-annotator agreement between manual and LLM annotations.

Supports both question annotations (schema A: FEIT/CAU/OOR/ADV) and
answer annotations (schema B: FEIT/CAU/OOR/ADV/DEFL).

Handles old schema labels via --map-old flag.

Usage
-----
    # Questions: manual vs LLM (current schema)
    python interrater_agreement.py questions

    # Questions: manual vs LLM with old→new label mapping
    python interrater_agreement.py questions --map-old

    # Answers: manual vs LLM
    python interrater_agreement.py answers

    # Explicit file paths
    python interrater_agreement.py --manual path/to/manual.csv --manual-col label \
                        --llm path/to/llm.csv --llm-col llm_label
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from paths import (
    MANUAL_Q_CSV, ANNOTATED_Q_CSV,
    MANUAL_A_CSV, ANNOTATED_A_CSV,
    DATA_DIR,
)

# ── Default file paths ─────────────────────────────────────────────────────────

DEFAULTS = {
    "questions": {
        "manual":     MANUAL_Q_CSV,
        "manual_col": "label",
        "llm":        ANNOTATED_Q_CSV,
        "llm_col":    "llm_label",
    },
    "answers": {
        "manual":     MANUAL_A_CSV,
        "manual_col": "label",
        "llm":        ANNOTATED_A_CSV,
        "llm_col":    "llm_label",
    },
}

# ── Label mappings (old schema → current schema) ───────────────────────────────

# Questions: old (v0.1/v0.2) → current (v0.4/v0.5)
OLD_TO_NEW_QUESTIONS: dict[str, str] = {
    "INF-CIJ": "FEIT",
    "INF-OOR": "OOR",   # ambiguous: could be FEIT/CAU/OOR — OOR is closest
    "INF":     "FEIT",
    "LEG":     "ADV",
    "POL":     "OOR",
    "VEC":     "OOR",   # political accountability → closest cognitive equivalent
    "AGN":     "AGN",
}

# ── Join key ──────────────────────────────────────────────────────────────────

JOIN_COLS = ["src_id", "vraag_nr"]


def _make_key(df: pd.DataFrame) -> pd.Series:
    return df["src_id"].astype(str) + "||" + df["vraag_nr"].astype(str)


# ── Metrics ───────────────────────────────────────────────────────────────────


def percent_agreement(y1: list, y2: list) -> float:
    if not y1:
        return float("nan")
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


def cohen_kappa(y1: list, y2: list, labels: list[str]) -> float:
    """Cohen's kappa for nominal categories."""
    n = len(y1)
    if n == 0:
        return float("nan")

    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion matrix
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        i = label_idx.get(a, -1)
        j = label_idx.get(b, -1)
        if i >= 0 and j >= 0:
            cm[i][j] += 1

    observed = sum(cm[i][i] for i in range(k)) / n
    row_sums = [sum(cm[i]) / n for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) / n for j in range(k)]
    expected = sum(row_sums[i] * col_sums[i] for i in range(k))

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion_matrix_str(y1: list, y2: list, labels: list[str],
                          rater1: str = "manual", rater2: str = "LLM") -> str:
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        i = label_idx.get(a, -1)
        j = label_idx.get(b, -1)
        if i >= 0 and j >= 0:
            cm[i][j] += 1

    # Format
    col_w  = max(len(l) for l in labels) + 2
    row_w  = max(len(l) for l in labels) + 2
    header = f"{'':>{row_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    lines  = [f"  {rater1} (rows) vs {rater2} (cols)", header]
    for i, lbl in enumerate(labels):
        row = f"{lbl:>{row_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(k))
        lines.append(row)
    return "\n".join(lines)


# ── Weighted kappa (ordinal labels) ──────────────────────────────────────────


def weighted_kappa(y1: list, y2: list, labels: list[str]) -> float:
    """Linear-weighted Cohen's kappa for ordinal label ordering."""
    n = len(y1)
    if n == 0:
        return float("nan")
    k = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        i = label_idx.get(a, -1)
        j = label_idx.get(b, -1)
        if i >= 0 and j >= 0:
            cm[i][j] += 1

    row_sums = [sum(cm[i]) for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) for j in range(k)]

    num_obs = num_exp = 0.0
    for i in range(k):
        for j in range(k):
            w = abs(i - j) / (k - 1) if k > 1 else 0
            num_obs += w * cm[i][j]
            num_exp += w * row_sums[i] * col_sums[j] / n

    return 1.0 - (num_obs / num_exp) if num_exp else float("nan")


# ── Main logic ────────────────────────────────────────────────────────────────


def run(
    manual_path: Path,
    manual_col: str,
    llm_path: Path,
    llm_col: str,
    map_old: bool,
    schema: str,
) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    if not manual_path.exists():
        sys.exit(f"Manual file not found: {manual_path}")
    if not llm_path.exists():
        sys.exit(f"LLM file not found: {llm_path}")

    manual = pd.read_csv(manual_path, sep="\t")
    llm    = pd.read_csv(llm_path, sep="\t")

    print(f"Manual file : {manual_path.name}  ({len(manual)} rows, col='{manual_col}')")
    print(f"LLM file    : {llm_path.name}  ({len(llm)} rows, col='{llm_col}')")

    # ── Filter to annotated rows ───────────────────────────────────────────────
    manual = manual[manual[manual_col].notna() & (manual[manual_col].astype(str).str.strip() != "")]
    llm    = llm[llm[llm_col].notna() & (llm[llm_col].astype(str).str.strip() != "")]

    # ── Join ─────────────────────────────────────────────────────────────────
    available_join = [c for c in JOIN_COLS if c in manual.columns and c in llm.columns]
    if available_join:
        manual["_key"] = _make_key(manual) if set(JOIN_COLS).issubset(manual.columns) else manual[available_join[0]].astype(str)
        llm["_key"]    = _make_key(llm)    if set(JOIN_COLS).issubset(llm.columns)    else llm[available_join[0]].astype(str)
        merged = manual[["_key", manual_col]].merge(
            llm[["_key", llm_col]], on="_key", how="inner"
        )
    else:
        # Fall back to positional join
        print("Warning: join columns not found, using positional join")
        n = min(len(manual), len(llm))
        merged = pd.DataFrame({
            manual_col: manual[manual_col].iloc[:n].values,
            llm_col:    llm[llm_col].iloc[:n].values,
        })

    print(f"\nOverlapping items: {len(merged)}")
    if len(merged) == 0:
        sys.exit("No overlapping items found — cannot compute agreement.")

    # ── Optional label mapping ────────────────────────────────────────────────
    if map_old:
        mapping = OLD_TO_NEW_QUESTIONS
        before = merged[llm_col].value_counts().to_dict()
        merged[llm_col] = merged[llm_col].map(lambda x: mapping.get(x, x))
        after = merged[llm_col].value_counts().to_dict()
        print(f"\nApplied old→new label mapping: {mapping}")
        print(f"  Before: {before}")
        print(f"  After:  {after}")

    # ── Determine labels for metrics ──────────────────────────────────────────
    if schema == "questions":
        ordered_labels = ["FEIT", "CAU", "OOR", "ADV", "AGN"]
    else:
        ordered_labels = ["FEIT", "OOR", "ADV", "DEFL"]

    all_seen = sorted(set(merged[manual_col].tolist() + merged[llm_col].tolist()))
    known    = [l for l in ordered_labels if l in all_seen]
    unknown  = [l for l in all_seen if l not in ordered_labels and l not in ("?", "")]
    labels_for_metrics = known + [l for l in unknown if l]

    if unknown:
        print(f"\nUnknown labels (kept as-is): {unknown}")

    # ── Filter out ? and empty ────────────────────────────────────────────────
    valid = merged[
        ~merged[manual_col].isin(["?", "", "nan"]) &
        ~merged[llm_col].isin(["?", "", "nan"])
    ]
    print(f"Items after excluding '?' / empty: {len(valid)}")

    if len(valid) == 0:
        sys.exit("No valid items to compare.")

    y_manual = valid[manual_col].tolist()
    y_llm    = valid[llm_col].tolist()

    # ── Metrics ───────────────────────────────────────────────────────────────
    pa    = percent_agreement(y_manual, y_llm)
    kappa = cohen_kappa(y_manual, y_llm, labels_for_metrics)
    wk    = weighted_kappa(y_manual, y_llm, labels_for_metrics)

    print(f"\n{'─'*50}")
    print(f"  Percent agreement : {pa:.3f}  ({pa*100:.1f}%)")
    print(f"  Cohen's κ         : {kappa:.3f}")
    print(f"  Weighted κ (lin.) : {wk:.3f}  (ordinal, labels ordered {labels_for_metrics})")
    print(f"{'─'*50}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\nConfusion matrix (n={len(valid)}):")
    print(confusion_matrix_str(y_manual, y_llm, labels_for_metrics))

    # ── Per-label breakdown ───────────────────────────────────────────────────
    print("\nPer-label agreement:")
    header = f"  {'Label':<8} {'Manual_n':>8} {'LLM_n':>8} {'P(agree|label)':>16}"
    print(header)
    for lbl in labels_for_metrics:
        m_rows = [(a, b) for a, b in zip(y_manual, y_llm) if a == lbl]
        l_rows = [(a, b) for a, b in zip(y_manual, y_llm) if b == lbl]
        agree  = sum(1 for a, b in m_rows if a == b)
        pa_lbl = agree / len(m_rows) if m_rows else float("nan")
        print(f"  {lbl:<8} {len(m_rows):>8} {len(l_rows):>8} {pa_lbl:>16.3f}")

    # ── Disagreement sample ───────────────────────────────────────────────────
    disagree = valid[valid[manual_col] != valid[llm_col]]
    if len(disagree) > 0:
        print(f"\nDisagreements: {len(disagree)} items")
        sample = disagree.head(10)
        for _, row in sample.iterrows():
            key = row.get("_key", "–")
            print(f"  [{row[manual_col]} vs {row[llm_col]}]  key={key}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-annotator agreement: manual vs LLM")
    sub = parser.add_subparsers(dest="schema")

    for name in ("questions", "answers"):
        p = sub.add_parser(name, help=f"Use default paths for {name}")
        p.add_argument("--map-old", action="store_true",
                       help="Map old schema labels (INF-CIJ/INF-OOR/LEG/…) to current schema")
        p.add_argument("--manual",     default=None)
        p.add_argument("--manual-col", default=None)
        p.add_argument("--llm",        default=None)
        p.add_argument("--llm-col",    default=None)

    # Also allow fully explicit mode (no subcommand)
    parser.add_argument("--manual",     default=None, help="Path to manual annotations CSV")
    parser.add_argument("--manual-col", default="label")
    parser.add_argument("--llm",        default=None, help="Path to LLM annotations CSV")
    parser.add_argument("--llm-col",    default="llm_label")
    parser.add_argument("--map-old",    action="store_true")

    args = parser.parse_args()

    schema = args.schema or "questions"
    defaults = DEFAULTS[schema]

    manual_path = Path(args.manual)     if args.manual     else defaults["manual"]
    manual_col  = args.manual_col       if args.manual_col else defaults["manual_col"]
    llm_path    = Path(args.llm)        if args.llm        else defaults["llm"]
    llm_col     = args.llm_col          if args.llm_col    else defaults["llm_col"]
    map_old     = args.map_old

    run(manual_path, manual_col, llm_path, llm_col, map_old, schema)


if __name__ == "__main__":
    main()
