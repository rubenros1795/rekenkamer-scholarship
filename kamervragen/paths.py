"""
paths.py — Canonical data paths for the kamervragen analysis.

Data resolution order:
  1. results/  at the repo root        (default, committed CSVs)
  2. REKENKAMER_DATA_DIR env var       (point to any external location)
  3. ~/Documents/Data/rekenkamer/      (fallback for local large-file store)

Raw source data (large xlsx / json) lives in ~/Documents/Data/rekenkamer/
and is never committed to git.

Annotated CSV naming convention
--------------------------------
  kamervragen_{source}_{schema}_{version}.csv

  source  : llm | manual
  schema  : q (questions, schema A) | a (answers, schema B)
  version : v0.6, v0.7, v0.2, v0.3, …  (guideline version used to annotate)

When a new annotation run is complete, update the ANNOTATED_Q_CSV /
ANNOTATED_A_CSV / MANUAL_Q_CSV / MANUAL_A_CSV aliases at the bottom.
All analysis scripts import only the aliases and require no further changes.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Results: live in repo results/ (committed, small CSVs)
RESULTS_DIR: Path = _REPO / "results"

# Raw source data: large files stored outside the repo
DATA_DIR: Path = Path(
    os.environ.get("REKENKAMER_DATA_DIR", str(Path.home() / "Documents" / "Data" / "rekenkamer"))
)

# ── Non-annotated output files ─────────────────────────────────────────────────
STRUCTURED_CSV   = RESULTS_DIR / "kamervragen-structured.csv"
STRUCTURED_SPLIT = RESULTS_DIR / "kamervragen-structured-split.csv"
RAW_CSV          = RESULTS_DIR / "kamervragen-raw.csv"
METADATA_CSV     = RESULTS_DIR / "kamervragen-metadata.csv"

# ── Raw source data (large, not in git) ───────────────────────────────────────
REKENKAMER_XLSX  = DATA_DIR / "rekenkamer.xlsx"
META_REPORTS     = DATA_DIR / "meta-reports.json"

# ── Versioned annotation files — LLM questions (schema A) ─────────────────────
ANNOTATED_Q_V06  = RESULTS_DIR / "kamervragen_annotated_q_v0.6.csv"
ANNOTATED_Q_V07  = RESULTS_DIR / "kamervragen_annotated_q_v0.7.csv"

# ── Versioned annotation files — LLM answers (schema B) ───────────────────────
ANNOTATED_A_V02  = RESULTS_DIR / "kamervragen_annotated_a_v0.2.csv"
ANNOTATED_A_V03  = RESULTS_DIR / "kamervragen_annotated_a_v0.3.csv"

# ── Versioned annotation files — manual questions (schema A) ──────────────────
MANUAL_Q_V06     = RESULTS_DIR / "kamervragen_manual_q_v0.6.csv"
MANUAL_Q_V07     = RESULTS_DIR / "kamervragen_manual_q_v0.7.csv"

# ── Versioned annotation files — manual answers (schema B) ────────────────────
MANUAL_A_V02     = RESULTS_DIR / "kamervragen_manual_a_v0.2.csv"
MANUAL_A_V03     = RESULTS_DIR / "kamervragen_manual_a_v0.3.csv"

# ── Current aliases — update when a new annotation run completes ───────────────
ANNOTATED_Q_CSV  = ANNOTATED_Q_V07
ANNOTATED_A_CSV  = ANNOTATED_A_V03
MANUAL_Q_CSV     = MANUAL_Q_V07
MANUAL_A_CSV     = MANUAL_A_V03

# ── Annotation state files live in kamervragen/state/ ─────────────────────────
STATE_DIR: Path = _HERE / "state"

# ── Output directories ─────────────────────────────────────────────────────────
REPORTS_DIR: Path = _HERE / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
