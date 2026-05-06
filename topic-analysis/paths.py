"""
paths.py — Data paths for the topic analysis.

LDA outputs live outside the repo (large files).  Override with env vars:
  REKENKAMER_LDA_DIR      300-topic model (1972-2022 chunked Handelingen)
  REKENKAMER_LDA_DIR_500  500-topic model (1945-2022 chunked Handelingen)
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

RESULTS_DIR: Path = _REPO / "results"
RESOURCES_DIR: Path = _REPO / "resources"

# ── LDA model directories (large, not in git) ─────────────────────────────────

# 300-topic model used in references / partial-correlation analysis
LDA_DIR: Path = Path(
    os.environ.get(
        "REKENKAMER_LDA_DIR",
        str(Path.home() / "Downloads" / "lda" / "data-1972-2022-chunked"),
    )
)

# 500-topic model used in control-topic / Verantwoordingsdag analysis
LDA_DIR_500: Path = Path(
    os.environ.get(
        "REKENKAMER_LDA_DIR_500",
        str(Path.home() / "Downloads" / "lda" / "lda" / "data-1945-2022-chunked"),
    )
)

# ── Input data ─────────────────────────────────────────────────────────────────

# Sentence-match output: speeches that reference Rekenkamer reports
REFERENCES_CSV: Path = RESULTS_DIR / "references.csv"

# Noun totals per date for relative-frequency normalisation
NOUN_TOTAL_CSV: Path = RESOURCES_DIR / "noun-total-proc.csv"

# Cabinet composition for coalition/opposition classification
CABINETS_CSV: Path = RESOURCES_DIR / "cabinets.csv"
