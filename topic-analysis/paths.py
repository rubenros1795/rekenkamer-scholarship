"""
paths.py — Data paths for the topic analysis.

LDA output lives outside the repo (large files).  Set REKENKAMER_LDA_DIR
or drop the folder at the default location.  Everything else resolves from
the repo layout automatically.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

RESULTS_DIR: Path = _REPO / "results"

# LDA output directory — large, not in git
LDA_DIR: Path = Path(
    os.environ.get(
        "REKENKAMER_LDA_DIR",
        str(Path.home() / "Downloads" / "lda" / "data-1972-2022-chunked"),
    )
)

# Hits (sentence-match output)
HITS_CSV: Path = RESULTS_DIR / "hits.csv"

# Noun totals for relative-frequency normalisation
NOUN_TOTAL_CSV: Path = _HERE / "proc-hits.csv"

# Cabinet reference table
CABINETS_CSV: Path = _REPO / "resources" / "cabinets.csv"
