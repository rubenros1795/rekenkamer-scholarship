#!/usr/bin/env python3
"""
match_reports.py — Match kamervragen rapport_titel to AR report texts and
compute report-level metadata features.

Pipeline:
  1. Load unique rapport_titel values from kamervragen_annotated_q_v0.6.csv.
  2. Clean/extract the embedded report title from 'Beantwoording...' strings.
  3. Fuzzy-match against titles in rekenkamer.xlsx (report texts) and
     meta-reports.json (report catalog with policy-domain categories).
  4. For matched Excel reports, compute linguistic metadata from doc_content.
  5. Merge category info from meta-reports.json.
  6. Save to data/report_metadata.csv.

Output columns:
  rapport_titel          original rapport_titel from Q data
  search_title           cleaned title used for matching
  matched_title          best-matched report title
  match_score            fuzzy match score (0-100)
  match_source           'excel' | 'meta' | None
  report_type            'verantwoordingsonderzoek' | 'thematisch' | 'overig'
  category               ministry/domain code from meta-reports (e.g. DEF, FIN)
  rep_nwoorden           word count of report text
  rep_ttr                type-token ratio of report text
  rep_hapax_ratio        hapax legomena ratio
  rep_gem_zinslengte     average sentence length
  rep_ratio_nominalisaties nominalisations per 100 words
  rep_formele_taal_ratio  ratio of words ≥6 characters
  rep_ratio_afkortingen  abbreviations per 100 words

Usage:
    python match_reports.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

from paths import ANNOTATED_Q_CSV, DATA_DIR, REPORTS_DIR

# ── External data paths ────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent   # Rekenkamer/

EXCEL_PATH      = _REPO / "_DATA" / "rekenkamer.xlsx"
META_JSON_PATH  = _REPO / "_DATA" / "meta-reports.json"
OUTPUT_CSV      = DATA_DIR / "report_metadata.csv"

MATCH_THRESHOLD_HIGH = 80   # confident match
MATCH_THRESHOLD_LOW  = 65   # borderline, flagged

# ── Text helpers (same as extract_metadata.py) ────────────────────────────────

import re as _re

_RE_SENTENCE        = _re.compile(r'[.!?]+(?:\s|$)')
_RE_NOM             = _re.compile(r'\b\w{5,}(?:ing|tie|iteit|heid|schap|isme|ering)\b', _re.IGNORECASE)
_RE_ACRONYM         = _re.compile(r'\b[A-Z]{2,6}\b')
_RE_ABBREV_DOT      = _re.compile(r'\b[a-zA-Z]{1,4}\.')


def _tokens(text: str) -> list[str]:
    return _re.findall(
        r"\b[a-zA-ZàáâäãåæçèéêëìíîïðñòóôõöùúûüýÿÀ-ÖØ-öø-ÿ']+\b",
        str(text).lower()
    )


def _ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def _hapax_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    return sum(1 for v in freq.values() if v == 1) / len(freq)


def _avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in _RE_SENTENCE.split(text)]
    lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
    return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


def _nominalisaties(text: str) -> int:
    return len(_RE_NOM.findall(text))


def _abbreviations(text: str) -> int:
    return len(_RE_ACRONYM.findall(text)) + len(_RE_ABBREV_DOT.findall(text))


def _formele_taal_ratio(tokens: list[str]) -> float:
    return sum(1 for t in tokens if len(t) >= 6) / len(tokens) if tokens else 0.0


def compute_text_features(text: str) -> dict:
    """Compute linguistic features from a report's full text."""
    text = str(text or "")
    tokens = _tokens(text)
    n = len(tokens)
    if n == 0:
        return {k: np.nan for k in [
            "rep_nwoorden", "rep_ttr", "rep_hapax_ratio",
            "rep_gem_zinslengte", "rep_ratio_nominalisaties",
            "rep_formele_taal_ratio", "rep_ratio_afkortingen",
        ]}
    n_nom  = _nominalisaties(text)
    n_abbr = _abbreviations(text)
    return {
        "rep_nwoorden":             n,
        "rep_ttr":                  round(_ttr(tokens), 4),
        "rep_hapax_ratio":          round(_hapax_ratio(tokens), 4),
        "rep_gem_zinslengte":       _avg_sentence_length(text),
        "rep_ratio_nominalisaties": round(100 * n_nom  / n, 2),
        "rep_formele_taal_ratio":   round(_formele_taal_ratio(tokens), 4),
        "rep_ratio_afkortingen":    round(100 * n_abbr / n, 2),
    }


# ── Title cleaning ─────────────────────────────────────────────────────────────


def extract_report_title(s: str) -> str:
    """
    Extract the embedded AR report title from a 'Beantwoording...' string.
    Falls back to the original string if no pattern matches.
    """
    # Pattern: 'over/bij (het) rapport X'
    m = re.search(
        r'(?:over|bij)\s+het\s+rapport\s+(.+)',
        s, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # Pattern: 'bij de publicatie X'
    m = re.search(r'bij de publicatie\s+(.+)', s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern: 'bij de brief.*over X'
    m = re.search(r'bij de brief.*?over\s+(.+)', s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern: Resultaten verantwoordingsonderzoek YYYY bij het Ministerie van X
    m = re.search(
        r'(Resultaten verantwoordingsonderzoek\s+\d{4}\s+(?:bij het Ministerie van|'
        r'Ministerie van|bij het|van het Ministerie van|Rijksbreed)\s+\S.+)',
        s, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # Pattern: 'bij Programma X', 'bij het Werkprogramma', etc.
    m = re.search(r'(?:^Beantwoording vragen Tweede Kamer\s+)(.+)', s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return s


def classify_report_type(title: str) -> str:
    t = str(title).lower()
    if "verantwoordingsonderzoek" in t or "jaarverslag" in t or "nationale verklaring" in t:
        return "verantwoordingsonderzoek"
    if any(k in t for k in ["werkprogramma", "verslag", "errata", "aanbieding",
                             "actualisering", "installatie", "advies", "rondetafel"]):
        return "overig"
    return "thematisch"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading data…")
    q   = pd.read_csv(ANNOTATED_Q_CSV, sep="\t")
    xl  = pd.read_excel(EXCEL_PATH, usecols=["_id", "title", "published_at", "doc_content"])
    with open(META_JSON_PATH) as f:
        meta_list = json.load(f)
    df_meta = pd.DataFrame(meta_list)

    # Separate actual reports from Q&A letters in Excel
    is_qa = xl["title"].str.contains(
        r"antwoord|kamervragen|beantwoord", case=False, na=False
    )
    xl_reports = xl[~is_qa].reset_index(drop=True)
    print(f"  Excel reports (non-Q&A): {len(xl_reports)}")
    print(f"  Meta-reports catalog:    {len(df_meta)}")

    # Unique rapport_titels from Q data
    raps = (
        q[["rapport_titel", "src_id"]]
        .drop_duplicates("rapport_titel")
        .dropna(subset=["rapport_titel"])
        .reset_index(drop=True)
    )
    print(f"  Unique rapport_titels:   {len(raps)}")

    # Build combined title catalog: Excel first, then meta-reports
    xl_titles   = xl_reports["title"].tolist()
    meta_titles = df_meta["title"].tolist()
    catalog     = xl_titles + meta_titles   # index offset: len(xl_titles)

    rows = []
    for _, row in raps.iterrows():
        rt = str(row["rapport_titel"])
        is_beantwoording = bool(re.match(r"Beantwoording|antwoorden op", rt, re.IGNORECASE))
        search_title = extract_report_title(rt) if is_beantwoording else rt

        res = process.extractOne(
            search_title, catalog,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=MATCH_THRESHOLD_LOW,
        )

        rec: dict = {"rapport_titel": rt, "search_title": search_title}

        if res:
            matched_title, score, idx = res
            rec["matched_title"] = matched_title
            rec["match_score"]   = round(score, 1)

            if idx < len(xl_titles):
                rec["match_source"] = "excel"
                doc_content = str(xl_reports.iloc[idx]["doc_content"] or "")
                rec.update(compute_text_features(doc_content))
                # Try to find category in meta-reports via title
                meta_res = process.extractOne(
                    matched_title, meta_titles,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=70,
                )
                rec["category"] = df_meta.iloc[meta_res[2]]["category"] if meta_res else None
            else:
                rec["match_source"] = "meta"
                meta_idx = idx - len(xl_titles)
                rec["category"] = df_meta.iloc[meta_idx]["category"]
                # No text available from meta-only match
                rec.update({k: np.nan for k in [
                    "rep_nwoorden", "rep_ttr", "rep_hapax_ratio",
                    "rep_gem_zinslengte", "rep_ratio_nominalisaties",
                    "rep_formele_taal_ratio", "rep_ratio_afkortingen",
                ]})
        else:
            rec.update({
                "matched_title": None,
                "match_score":   0.0,
                "match_source":  None,
                "category":      None,
                **{k: np.nan for k in [
                    "rep_nwoorden", "rep_ttr", "rep_hapax_ratio",
                    "rep_gem_zinslengte", "rep_ratio_nominalisaties",
                    "rep_formele_taal_ratio", "rep_ratio_afkortingen",
                ]},
            })

        rec["report_type"] = classify_report_type(search_title)
        rows.append(rec)

    result = pd.DataFrame(rows)

    # Summary
    high  = (result["match_score"] >= MATCH_THRESHOLD_HIGH).sum()
    med   = ((result["match_score"] >= MATCH_THRESHOLD_LOW) &
             (result["match_score"] <  MATCH_THRESHOLD_HIGH)).sum()
    low   = (result["match_score"] <  MATCH_THRESHOLD_LOW).sum()
    print(f"\nMatch quality:")
    print(f"  High confidence (≥{MATCH_THRESHOLD_HIGH}): {high}")
    print(f"  Moderate ({MATCH_THRESHOLD_LOW}-{MATCH_THRESHOLD_HIGH-1}):        {med}")
    print(f"  No match (<{MATCH_THRESHOLD_LOW}):           {low}")
    print(f"\nReport types: {result['report_type'].value_counts().to_dict()}")
    print(f"Category coverage: {result['category'].notna().sum()} / {len(result)}")
    print(f"Text features coverage: {result['rep_nwoorden'].notna().sum()} / {len(result)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"\nSaved → {OUTPUT_CSV}")

    # Show sample
    print("\nSample matches:")
    print(result[["rapport_titel", "matched_title", "match_score",
                  "match_source", "category", "report_type",
                  "rep_nwoorden", "rep_ratio_afkortingen"]].head(15).to_string())


if __name__ == "__main__":
    main()
