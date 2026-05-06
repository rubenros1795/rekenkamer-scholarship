#!/usr/bin/env python3
"""
extract_metadata.py — Document-level metadata extraction for the kamervragen corpus.

For each source document (identified by src_id) the following metrics are computed:

  Broninfo
  --------
  rapport_titel        : Title of the report
  vergaderjaar         : Parliamentary year
  datum                : Date of the document
  doc_url              : URL to the source document

  Commissie
  ---------
  commissie            : Parliamentary committee (regex then fuzzy-matched)
  commissie_score      : Fuzzy match confidence (100 = exact substring match)

  Documentikenmerken
  ------------------
  n_woorden            : Total word count of doc_content
  n_unieke_woorden     : Unique word types (vocabulary size)
  ttr                  : Type-Token Ratio — lexical diversity (uniq/total)
  hapax_ratio          : Hapax legomena / vocabulary (words appearing exactly once)
  gem_zinslengte       : Average sentence length in words
  n_nominalisaties     : Count of Dutch nominalisations (-ing/-tie/-heid/-schap)
  ratio_nominalisaties : Nominalisaties per 100 words
  formele_taal_ratio   : Ratio of words with 6+ characters (proxy for formal register)
  n_afkortingen        : Count of abbreviations (acronyms + dotted abbreviations)
  ratio_afkortingen    : Abbreviations per 100 words

  Vragenprofiel
  -------------
  n_vragen             : Total questions (from structured CSV)
  n_unieke_antwoorden  : Direct answers (antwoord_zie_nr is empty)
  n_verwijzingen       : Referential answers (antwoord_zie_nr is set)
  ratio_verwijzingen   : % answers that refer to another question
  gem_vraaglengte      : Average question length in words
  gem_antwoordlengte   : Average direct-answer length in words (excl. referential)
  antwoord_vraag_ratio : avg answer words / avg question words (verbosity index)
  vraag_ttr            : Average Type-Token Ratio per individual question
  n_ext_verwijzingen   : Count of external document references in answers

Output: results/kamervragen-metadata.csv (tab-separated, one row per document)

Usage:
    uv run python "Analysis - Kamervragen/extract_metadata.py"
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

# ── Paths ─────────────────────────────────────────────────────────────────────

from paths import RAW_CSV, STRUCTURED_CSV, METADATA_CSV as OUTPUT_CSV

# ── Commissies ────────────────────────────────────────────────────────────────

COMMISSIES: list[str] = sorted([
    "Commissie voor de Rijksuitgaven",
    "Vaste commissie voor Binnenlandse Zaken",
    "Vaste commissie voor Buitenlandse Handel en Ontwikkelingssamenwerking",
    "Vaste commissie voor Buitenlandse Zaken",
    "Vaste commissie voor Defensie",
    "Vaste commissie voor Digitale Zaken",
    "Vaste commissie voor Economische Zaken",
    "Vaste commissie voor Economische Zaken en Klimaat",
    "Vaste commissie voor Europese Zaken",
    "Vaste commissie voor Financiën",
    "Vaste commissie voor Financiën en de Rijksuitgaven",
    "Vaste commissie voor Immigratie en Asiel",
    "Vaste commissie voor Infrastructuur en Milieu",
    "Vaste commissie voor Infrastructuur en Waterstaat",
    "Vaste commissie voor Justitie en Veiligheid",
    "Vaste commissie voor Klimaat en Energie",
    "Vaste commissie voor Koninkrijksrelaties",
    "Vaste commissie voor Landbouw, Natuur en Voedselkwaliteit",
    "Vaste commissie voor Medische Ethiek",
    "Vaste commissie voor Natuur en Stikstof",
    "Vaste commissie voor Onderwijs, Cultuur en Wetenschap",
    "Vaste commissie voor Sociale Zaken en Werkgelegenheid",
    "Vaste commissie voor Staatkundige Vernieuwing",
    "Vaste commissie voor Veiligheid en Justitie",
    "Vaste commissie voor Volksgezondheid, Welzijn en Sport",
    "Vaste commissie voor Volkshuisvesting en Ruimtelijke Ordening",
    "Vaste commissie voor Wonen en Ruimtelijke Ordening",
])

FUZZY_THRESHOLD = 72  # minimum rapidfuzz score to accept a commissie match

# ── Regex patterns ────────────────────────────────────────────────────────────

# Extract the "commissie voor X" fragment from the document header
RE_COMMISSIE = re.compile(
    r'commissie\s+voor\s+(?:de\s+|het\s+)?'
    r'([A-Za-zÀ-ÿ,\s]{5,80}?)'
    r'(?=\sgestelde|\s*,|\s*\.|\s*\(|\s*$)',
    re.IGNORECASE,
)

# Sentence boundaries (split on . ! ? followed by whitespace or end)
RE_SENTENCE = re.compile(r'[.!?]+(?:\s|$)')

# Dutch nominalisations
RE_NOMINALISATIE_SUFFIXES = re.compile(
    r'\b\w{5,}(?:ing|tie|iteit|heid|schap|isme|ering)\b', re.IGNORECASE
)

# Acronyms: 2–6 uppercase letters (EU, VWS, AWBZ, NATO, …)
RE_ACRONYM = re.compile(r'\b[A-Z]{2,6}\b')

# Dotted abbreviations: 1–4 letters followed by a period not at sentence end
# e.g. bijv. o.a. etc. art. nr. p. t/m
RE_ABBREV_DOT = re.compile(r'\b[a-zA-Z]{1,4}\.')

# ── Text helpers ──────────────────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, Dutch-alphabet aware."""
    return re.findall(r"\b[a-zA-ZàáâäãåæçèéêëìíîïðñòóôõöùúûüýÿÀ-ÖØ-öø-ÿ']+\b",
                      text.lower())


def _ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _hapax_ratio(tokens: list[str]) -> float:
    """Hapax legomena / vocabulary size."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(freq)


def _avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in RE_SENTENCE.split(text)]
    lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
    return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


def _nominalisaties(text: str) -> int:
    return len(RE_NOMINALISATIE_SUFFIXES.findall(text))


def _abbreviations(text: str) -> int:
    return len(RE_ACRONYM.findall(text)) + len(RE_ABBREV_DOT.findall(text))


def _formele_taal_ratio(tokens: list[str]) -> float:
    """Ratio of words with 6+ characters — proxy for formal/bureaucratic register."""
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if len(t) >= 6) / len(tokens)


# ── Commissie extraction ──────────────────────────────────────────────────────


def extract_commissie(doc_content: str, header_chars: int = 1500) -> tuple[str, float]:
    """
    1. Regex: look for 'commissie voor X' in the header and fuzzy-match the
       extracted fragment against the canonical COMMISSIES list.
    2. Fallback: fuzzy partial-ratio match of the whole header against each
       canonical commissie name.
    Returns (canonical_name, score).  score==100 means exact substring match.
    """
    header = doc_content[:header_chars]

    # Step 1 — regex extraction + canonical match
    m = RE_COMMISSIE.search(header)
    if m:
        fragment = "commissie voor " + m.group(1).strip()
        result = process.extractOne(
            fragment,
            COMMISSIES,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=60,
        )
        if result:
            return result[0], float(result[1])

    # Step 2 — direct substring check (case-insensitive)
    header_lower = header.lower()
    for c in COMMISSIES:
        if c.lower() in header_lower:
            return c, 100.0

    # Step 3 — broad fuzzy partial-ratio on header
    result = process.extractOne(
        header,
        COMMISSIES,
        scorer=fuzz.partial_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if result:
        return result[0], float(result[1])

    return "", 0.0


# ── Per-document computation ──────────────────────────────────────────────────


def compute_doc_metrics(
    src_id: str,
    doc_content: str,
    grp: pd.DataFrame,          # rows from structured CSV for this src_id
) -> dict:
    text = str(doc_content or "")
    tokens = _tokens(text)
    n = len(tokens)

    commissie, commissie_score = extract_commissie(text)

    # ── Document-level text metrics ────────────────────────────────────────
    n_uniek = len(set(tokens))
    ttr = round(_ttr(tokens), 4)
    hapax = round(_hapax_ratio(tokens), 4)
    gem_zin = _avg_sentence_length(text)
    n_nom = _nominalisaties(text)
    n_abbr = _abbreviations(text)
    formeel = round(_formele_taal_ratio(tokens), 4)

    # ── Question / answer metrics ──────────────────────────────────────────
    n_vragen = len(grp)

    # Referential answers: antwoord_zie_nr is set (not NaN, not 0)
    heeft_verwijzing = grp["antwoord_zie_nr"].notna() & (grp["antwoord_zie_nr"] != 0)
    n_verwijzingen   = int(heeft_verwijzing.sum())
    n_uniek_antw     = n_vragen - n_verwijzingen

    ratio_verw = round(n_verwijzingen / n_vragen, 4) if n_vragen else 0.0

    # Question lengths
    vraag_lengths = grp["vraag"].dropna().apply(lambda q: len(q.split()))
    gem_vraag = round(vraag_lengths.mean(), 2) if len(vraag_lengths) else 0.0

    # Direct answer lengths (exclude referential answers and empty answers)
    direct_antw = grp.loc[
        ~heeft_verwijzing & grp["antwoord"].notna() & (grp["antwoord"].str.strip() != ""),
        "antwoord",
    ]
    antw_lengths = direct_antw.apply(lambda a: len(str(a).split()))
    gem_antw = round(antw_lengths.mean(), 2) if len(antw_lengths) else 0.0

    antw_vraag_ratio = round(gem_antw / gem_vraag, 3) if gem_vraag else 0.0

    # Per-question TTR (avg lexical diversity of individual questions)
    q_ttrs = grp["vraag"].dropna().apply(lambda q: _ttr(_tokens(q)))
    vraag_ttr = round(q_ttrs.mean(), 4) if len(q_ttrs) else 0.0

    # External references mentioned in answers (verwijzingen column)
    n_ext_verw = int(grp["verwijzingen"].notna().sum())

    return {
        "src_id":               src_id,
        "commissie":            commissie,
        "commissie_score":      round(commissie_score, 1),
        "n_woorden":            n,
        "n_unieke_woorden":     n_uniek,
        "ttr":                  ttr,
        "hapax_ratio":          hapax,
        "gem_zinslengte":       gem_zin,
        "n_nominalisaties":     n_nom,
        "ratio_nominalisaties": round(100 * n_nom / n, 2) if n else 0.0,
        "formele_taal_ratio":   formeel,
        "n_afkortingen":        n_abbr,
        "ratio_afkortingen":    round(100 * n_abbr / n, 2) if n else 0.0,
        "n_vragen":             n_vragen,
        "n_unieke_antwoorden":  n_uniek_antw,
        "n_verwijzingen":       n_verwijzingen,
        "ratio_verwijzingen":   ratio_verw,
        "gem_vraaglengte":      gem_vraag,
        "gem_antwoordlengte":   gem_antw,
        "antwoord_vraag_ratio": antw_vraag_ratio,
        "vraag_ttr":            vraag_ttr,
        "n_ext_verwijzingen":   n_ext_verw,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Loading {RAW_CSV.name} …")
    raw = pd.read_csv(RAW_CSV, sep="\t",
                      usecols=["_id", "doc_content", "doc_url"])
    raw = raw.rename(columns={"_id": "src_id"})

    print(f"Loading {STRUCTURED_CSV.name} …")
    struct = pd.read_csv(STRUCTURED_CSV, sep="\t")

    # Broninfo: take first row per src_id
    broninfo = (
        struct.groupby("src_id", sort=False)
        .first()
        [["rapport_titel", "vergaderjaar", "datum"]]
        .reset_index()
    )

    print(f"Computing metrics for {len(raw)} documents …")
    rows = []
    for _, doc_row in raw.iterrows():
        src_id = doc_row["src_id"]
        grp = struct[struct["src_id"] == src_id]
        metrics = compute_doc_metrics(src_id, doc_row["doc_content"], grp)
        metrics["doc_url"] = doc_row.get("doc_url", "")
        rows.append(metrics)

    result = pd.DataFrame(rows)

    # Merge broninfo
    result = result.merge(broninfo, on="src_id", how="left")

    # Reorder columns
    col_order = [
        "src_id", "rapport_titel", "vergaderjaar", "datum", "doc_url",
        "commissie", "commissie_score",
        "n_woorden", "n_unieke_woorden", "ttr", "hapax_ratio",
        "gem_zinslengte", "n_nominalisaties", "ratio_nominalisaties",
        "formele_taal_ratio", "n_afkortingen", "ratio_afkortingen",
        "n_vragen", "n_unieke_antwoorden", "n_verwijzingen",
        "ratio_verwijzingen", "gem_vraaglengte", "gem_antwoordlengte",
        "antwoord_vraag_ratio", "vraag_ttr", "n_ext_verwijzingen",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"Saved {len(result)} rows → {OUTPUT_CSV}")

    # Quick summary
    print("\n── Commissie coverage ──────────────────────────────────────────")
    found = result["commissie"].notna() & (result["commissie"] != "")
    print(f"  Herkend: {found.sum()} / {len(result)} ({100*found.mean():.1f}%)")
    print(f"  Top commissies:\n{result['commissie'].value_counts().head(8).to_string()}")

    print("\n── Corpus-level summary ────────────────────────────────────────")
    num_cols = ["n_woorden", "ttr", "gem_zinslengte", "n_vragen",
                "ratio_verwijzingen", "gem_vraaglengte", "gem_antwoordlengte"]
    print(result[num_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
