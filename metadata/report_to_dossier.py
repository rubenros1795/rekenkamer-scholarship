"""
Minimal: src_title → SRU → dossier_nummer (with majority vote).

Usage:
    from title_to_dossier import resolve_dossier

    dossier, info = resolve_dossier("Beantwoording Kamervragen over het rapport Kosten jeugdbescherming en jeugdreclassering")
    # → ("31839", {"votes": {"31839": 3}, "n_hits": 3, "tier": 1, ...})

Tests:  python title_to_dossier.py --test
Demo:   python title_to_dossier.py --demo "your title here"
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

SRU_URL = "https://repository.overheid.nl/sru"

# ---------- Token extraction ----------

BOILERPLATE = {
    "de", "het", "een", "van", "en", "in", "op", "voor", "bij", "over",
    "aan", "te", "tot", "met", "om", "uit", "naar", "door", "is", "zijn",
    "of", "als", "dan", "ook", "niet", "geen", "deze", "dit", "die",
    "rapport", "rapporten", "aanbieding", "resultaten",
    "kamervragen", "beantwoording", "lijst", "vragen", "antwoorden",
    "brief", "kabinetsreactie", "bestuurlijke", "reactie",
}


def extract_tokens(title: str, max_tokens: int = 6) -> list[str]:
    """Distinctive tokens for SRU title search."""
    raw = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", title.lower())
    out: list[str] = []
    for t in raw:
        if t.isdigit() and len(t) == 4:
            out.append(t)  # year — keep
        elif t in BOILERPLATE or len(t) < 3:
            continue
        else:
            out.append(t)
    seen: set[str] = set()
    deduped = [t for t in out if not (t in seen or seen.add(t))]
    return deduped[:max_tokens]


# ---------- SRU query + fetch ----------

def build_url(tokens: list[str], strict_creator: bool = True,
              max_records: int = 20) -> str:
    """Build the SRU query URL."""
    token_str = " ".join(tokens)
    parts = [
        "c.product-area==officielepublicaties",
        "dt.type==Kamerstuk",
        f'dt.title all "{token_str}"',
    ]
    if strict_creator:
        parts.append('dt.creator=="Algemene Rekenkamer"')
    cql = " AND ".join(parts)
    params = {"query": cql, "maximumRecords": max_records,
              "httpAccept": "application/xml"}
    return f"{SRU_URL}?{urlencode(params)}"


def fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


# ---------- XML parsing ----------

# Note: KOOP SRU response uses a few namespaces, but identifiers can also
# appear without namespace prefix. We extract identifiers by regex on the
# response body — robust to namespace variation.

KST_PATTERN = re.compile(r"\bkst-(\d+)(?:-([IVX]+[A-Z]?|[A-Z]))?-(\d+)\b")


def extract_kst_ids(xml_bytes: bytes) -> list[tuple[str, str | None, str]]:
    """
    Pull all kst-... identifiers out of the SRU response body.

    Returns list of (dossier_base, suffix, nummer) tuples.
    Deduplicates within a single response.
    """
    text = xml_bytes.decode("utf-8", errors="replace")
    seen: set[tuple] = set()
    out: list[tuple[str, str | None, str]] = []
    for m in KST_PATTERN.finditer(text):
        key = (m.group(1), m.group(2), m.group(3))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def total_hits(xml_bytes: bytes) -> int | None:
    """Pull <numberOfRecords> from SRU response."""
    m = re.search(r"<[^>]*numberOfRecords[^>]*>(\d+)<", xml_bytes.decode("utf-8", errors="replace"))
    return int(m.group(1)) if m else None


# ---------- Resolve ----------

@dataclass
class ResolveResult:
    dossier_base: str | None
    dossier_suffix: str | None
    confidence: str               # "high" | "medium" | "low" | "none"
    n_supporting: int             # how many kst-ids voted for this dossier
    n_total: int                  # how many kst-ids in total
    tier: int                     # which tier worked
    sample_kst_ids: list[str]     # kst-ids that voted for the chosen dossier
    query_url: str


def resolve_dossier(title: str, *, sleep_s: float = 0.5,
                    verbose: bool = False) -> ResolveResult:
    """
    Resolve src_title → dossier number via SRU with majority voting.

    Strategy:
      Tier 1: title + creator filter (strict)
      Tier 2: title only (broader, used if Tier 1 returns nothing)

    Voting:
      Group all returned kst-ids by (dossier_base, suffix).
      Pick the (base, suffix) combo with the most votes.
      Confidence = supporting_votes / total_votes.
    """
    tokens = extract_tokens(title)
    if not tokens:
        return ResolveResult(None, None, "none", 0, 0, 0, [], "")

    if verbose:
        print(f"  tokens: {tokens}", file=sys.stderr)

    for tier, strict in [(1, True), (2, False)]:
        url = build_url(tokens, strict_creator=strict)
        if verbose:
            print(f"  tier {tier}: {url}", file=sys.stderr)
        try:
            xml = fetch(url)
        except Exception as e:
            if verbose:
                print(f"  tier {tier} fetch failed: {e}", file=sys.stderr)
            continue
        time.sleep(sleep_s)

        ids = extract_kst_ids(xml)
        if not ids:
            if verbose:
                print(f"  tier {tier}: no kst-ids found", file=sys.stderr)
            continue

        # Vote on (base, suffix) — the document number within the dossier
        # is irrelevant for dossier identification.
        votes = Counter((base, suffix) for base, suffix, _ in ids)
        (winning_base, winning_suffix), n_supporting = votes.most_common(1)[0]
        n_total = len(ids)
        ratio = n_supporting / n_total

        confidence = (
            "high"   if ratio >= 0.8 and n_supporting >= 2 else
            "medium" if ratio >= 0.5 else
            "low"
        )

        sample = [
            f"kst-{b}{'-' + s if s else ''}-{n}"
            for b, s, n in ids
            if (b, s) == (winning_base, winning_suffix)
        ][:5]

        return ResolveResult(
            dossier_base=winning_base,
            dossier_suffix=winning_suffix,
            confidence=confidence,
            n_supporting=n_supporting,
            n_total=n_total,
            tier=tier,
            sample_kst_ids=sample,
            query_url=url,
        )

    return ResolveResult(None, None, "none", 0, 0, 0, [], "")


def format_dossier(r: ResolveResult) -> str | None:
    """Render dossier as '36560-I' or '31839' or None."""
    if not r.dossier_base:
        return None
    if r.dossier_suffix:
        return f"{r.dossier_base}-{r.dossier_suffix}"
    return r.dossier_base


# ---------- Pandas helper (optional) ----------

def resolve_dataframe(df, title_col: str = "src_title",
                      sleep_s: float = 0.5, verbose: bool = True):
    """
    Apply resolve_dossier to a dataframe. Deduplicates by title first,
    then joins back. Returns df with new columns.

    Requires pandas (not imported at module level to keep deps minimal).
    """
    import pandas as pd

    unique_titles = df[title_col].dropna().drop_duplicates().tolist()
    if verbose:
        print(f"Resolving {len(unique_titles)} unique titles...", file=sys.stderr)

    results = {}
    for i, title in enumerate(unique_titles, 1):
        r = resolve_dossier(title, sleep_s=sleep_s)
        results[title] = r
        if verbose and i % 10 == 0:
            print(f"  {i}/{len(unique_titles)}", file=sys.stderr)

    df = df.copy()
    df["dossier_resolved"] = df[title_col].map(
        lambda t: format_dossier(results.get(t)) if t in results else None)
    df["dossier_confidence"] = df[title_col].map(
        lambda t: results.get(t).confidence if t in results else "none")
    df["dossier_n_supporting"] = df[title_col].map(
        lambda t: results.get(t).n_supporting if t in results else 0)
    df["dossier_n_total"] = df[title_col].map(
        lambda t: results.get(t).n_total if t in results else 0)
    df["dossier_tier"] = df[title_col].map(
        lambda t: results.get(t).tier if t in results else 0)
    return df


# ---------- Tests ----------

# Synthetic SRU response for testing extraction without hitting the API.
SAMPLE_XML = b"""<?xml version="1.0"?>
<searchRetrieveResponse xmlns="http://docs.oasis-open.org/ns/search-ws/sruResponse">
  <numberOfRecords>4</numberOfRecords>
  <records>
    <record><recordData>
      <identifier>kst-31839-1</identifier>
      <title>Aanbieding rapport Kosten jeugdbescherming</title>
    </recordData></record>
    <record><recordData>
      <identifier>kst-31839-2</identifier>
      <title>Lijst van vragen en antwoorden Kosten jeugdbescherming</title>
    </recordData></record>
    <record><recordData>
      <identifier>kst-31839-158</identifier>
      <title>Brief regering Kosten jeugdbescherming</title>
    </recordData></record>
    <record><recordData>
      <identifier>kst-32500-12</identifier>
      <title>Onrelated document mentioning jeugdbescherming briefly</title>
    </recordData></record>
  </records>
</searchRetrieveResponse>
"""

SAMPLE_XML_AMBIGUOUS = b"""<?xml version="1.0"?>
<searchRetrieveResponse>
  <numberOfRecords>4</numberOfRecords>
  <records>
    <record><identifier>kst-36560-I-1</identifier></record>
    <record><identifier>kst-36560-I-2</identifier></record>
    <record><identifier>kst-36560-VIII-1</identifier></record>
    <record><identifier>kst-35830-I-3</identifier></record>
  </records>
</searchRetrieveResponse>
"""


def run_tests() -> None:
    print("=== Test: extract_tokens ===")
    assert extract_tokens("Kosten jeugdbescherming en jeugdreclassering") \
        == ["kosten", "jeugdbescherming", "jeugdreclassering"]
    assert extract_tokens("Resultaten verantwoordingsonderzoek 2023 De Koning") \
        == ["verantwoordingsonderzoek", "2023", "koning"]
    print("  OK")

    print("\n=== Test: extract_kst_ids ===")
    ids = extract_kst_ids(SAMPLE_XML)
    print(f"  Found {len(ids)} unique kst-ids:")
    for b, s, n in ids:
        print(f"    base={b} suffix={s} nummer={n}")
    assert len(ids) == 4
    assert ("31839", None, "1") in ids
    assert ("32500", None, "12") in ids
    print("  OK")

    print("\n=== Test: total_hits ===")
    assert total_hits(SAMPLE_XML) == 4
    print("  OK (parsed numberOfRecords=4)")

    print("\n=== Test: voting on clean case ===")
    # Simulate the voting logic on SAMPLE_XML
    ids = extract_kst_ids(SAMPLE_XML)
    votes = Counter((b, s) for b, s, _ in ids)
    print(f"  votes: {dict(votes)}")
    winner, n = votes.most_common(1)[0]
    print(f"  winner: {winner} with {n}/{sum(votes.values())} votes")
    assert winner == ("31839", None)
    assert n == 3
    print("  OK — 31839 wins 3-1 against the spurious 32500 hit")

    print("\n=== Test: voting on ambiguous (multi-suffix) case ===")
    ids = extract_kst_ids(SAMPLE_XML_AMBIGUOUS)
    votes = Counter((b, s) for b, s, _ in ids)
    print(f"  votes: {dict(votes)}")
    winner, n = votes.most_common(1)[0]
    print(f"  winner: {winner} with {n}/{sum(votes.values())} votes")
    # 36560-I has 2 votes, others have 1 each — 36560-I wins
    assert winner == ("36560", "I")
    print("  OK — correctly picks 36560-I when chapter-suffixed dossiers split votes")

    print("\n=== Test: confidence calculation ===")
    cases = [
        # (n_supporting, n_total, expected_confidence)
        # Rule: ratio≥0.8 AND n_supporting≥2 → high; ratio≥0.5 → medium; else low
        (3, 3, "high"),     # 100%, 3 supporting
        (3, 4, "medium"),   # 75% < 80%
        (4, 5, "high"),     # 80%, 4 supporting
        (2, 3, "medium"),   # 67%
        (2, 4, "medium"),   # 50%
        (1, 3, "low"),      # 33%
        (1, 1, "medium"),   # 100% ratio but only 1 supporting → falls through to medium
        (1, 2, "medium"),   # 50%
    ]
    for n_sup, n_tot, expected in cases:
        ratio = n_sup / n_tot
        got = ("high" if ratio >= 0.8 and n_sup >= 2
               else "medium" if ratio >= 0.5
               else "low")
        status = "OK " if got == expected else "FAIL"
        print(f"  [{status}] {n_sup}/{n_tot} ({ratio:.0%}) → {got} (expected {expected})")


def run_demo(title: str) -> None:
    print(f"Resolving: {title}")
    print(f"Tokens: {extract_tokens(title)}")
    print(f"Querying SRU...")
    r = resolve_dossier(title, verbose=True)
    print(f"\nResult:")
    print(f"  dossier:    {format_dossier(r)}")
    print(f"  confidence: {r.confidence}")
    print(f"  votes:      {r.n_supporting}/{r.n_total}")
    print(f"  tier:       {r.tier}")
    print(f"  samples:    {r.sample_kst_ids}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--demo", metavar="TITLE",
                    help="Resolve a single title against the live API")
    args = ap.parse_args()
    if args.test:
        run_tests()
    elif args.demo:
        run_demo(args.demo)
    else:
        ap.print_help()
