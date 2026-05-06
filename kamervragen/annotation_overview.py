#!/usr/bin/env python3
"""
annotation_overview.py — Descriptive overview of kamervragen annotation results.

Produces:
  - Console summary
  - reports/overview-report.md

Usage:
    python annotation_overview.py
"""

from pathlib import Path

import pandas as pd

from paths import ANNOTATED_Q_CSV, ANNOTATED_A_CSV, METADATA_CSV, REPORTS_DIR

# ── Label orders ───────────────────────────────────────────────────────────────

Q_LABELS = ["FEIT", "CAU", "OOR", "ADV", "?"]
A_LABELS = ["FEIT", "CAU", "OOR", "ADV", "DEFL"]
CONF_ORDER = ["H", "M", "L"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def _normalise_year(s: str) -> str:
    """Normalise vergaderjaar variants to a canonical four-digit start year."""
    s = str(s).strip()
    if s.startswith("<"):
        return s
    s = s.replace("/", "-").replace("–", "-")
    parts = s.split("-")
    if len(parts) >= 2 and len(parts[0]) == 4:
        return f"{parts[0]}-{parts[1]}"
    return parts[0] if parts[0].isdigit() else s


# ── Label descriptions ─────────────────────────────────────────────────────────

Q_LABEL_DESC = {
    "FEIT": "Factual question — answer directly citable from the report",
    "CAU":  "Causal question — AR must explain a mechanism or cause",
    "OOR":  "Judgement question — AR evaluates, interprets, or assesses",
    "ADV":  "Advisory question — AR gives prescriptive recommendations (includes follow-up research requests)",
    "?":    "Uncodeable — extraction artefact",
}

A_LABEL_DESC = {
    "FEIT": "AR reports facts ('our research found that…')",
    "CAU":  "AR constructs a causal explanation ('this is caused by…', 'the mechanism is…')",
    "OOR":  "AR takes a position ('in our view…', 'we find that…')",
    "ADV":  "AR gives a recommendation ('we recommend…', 'the minister should…')",
    "DEFL": "AR deflects — out of scope or refers to the responsible minister",
}


# ── Analysis ───────────────────────────────────────────────────────────────────


def analyse() -> str:
    """Run the overview analysis and return the Markdown report as a string."""
    q = pd.read_csv(ANNOTATED_Q_CSV, sep="\t")
    a = pd.read_csv(ANNOTATED_A_CSV, sep="\t")
    m = pd.read_csv(METADATA_CSV, sep="\t")

    q["vergaderjaar"] = q["vergaderjaar"].apply(_normalise_year)

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append("\n" + "#" * level + " " + text)

    def p(text: str = "") -> None:
        lines.append(text)

    # ── Title ─────────────────────────────────────────────────────────────────
    lines.append("# Overview Report: Parliamentary Questions to the Netherlands Court of Audit")
    p()
    p("Automated annotation of question and answer labels using Claude "
      "(question schema v0.7 / answer schema v0.3).")
    p(f"Dataset: {len(q)} questions · {len(a)} answers · "
      f"{q['src_id'].nunique()} source documents")
    p()

    # ── 1. Question label distribution ────────────────────────────────────────
    h(2, "1. Question Label Distribution (Schema A)")
    p()
    p("Each question is assigned one of four primary labels reflecting the "
      "**cognitive demand** placed on the Court of Audit (AR), ordered from "
      "passive retrieval (FEIT) to active prescription (ADV):")
    p()
    for lbl in Q_LABELS:
        if lbl in Q_LABEL_DESC:
            p(f"- **{lbl}** — {Q_LABEL_DESC[lbl]}")
    p()

    q_counts = q["llm_label"].value_counts()
    q_total = len(q)

    rows = []
    for lbl in Q_LABELS:
        n = int(q_counts.get(lbl, 0))
        rows.append({
            "Label": lbl,
            "Description": Q_LABEL_DESC.get(lbl, ""),
            "N": n,
            "%": f"{100 * n / q_total:.1f}",
        })
    df_q = pd.DataFrame(rows)
    df_q.loc[len(df_q)] = {"Label": "**Total**", "Description": "", "N": q_total, "%": "100.0"}
    p(_md_table(df_q))
    p()
    p("**OOR** (judgement questions) is the most frequent category, "
      "followed by **FEIT** (factual questions). **ADV** includes requests for "
      "recommendations and follow-up research (formerly the separate AGN category, "
      "merged in schema v0.7). The distribution reflects a parliament that primarily "
      "uses AR reports to probe evaluative assessments rather than simply retrieve facts.")
    p()

    # ── 2. Confidence distribution (questions) ────────────────────────────────
    h(2, "2. Annotator Confidence (Questions)")
    p()
    p("The LLM annotator assigns a confidence level alongside each label:")
    p("- **H** (High) — one label clearly fits, no serious alternative")
    p("- **M** (Medium) — one plausible alternative, choice is defensible")
    p("- **L** (Low) — two or more equally plausible labels")
    p()
    q_conf = q["llm_confidence"].value_counts()
    conf_rows = []
    for c in CONF_ORDER:
        n = int(q_conf.get(c, 0))
        conf_rows.append({"Confidence": c, "N": n, "%": f"{100 * n / q_total:.1f}"})
    p(_md_table(pd.DataFrame(conf_rows)))
    p()
    p("The predominance of H confidence indicates that most questions are "
      "unambiguously classifiable. The ~29% M cases correspond to boundary "
      "zones: FEIT/CAU (causal explanation quoted verbatim from the report) and "
      "OOR/ADV (evaluative framing vs. prescriptive intent).")
    p()

    # ── 3. Answer label distribution ──────────────────────────────────────────
    h(2, "3. Answer Label Distribution (Schema B)")
    p()
    p("AR answers are annotated with one of five labels reflecting what AR "
      "**primarily delivers** in its response:")
    p()
    for lbl in A_LABELS:
        if lbl in A_LABEL_DESC:
            p(f"- **{lbl}** — {A_LABEL_DESC[lbl]}")
    p()

    a_counts = a["llm_label"].value_counts()
    a_total = len(a)

    rows = []
    for lbl in A_LABELS:
        n = int(a_counts.get(lbl, 0))
        rows.append({
            "Label": lbl,
            "Description": A_LABEL_DESC.get(lbl, ""),
            "N": n,
            "%": f"{100 * n / a_total:.1f}",
        })
    df_a = pd.DataFrame(rows)
    df_a.loc[len(df_a)] = {"Label": "**Total**", "Description": "", "N": a_total, "%": "100.0"}
    p(_md_table(df_a))
    p()
    p(f"*Note: {len(q) - a_total} of {len(q)} questions have answers shorter than 30 characters "
      "(mostly cross-referential answers) and were not annotated.*")
    p()
    p("**DEFL** (deflection) is now the most frequent answer type: more than one in "
      "three answers does not engage substantively with the question, either because "
      "the topic falls outside the report's scope or is referred to the responsible "
      "minister. **FEIT** is the second most common. **CAU** (causal explanation) is "
      "a new category in schema v0.3, capturing answers where AR constructs a causal "
      "account rather than citing a finding.")
    p()

    # ── 4. Q × A cross-tabulation ─────────────────────────────────────────────
    h(2, "4. Question × Answer Cross-tabulation")
    p()
    merged = q.merge(
        a[["src_id", "vraag_nr", "llm_label"]].rename(columns={"llm_label": "a_label"}),
        on=["src_id", "vraag_nr"],
        how="inner",
    )
    p(f"Matched pairs: {len(merged)} question–answer pairs joined on `src_id` + `vraag_nr`.")
    p()
    crosstab = pd.crosstab(merged["llm_label"], merged["a_label"])
    q_order = [l for l in Q_LABELS if l in crosstab.index]
    a_order = [l for l in A_LABELS if l in crosstab.columns]
    crosstab = crosstab.loc[q_order, a_order]

    ct_df = crosstab.reset_index().rename(columns={"llm_label": "Q \\ A"})
    p(_md_table(ct_df))
    p()
    p("*Rows = question label; columns = answer label.*")
    p()

    # Compute row percentages for narrative
    ct_pct = crosstab.div(crosstab.sum(axis=1), axis=0).mul(100).round(1)
    p("**Key observations:**")
    p(f"- **OOR → OOR** ({ct_pct.loc['OOR', 'OOR']:.0f}%): judgement questions "
      "most often receive evaluative answers — AR engages as a judge when audit "
      "findings support the question. This is the dominant cell.")
    p(f"- **OOR → DEFL** ({ct_pct.loc['OOR', 'DEFL']:.0f}%): when an evaluative "
      "question exceeds the report's scope, AR refuses engagement rather than "
      "substituting a factual answer.")
    p(f"- **OOR → FEIT** ({ct_pct.loc['OOR', 'FEIT']:.0f}%): cognitive downgrading "
      "exists — AR retreats to factual findings rather than taking an evaluative "
      "stance — but is no longer the dominant response to OOR questions.")
    p(f"- **ADV → ADV** ({ct_pct.loc['ADV', 'ADV']:.0f}%): advisory questions "
      "show the highest within-label alignment; prescriptive intent is reliably "
      "matched when within mandate.")
    p(f"- **ADV → DEFL** ({ct_pct.loc['ADV', 'DEFL']:.0f}%): the mandate limit — "
      "AR refuses prescriptive demands that fall outside the audit scope.")
    p(f"- **FEIT → DEFL** ({ct_pct.loc['FEIT', 'DEFL']:.0f}%): one-third of factual "
      "questions are deflected — parliament asks about specifics the audit did not cover.")
    p()

    # ── 5. Temporal trends ─────────────────────────────────────────────────────
    h(2, "5. Label Distribution over Parliamentary Years")
    p()
    q_yr = q[~q["vergaderjaar"].str.startswith("<")].copy()
    yr_counts = (
        q_yr.groupby(["vergaderjaar", "llm_label"])
        .size()
        .unstack(fill_value=0)
    )
    yr_counts = yr_counts[[l for l in ["FEIT", "CAU", "OOR", "ADV"] if l in yr_counts.columns]]
    yr_totals = yr_counts.sum(axis=1)
    yr_pct = yr_counts.div(yr_totals, axis=0).mul(100).round(1)

    yr_table = yr_pct.reset_index()
    yr_table.columns.name = None
    yr_table = yr_table.rename(columns={"vergaderjaar": "Year"})
    for col in [c for c in yr_table.columns if c != "Year"]:
        yr_table[col] = yr_table[col].apply(lambda x: f"{x:.1f}%")
    yr_table["N"] = yr_totals.values
    p(_md_table(yr_table))
    p()
    p("*Values are percentages per year; N = total questions in that year. "
      "Years with N < 10 should be interpreted with caution.*")
    p()

    # ── 6. Per commissie ──────────────────────────────────────────────────────
    h(2, "6. Label Distribution by Parliamentary Committee (top 10)")
    p()
    p("The parliamentary committee (*commissie*) that submitted the questions "
      "serves as a proxy for the policy domain of the underlying AR report. "
      "Note: committee assignment is extracted from the kamerstuk header via "
      "regex + fuzzy matching and may occasionally be misidentified.")
    p()
    qm = q.merge(m[["src_id", "commissie"]], on="src_id", how="left")
    qm = qm[qm["commissie"].notna() & (qm["commissie"] != "")]
    comm_counts = (
        qm.groupby(["commissie", "llm_label"])
        .size()
        .unstack(fill_value=0)
    )
    comm_counts = comm_counts[[l for l in Q_LABELS if l in comm_counts.columns]]
    comm_totals = comm_counts.sum(axis=1).sort_values(ascending=False)
    top10 = comm_totals.head(10).index
    comm_top = comm_counts.loc[top10].copy()
    comm_top["N"] = comm_totals.loc[top10]

    comm_df = comm_top.reset_index().rename(columns={"commissie": "Committee"})
    p(_md_table(comm_df))
    p()
    p(f"*{qm['src_id'].nunique()} documents with identified committee "
      f"out of {q['src_id'].nunique()} total.*")
    p()

    # ── 7. Document-level statistics ──────────────────────────────────────────
    h(2, "7. Document-level Statistics")
    p()
    p("*Note: the unit of analysis here is the kamerstuk — the formal Q&A letter "
      "sent by AR to parliament — not the underlying AR report. See the Effects "
      "Report for a discussion of this distinction.*")
    p()
    q_per_doc = q.groupby("src_id").size()
    p(f"- Average {q_per_doc.mean():.1f} questions per document "
      f"(min={q_per_doc.min()}, max={q_per_doc.max()}, median={q_per_doc.median():.0f})")
    p(f"- {len(m)} documents with extracted metadata; "
      f"{(m['commissie'] != '').sum()} with identified committee "
      f"({100*(m['commissie'] != '').mean():.0f}%)")
    p()

    num_stats = m[["n_woorden", "gem_zinslengte", "gem_vraaglengte",
                   "gem_antwoordlengte", "ratio_afkortingen", "formele_taal_ratio"]].describe().round(2)
    col_rename = {
        "n_woorden": "words (doc)",
        "gem_zinslengte": "avg sentence len",
        "gem_vraaglengte": "avg question len",
        "gem_antwoordlengte": "avg answer len",
        "ratio_afkortingen": "abbrev/100w",
        "formele_taal_ratio": "formal ratio",
    }
    num_stats = num_stats.rename(columns=col_rename)
    stat_rows = []
    for stat in ["mean", "std", "min", "50%", "max"]:
        row = {"Statistic": stat}
        for col in num_stats.columns:
            row[col] = num_stats.loc[stat, col]
        stat_rows.append(row)
    p(_md_table(pd.DataFrame(stat_rows)))
    p()

    # ── 8. Answer length by Q×A label ─────────────────────────────────────────
    h(2, "8. Answer Length by Question × Answer Label")
    p()
    merged2 = merged.copy()
    merged2["antwoord_nw"] = merged2["antwoord"].dropna().apply(
        lambda x: len(str(x).split())
    )
    agg = merged2.groupby(["llm_label", "a_label"])["antwoord_nw"].median().round(1).unstack()
    agg = agg.loc[
        [l for l in Q_LABELS if l in agg.index],
        [l for l in A_LABELS if l in agg.columns],
    ]
    agg_df = agg.reset_index().rename(columns={"llm_label": "Q \\ A (median words)"})
    p(_md_table(agg_df))
    p()
    p("*Median word count of the answer text, by question and answer label. "
      "DEFL answers are short by definition (typically a single redirecting sentence). "
      "OOR and ADV answer types tend to be the longest, reflecting the elaboration "
      "needed to convey a nuanced position or recommendation.*")
    p()

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    report = analyse()
    out = REPORTS_DIR / "overview-report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
