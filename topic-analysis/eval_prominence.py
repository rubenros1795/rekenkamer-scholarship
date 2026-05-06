#!/usr/bin/env python3
"""
eval_prominence.py — Topic prominence analysis for evaluation/control discourse
in Dutch parliamentary debate (1945–2022).

Computes yearly averages, smoothed trends, and structural break analysis for
the six evaluation-related LDA topics, saving figures and CSVs to results/.

Usage: python eval_prominence.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("/home/rb/Downloads/lda/lda/data-1945-2022-chunked")
RESULTS    = Path(__file__).resolve().parent.parent / "results"
FIGS       = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Evaluation topics (from lda.ipynb control_topics) ─────────────────────────

EVAL_TOPICS = {
    78:  "T78 · Fin. verantwoording",   # controle financieel verantwoording jaarverslag
    88:  "T88 · Integriteit",            # integriteit transparantie gedragscode
    100: "T100 · Inspectie & toezicht", # inspectie toezicht inspecteur kwaliteit
    194: "T194 · Beleidsevaluatie",     # evaluatie effect termijn wet
    253: "T253 · Fraude & misbruik",    # fraude misbruik fraudebestrijding
    317: "T317 · Toezichthouders",      # toezichthouder accountant autoriteit
}

TOPIC_COLS = [str(t) for t in EVAL_TOPICS]

# Analytical periods
PERIODS = {
    "1945–1969": (1945, 1969),
    "1970–1979": (1970, 1979),
    "1980–1989": (1980, 1989),
    "1990–1999": (1990, 1999),
    "2000–2009": (2000, 2009),
    "2010–2022": (2010, 2022),
}

PERIOD_COLORS = ["#999999", "#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377"]

# ── Load data ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading topic proportions …")
    usecols = ["member_or_speaker", "party-ref", "date"] + TOPIC_COLS
    df = pd.read_csv(DATA_DIR / "averaged_topics.csv", usecols=usecols)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    print(f"  {len(df):,} rows, {df['year'].min()}–{df['year'].max()}")
    return df


# ── Aggregate helpers ──────────────────────────────────────────────────────────

def yearly_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean topic proportion per year (equal-weight across speaker-day rows)."""
    return df.groupby("year")[TOPIC_COLS].mean().rename(columns=str)


def composite_eval(yearly: pd.DataFrame) -> pd.Series:
    """Unweighted mean across all six eval topics → single 'evaluation discourse' index."""
    return yearly[TOPIC_COLS].mean(axis=1)


def structural_breaks(series: pd.Series, n_bkps: int = 4) -> list[int]:
    """
    Locate structural breaks using the PELT algorithm (ruptures library).
    Falls back to evenly-spaced breakpoints if ruptures is unavailable.
    """
    try:
        import ruptures as rpt
        signal = series.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=1)
        # result is 1-indexed positions ending with len(series)
        break_years = [series.index[i - 1] for i in result[:-1]]
        return break_years
    except ImportError:
        # Even-spaced fallback
        years = list(series.index)
        step = len(years) // (n_bkps + 1)
        return [years[i * step] for i in range(1, n_bkps + 1)]


def period_means(yearly: pd.DataFrame) -> pd.DataFrame:
    """Mean per analytical period for each eval topic."""
    rows = []
    for label, (y0, y1) in PERIODS.items():
        sub = yearly.loc[y0:y1]
        row = sub.mean()
        row.name = label
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_individual_topics(yearly: pd.DataFrame, breaks: list[int]) -> None:
    """One panel per eval topic: raw yearly + smoothed trend + period shading."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    smoothed = yearly[TOPIC_COLS].ewm(span=4).mean()

    for ax, (tid, label) in zip(axes, EVAL_TOPICS.items()):
        col = str(tid)
        ax.fill_between(yearly.index, yearly[col], alpha=0.15, color="#4477AA")
        ax.plot(smoothed.index, smoothed[col], lw=1.8, color="#4477AA", label="Trend (EWM-4)")
        for bk in breaks:
            ax.axvline(bk, color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Gem. aandeel", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlim(1945, 2022)

    fig.suptitle(
        "Prominentie van evaluatie-gerelateerde topics in het Nederlandse parlement (1945–2022)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = FIGS / "eval_topics_individual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parent)}")


def fig_composite_index(composite: pd.Series, breaks: list[int]) -> None:
    """Composite evaluation discourse index with structural breaks and period bands."""
    fig, ax = plt.subplots(figsize=(13, 5))
    smoothed = composite.ewm(span=4).mean()

    # Period shading
    period_items = list(PERIODS.items())
    for i, ((label, (y0, y1)), color) in enumerate(zip(period_items, PERIOD_COLORS)):
        ax.axvspan(y0, min(y1 + 1, 2023), alpha=0.07, color=color, label=label)

    ax.fill_between(composite.index, composite.values, alpha=0.15, color="#333333")
    ax.plot(smoothed.index, smoothed.values, lw=2.2, color="#333333", label="Trend (EWM-4)")
    for bk in breaks:
        ax.axvline(bk, color="firebrick", ls="--", lw=1, alpha=0.8)
        ax.text(bk + 0.3, ax.get_ylim()[1] * 0.98, str(bk),
                color="firebrick", fontsize=7, va="top")

    ax.set_title(
        "Composiet evaluatie-discours index (gem. T78, T88, T100, T194, T253, T317)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Jaar")
    ax.set_ylabel("Gemiddeld topic-aandeel")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_xlim(1945, 2022)
    fig.tight_layout()
    out = FIGS / "eval_composite_index.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parent)}")


def fig_period_heatmap(period_df: pd.DataFrame) -> None:
    """Heatmap: eval topic × period, normalized to 1945–1969 baseline."""
    baseline = period_df.loc["1945–1969"]
    normed = period_df.div(baseline)

    fig, ax = plt.subplots(figsize=(10, 4))
    import matplotlib.cm as cm
    cax = ax.imshow(normed.T.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=2.5)
    fig.colorbar(cax, ax=ax, label="Ratio t.o.v. 1945–1969")

    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(list(PERIODS.keys()), fontsize=9)
    ax.set_yticks(range(len(EVAL_TOPICS)))
    ax.set_yticklabels(list(EVAL_TOPICS.values()), fontsize=9)
    ax.set_title("Relatieve prominentie per periode (t.o.v. 1945–1969 baseline)", fontweight="bold")

    for i, row in enumerate(normed.T.values):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.7 < val < 1.8 else "white")

    fig.tight_layout()
    out = FIGS / "eval_period_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parent)}")


def fig_party_deviation(df: pd.DataFrame, period: tuple[int, int] = (2000, 2022)) -> None:
    """
    Z-score deviation of party from yearly mean for composite eval index,
    aggregated over a given period. Shows which parties over/under-use
    evaluation discourse relative to the floor.
    """
    sub = df[(df["year"] >= period[0]) & (df["year"] <= period[1])].copy()
    sub["composite"] = sub[TOPIC_COLS].mean(axis=1)

    # Daily mean (floor)
    daily_mean = sub.groupby("date")["composite"].transform("mean")
    sub["z"] = (sub["composite"] - daily_mean) / (sub.groupby("date")["composite"].transform("std") + 1e-9)

    # Party averages (parties with ≥ 200 speech-days)
    party_z = sub.groupby("party-ref")["z"].agg(["mean", "count"])
    party_z = party_z[party_z["count"] >= 200].sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, max(4, len(party_z) * 0.35)))
    colors = ["#EE6677" if m > 0 else "#4477AA" for m in party_z["mean"]]
    ax.barh(party_z.index, party_z["mean"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Gem. z-score afwijking (↑ = meer evaluatiediscours)", fontsize=9)
    ax.set_title(
        f"Partij-deviatie van evaluatiediscours {period[0]}–{period[1]} (t.o.v. daggemiddelde)",
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    out = FIGS / "eval_party_deviation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parent)}")


def fig_decade_composition(period_df: pd.DataFrame) -> None:
    """Stacked area chart showing composition of eval discourse across periods."""
    fig, ax = plt.subplots(figsize=(11, 5))
    topic_labels = list(EVAL_TOPICS.values())
    cols = TOPIC_COLS
    data = period_df[cols].values
    data_normed = data / data.sum(axis=1, keepdims=True)  # share within eval cluster

    palette = ["#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#66CCEE"]
    ax.stackplot(
        range(len(PERIODS)), data_normed.T,
        labels=topic_labels, colors=palette, alpha=0.85,
    )
    ax.set_xticks(range(len(PERIODS)))
    ax.set_xticklabels(list(PERIODS.keys()), fontsize=9)
    ax.set_ylabel("Aandeel binnen evaluatie-cluster")
    ax.set_title("Interne compositie van het evaluatie-cluster per periode", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIGS / "eval_cluster_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parent)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data()

    print("\n[1] Computing yearly averages …")
    yearly = yearly_means(df)
    composite = composite_eval(yearly)

    print("[2] Detecting structural breaks …")
    breaks = structural_breaks(composite.loc[1970:])
    print(f"    Breakpoints: {breaks}")

    print("[3] Computing period means …")
    period_df = period_means(yearly)
    print(period_df.to_string())

    print("\n[4] Saving CSVs …")
    yearly.to_csv(RESULTS / "eval_topics_yearly.csv")
    composite.rename("composite").to_csv(RESULTS / "eval_composite_yearly.csv")
    period_df.to_csv(RESULTS / "eval_topics_period_means.csv")
    print(f"    Saved to {RESULTS.relative_to(RESULTS.parent)}/")

    print("\n[5] Generating figures …")
    fig_individual_topics(yearly, breaks)
    fig_composite_index(composite, breaks)
    fig_period_heatmap(period_df)
    fig_decade_composition(period_df)
    fig_party_deviation(df, period=(2000, 2022))

    print("\nDone.")


if __name__ == "__main__":
    main()
