#!/usr/bin/env python3
"""
eval_period_corr.py — Period-specific correlation analysis for evaluation discourse.

For each analytical period, identifies the non-evaluation topics that are most
strongly correlated with each evaluation topic, after CLR-transforming the
compositional topic proportions and regressing out other eval topics (partial
correlation via Lasso residuals).

Outputs per-period top-K association tables and comparison figures.

Usage: python eval_period_corr.py [--topk 30]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/rb/Downloads/lda/lda/data-1945-2022-chunked")
RESULTS  = Path(__file__).resolve().parent.parent / "results"
FIGS     = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

EVAL_TOPICS = {
    78:  "Fin. verantwoording",
    88:  "Integriteit",
    100: "Inspectie & toezicht",
    194: "Beleidsevaluatie",
    253: "Fraude & misbruik",
    317: "Toezichthouders",
}
EVAL_COLS  = [str(t) for t in EVAL_TOPICS]
ALL_TOPICS = list(range(500))

PERIODS = {
    "1970–1979": (1970, 1979),
    "1980–1989": (1980, 1989),
    "1990–1999": (1990, 1999),
    "2000–2009": (2000, 2009),
    "2010–2022": (2010, 2022),
}

FOCAL_TOPIC = 78   # main eval topic for detailed cross-period comparisons

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, dict[int, str]]:
    print("Loading topic proportions …")
    df = pd.read_csv(DATA_DIR / "averaged_topics.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    print(f"  {len(df):,} rows")

    # Topic keywords
    kd = pd.read_csv(
        DATA_DIR / "topic-keys.txt", sep="\t", header=None,
        names=["topic_id", "weight", "keys"],
    )
    ks = {int(row.topic_id): str(row["keys"]).strip() for _, row in kd.iterrows()}
    return df, ks


# ── CLR transform ──────────────────────────────────────────────────────────────

def clr_transform(mat: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform for compositional data."""
    mat = np.clip(mat, 1e-10, None)
    log_mat = np.log(mat)
    return log_mat - log_mat.mean(axis=1, keepdims=True)


# ── Correlation for one period ─────────────────────────────────────────────────

def period_correlations(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    focal: int = FOCAL_TOPIC,
    topk: int = 30,
) -> pd.DataFrame:
    """
    For a given period, compute:
      - Pearson r between CLR(focal_topic) and CLR(every_other_topic)
      - Partial r after regressing out other eval topics from focal topic

    Returns sorted DataFrame of top-k positive and top-k negative associations.
    """
    y0, y1 = year_range
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()

    # Daily aggregate to reduce speaker autocorrelation within a day
    topic_cols_all = [str(t) for t in ALL_TOPICS]
    daily = sub.groupby("date")[topic_cols_all].mean()

    # CLR transform
    clr = pd.DataFrame(
        clr_transform(daily.values),
        index=daily.index,
        columns=daily.columns,
    )

    focal_col = str(focal)
    other_eval = [c for c in EVAL_COLS if c != focal_col]

    # Partial correlation: residualise focal topic on other eval topics
    X = clr[other_eval].values
    y = clr[focal_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    las = Lasso(alpha=0.01, max_iter=10000)
    las.fit(X_scaled, y)
    focal_resid = y - las.predict(X_scaled)

    results = []
    for col in topic_cols_all:
        if col in EVAL_COLS:
            continue
        other = clr[col].values
        # Raw Pearson r
        r_raw, p_raw = stats.pearsonr(clr[focal_col].values, other)
        # Partial r (focal residualised on other eval topics)
        r_part, p_part = stats.pearsonr(focal_resid, other)
        results.append({
            "topic_id": int(col),
            "r_raw": r_raw,
            "p_raw": p_raw,
            "r_partial": r_part,
            "p_partial": p_part,
        })

    out = pd.DataFrame(results).sort_values("r_partial", ascending=False)
    out["n_days"] = len(daily)
    return out


def period_internal_corr(
    df: pd.DataFrame,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    """Correlation matrix among the 6 eval topics for a given period."""
    y0, y1 = year_range
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
    daily = sub.groupby("date")[EVAL_COLS].mean()
    clr = pd.DataFrame(
        clr_transform(daily.values),
        index=daily.index,
        columns=daily.columns,
    )
    return clr.corr(method="pearson")


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_top_associations_by_period(
    period_corrs: dict[str, pd.DataFrame],
    ks: dict[int, str],
    focal: int = FOCAL_TOPIC,
    topk: int = 20,
) -> None:
    """
    For each period, show the top-K positively correlated topics with the focal
    eval topic (partial correlation, CLR-transformed daily proportions).
    """
    n = len(period_corrs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 9), sharey=False)

    focal_label = EVAL_TOPICS[focal]

    for ax, (period, corr_df) in zip(axes, period_corrs.items()):
        top = corr_df.head(topk)
        labels = [f"T{tid}: {' '.join(ks.get(tid,'').split()[:4])}" for tid in top["topic_id"]]
        colors = ["#4477AA" if r > 0 else "#EE6677" for r in top["r_partial"]]
        ax.barh(labels[::-1], top["r_partial"].values[::-1], color=colors[::-1], alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(period, fontsize=10, fontweight="bold")
        ax.set_xlabel("Partiële r", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"Top-{topk} thematische associaties van '{focal_label}' per periode\n"
        "(partiële Pearson r, CLR-getransformeerd, daggemiddelden)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGS / f"eval_top_assoc_T{focal}_by_period.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_topic_trajectory(
    df: pd.DataFrame,
    topic_ids: list[int],
    ks: dict[int, str],
    focal: int = FOCAL_TOPIC,
    period_names: list[str] | None = None,
) -> None:
    """
    For a set of topics, show how their partial correlation with the focal eval
    topic changes across periods.  Reveals rising/falling thematic associations.
    """
    if period_names is None:
        period_names = list(PERIODS.keys())

    # Precompute correlations for all periods
    all_results: dict[str, pd.DataFrame] = {}
    for period_label, year_range in PERIODS.items():
        all_results[period_label] = period_correlations(df, year_range, focal=focal)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = plt.cm.tab10.colors

    for color, tid in zip(palette, topic_ids):
        label_short = " ".join(ks.get(tid, "").split()[:4])
        traj = [
            all_results[p].set_index("topic_id").loc[tid, "r_partial"]
            if tid in all_results[p]["topic_id"].values else np.nan
            for p in PERIODS
        ]
        ax.plot(period_names, traj, marker="o", lw=1.8, color=color,
                label=f"T{tid}: {label_short}")

    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_ylabel("Partiële r met evaluatietopic")
    ax.set_title(
        f"Temporele trajecten van geselecteerde topic-associaties met T{focal}",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = FIGS / f"eval_assoc_trajectories_T{focal}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_internal_corr_heatmaps(
    period_corrs: dict[str, pd.DataFrame],
) -> None:
    """
    Heatmap of correlations among the 6 eval topics per period.
    Shows how the cluster structure changes over decades.
    """
    n = len(period_corrs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    topic_labels = [f"T{t}" for t in EVAL_TOPICS]

    for ax, (period, cmat) in zip(axes, period_corrs.items()):
        cmat.columns = topic_labels
        cmat.index = topic_labels
        im = ax.imshow(cmat.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(6))
        ax.set_xticklabels(topic_labels, rotation=45, fontsize=7)
        ax.set_yticks(range(6))
        ax.set_yticklabels(topic_labels, fontsize=7)
        ax.set_title(period, fontsize=9, fontweight="bold")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cmat.values[i, j]:.2f}", ha="center", va="center", fontsize=6)

    fig.suptitle(
        "Interne correlaties binnen het evaluatie-cluster per periode (CLR, daggemiddelden)",
        fontweight="bold",
    )
    fig.colorbar(im, ax=axes[-1], label="Pearson r", fraction=0.04)
    fig.tight_layout()
    out = FIGS / "eval_internal_corr_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_rising_falling_assoc(
    all_period_corrs: dict[str, pd.DataFrame],
    ks: dict[int, str],
    focal: int = FOCAL_TOPIC,
    topk: int = 15,
) -> None:
    """
    Compare first (1970s) vs last (2010–2022) period: which topics newly emerged
    or faded as thematic neighbours of the focal eval topic.
    """
    first_key = "1970–1979"
    last_key  = "2010–2022"
    c_first = all_period_corrs[first_key].set_index("topic_id")
    c_last  = all_period_corrs[last_key].set_index("topic_id")

    common = c_first.index.intersection(c_last.index)
    delta = (c_last.loc[common, "r_partial"] - c_first.loc[common, "r_partial"]).sort_values()

    top_rising  = delta.tail(topk)
    top_falling = delta.head(topk)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _barh(ax, series, title, color):
        labels = [f"T{tid}: {' '.join(ks.get(tid,'').split()[:5])}" for tid in series.index]
        ax.barh(labels, series.values, color=color, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Δ partiële r (2010–2022 minus 1970–1979)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    _barh(ax1, top_rising,  "Opkomende associaties ↑",  "#228833")
    _barh(ax2, top_falling, "Afnemende associaties ↓",  "#EE6677")

    fig.suptitle(
        f"Verschuivingen in thematische associaties van '{EVAL_TOPICS[focal]}' (1970s → 2010s)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGS / f"eval_rising_falling_T{focal}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(topk: int = 30) -> None:
    df, ks = load_data()

    # ── 1. Top associations per period for focal topic ─────────────────────────
    print(f"\n[1] Computing top-{topk} associations for T{FOCAL_TOPIC} per period …")
    period_top: dict[str, pd.DataFrame] = {}
    for period_label, year_range in PERIODS.items():
        print(f"    {period_label} …", end=" ", flush=True)
        corr_df = period_correlations(df, year_range, focal=FOCAL_TOPIC, topk=topk)
        period_top[period_label] = corr_df
        # Annotate with keywords
        corr_df["keywords"] = corr_df["topic_id"].map(lambda t: " ".join(ks.get(t, "").split()[:6]))
        out = RESULTS / f"eval_T{FOCAL_TOPIC}_top_assoc_{period_label.replace('–', '-')}.csv"
        corr_df.head(topk).to_csv(out, index=False)
        print(f"saved → {out.name}")

    # ── 2. All eval topics × all other topics for each period (full) ───────────
    print(f"\n[2] Computing associations for all 6 eval topics per period …")
    for focal_id in EVAL_TOPICS:
        rows = []
        for period_label, year_range in PERIODS.items():
            print(f"    T{focal_id} {period_label} …", end=" ", flush=True)
            c = period_correlations(df, year_range, focal=focal_id)
            c["period"] = period_label
            rows.append(c.head(topk))
            print("ok")
        all_df = pd.concat(rows)
        all_df["keywords"] = all_df["topic_id"].map(lambda t: " ".join(ks.get(t, "").split()[:6]))
        out = RESULTS / f"eval_T{focal_id}_all_periods_top{topk}.csv"
        all_df.to_csv(out, index=False)
        print(f"    → {out.name}")

    # ── 3. Internal correlations among eval cluster ────────────────────────────
    print(f"\n[3] Internal correlation matrices …")
    period_internal: dict[str, pd.DataFrame] = {}
    for period_label, year_range in PERIODS.items():
        period_internal[period_label] = period_internal_corr(df, year_range)
        print(f"    {period_label} computed")

    # ── 4. Figures ─────────────────────────────────────────────────────────────
    print(f"\n[4] Generating figures …")
    fig_top_associations_by_period(period_top, ks, focal=FOCAL_TOPIC, topk=topk)
    fig_internal_corr_heatmaps(period_internal)
    fig_rising_falling_assoc(period_top, ks, focal=FOCAL_TOPIC, topk=15)

    # Select top rising topics in 2010–2022 for trajectory plot
    top_in_last = period_top["2010–2022"].head(12)["topic_id"].tolist()
    fig_topic_trajectory(df, top_in_last, ks, focal=FOCAL_TOPIC)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()
    main(topk=args.topk)
