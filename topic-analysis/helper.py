"""
helper.py — Data loading and visualisation helpers for hits.ipynb.

Keeps the notebook cells focused on narrative and figures.  Import with:
    from helper import *
or selectively:
    from helper import load_all, plot_freq_entropy, trending_topics
"""

from __future__ import annotations

import warnings
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import entropy
from scipy.special import softmax
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# ── Default style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

_PERIOD_COLORS = {"H1": "#4878CF", "H2": "#D65F5F"}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_hits(path: str | Path) -> pd.DataFrame:
    """Load hits.csv, parse dates, drop rows without a date."""
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"hits: {len(df):,} rows  |  date range: {df.date.min().date()} – {df.date.max().date()}")
    return df


def load_lda(lda_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load LDA outputs from *lda_dir*.

    Returns
    -------
    dist : DataFrame
        Document-topic distributions with a 'date' column added.
    meta : DataFrame
        Metadata (same row order as dist).
    topic_label : dict[int, str]
        Full label string per topic id.
    ks : dict[int, str]
        Top-10 keywords (comma-joined) per topic id.
    """
    lda_dir = Path(lda_dir)
    dist = pd.read_csv(lda_dir / "doc-topics.txt", sep="\t", header=None).iloc[:, 2:]
    dist.columns = range(dist.shape[1])
    meta = pd.read_csv(lda_dir / "metadata.csv")
    dist["date"] = pd.to_datetime(meta["date"])

    keys = pd.read_csv(lda_dir / "topic-keys.txt", sep="\t", header=None)
    topic_label = dict(zip(keys[0].astype(str), keys[2]))
    ks = dict(zip(keys[0], keys[2].str.split(" ").str[:10].str.join(", ")))

    print(f"dist: {len(dist):,} docs  |  {dist.shape[1] - 1} topics")
    return dist, meta, topic_label, ks


def load_noun_totals(path: str | Path) -> pd.DataFrame:
    """Load the noun-total normalisation table (date, count)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_cabinets(path: str | Path) -> pd.DataFrame:
    """Load the cabinet reference table with parsed dates."""
    cab = pd.read_csv(path).iloc[:-1, :]
    cab["Start Date"] = pd.to_datetime(cab["Start Date"], format="%d-%m-%Y")
    cab["End Date"] = pd.to_datetime(cab["End Date"], format="%d-%m-%Y")
    return cab


def load_all(hits_path, lda_dir, noun_total_path, cabinet_path=None):
    """Convenience loader — returns (hits, dist, meta, ks, total_nouns, cab)."""
    hits = load_hits(hits_path)
    dist, meta, topic_label, ks = load_lda(lda_dir)
    total_nouns = load_noun_totals(noun_total_path)
    cab = load_cabinets(cabinet_path) if cabinet_path else None
    return hits, dist, meta, ks, total_nouns, cab


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def to_half_year(s: pd.Series) -> pd.Series:
    """Map a datetime Series to 'YYYY-H1' / 'YYYY-H2' strings."""
    return s.dt.year.astype(str) + np.where(s.dt.month <= 6, "-H1", "-H2")


def add_periods(hits: pd.DataFrame, dist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add 'period' column and flag dist rows that appear in hits."""
    hits = hits.copy()
    dist = dist.copy()
    hits["period"] = to_half_year(hits["date"])
    dist["period"] = to_half_year(dist["date"])
    dist["in_hits"] = dist["date"].isin(set(hits["date"]))
    print(f"hit dates: {dist['in_hits'].sum():,} / {len(dist):,} dist rows flagged")
    return hits, dist


def relative_frequency(hits: pd.DataFrame, total_nouns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily hit frequency and normalise by total noun count.

    Returns a daily DataFrame with columns: date, f, rf
    """
    date_to_count = dict(zip(pd.to_datetime(total_nouns["date"]), total_nouns["count"]))
    daily = hits.groupby("date").size().reset_index(name="f")
    daily = daily[daily["date"].dt.year > 1848]
    daily["rf"] = daily["f"] / daily["date"].map(date_to_count)
    return daily


def quarterly_stats(daily: pd.DataFrame, lowess_frac: float = 0.05) -> pd.DataFrame:
    """
    Aggregate daily rf to quarters; compute entropy and summed rf.
    Adds LOESS-smoothed columns loess_entropy and loess_sum.
    """
    q = (
        daily
        .assign(quarter=lambda df: df["date"].dt.to_period("Q").dt.to_timestamp())
        .groupby("quarter")
        .agg(entropy_rf=("rf", entropy), sum_rf=("rf", "sum"))
        .reset_index()
        .sort_values("quarter")
    )
    x = np.arange(len(q))
    q["loess_entropy"] = lowess(q["entropy_rf"], x, frac=lowess_frac)[:, 1]
    q["loess_sum"] = lowess(q["sum_rf"], x, frac=lowess_frac)[:, 1]
    return q


def classify_coalition(hits: pd.DataFrame, cab: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'coalition_status' column: 'government' | 'coalition party' | 'opposition party'.
    Rows where status cannot be determined are None.
    """
    hits = hits.copy()
    hits["party_code"] = hits["party-ref"].str.split(".").str[-1].str.lower()

    gov_roles = {"minister", "staatssecretaris", "premier", "mp-minister", "government"}

    def _classify(row):
        mask = (cab["Start Date"] <= row["date"]) & (cab["End Date"] >= row["date"])
        active = cab[mask]
        if active.empty:
            return None
        coalition_parties = active.iloc[0]["Parties"].lower().split()
        if str(row.get("role", "")).lower() in gov_roles:
            return "government"
        return "coalition party" if row["party_code"] in coalition_parties else "opposition party"

    hits["coalition_status"] = hits.apply(_classify, axis=1)
    return hits.drop(columns=["party_code"])


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_COLS = list(range(300))
MIN_OBS = 30


def build_dam(dist: pd.DataFrame, hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align topic distributions with hit counts per date.

    Returns (dam, hits_day) where dam is the topic matrix and
    hits_day is the daily hit count indexed by date.
    """
    daily_hits = hits.groupby("date").size().rename("Y")
    dam = dist.set_index("date")[TOPIC_COLS].copy()
    dam["Y"] = dam.index.map(daily_hits)
    dam = dam.dropna(subset=["Y"])
    hits_day = dam.pop("Y")
    print(f"dam: {len(dam):,} docs  |  {dam.index.nunique():,} unique dates")
    return dam, hits_day


def pcorr_over_time(
    dam: pd.DataFrame,
    hits_day: pd.Series,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """
    Compute quarterly partial correlations between each topic and hit frequency.

    Uses precision-matrix inversion (pinv) per quarter.
    Returns a long DataFrame: period, topic, partial_corr.
    """
    df = dam.copy()
    df["Y"] = hits_day
    df = df.dropna()
    out = []
    for p, g in df.groupby(df.index.to_period("Q")):
        if len(g) < min_obs:
            continue
        try:
            X = g.values
            prec = np.linalg.pinv(np.cov(X, rowvar=False))
            d = np.sqrt(np.diag(prec))
            pc = -prec / np.outer(d, d)
            np.fill_diagonal(pc, 1.0)
            out.append(pd.DataFrame({
                "period": str(p),
                "topic": dam.columns,
                "partial_corr": pc[:-1, -1],
            }))
        except Exception as e:
            print(f"{p}: {e}")
    return pd.concat(out, ignore_index=True)


def topic_corr_matrix(dist_subset: pd.DataFrame) -> pd.DataFrame:
    """Pairwise partial correlation matrix for a topic distribution DataFrame."""
    X = dist_subset.values
    prec = np.linalg.pinv(np.cov(X, rowvar=False))
    d = np.sqrt(np.diag(prec))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=dist_subset.columns, columns=dist_subset.columns)


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC RANKING
# ─────────────────────────────────────────────────────────────────────────────

def trending_topics(
    resp: pd.DataFrame,
    n: int = 15,
    ewm_span: int = 12,
) -> pd.Series:
    """
    Return the top-N topics by Mann-Kendall trend slope in their
    EWM-smoothed partial-correlation time series.
    """
    try:
        from pymannkendall import original_test as pmk
    except ImportError:
        raise ImportError("pip install pymannkendall")
    return (
        resp.ewm(span=ewm_span).mean()
        .apply(lambda x: pmk(x).slope)
        .nlargest(n)
    )


def topic_scatter_data(
    resp: pd.DataFrame,
    dist_averaged: pd.DataFrame,
    ewm_span: int = 12,
) -> pd.DataFrame:
    """
    Build bubble-chart data: variance, mean, max partial-corr per topic,
    plus a z-score of the topic's overall prominence in the corpus.
    """
    from scipy.stats import zscore as _zscore

    zscores_raw = (
        dist_averaged
        .ewm(span=ewm_span).mean()
        .pipe(lambda df: pd.DataFrame(
            _zscore(df.values, axis=1), index=df.index, columns=df.columns
        ))
        .mean()
    )
    topic_z = pd.Series(np.log2(softmax(zscores_raw.values)), index=zscores_raw.index)

    return pd.DataFrame({
        "variance": resp.var(axis=0),
        "mean_corr": resp.mean(axis=0),
        "max_corr": resp.max(axis=0),
        "topic_z": resp.columns.map(topic_z),
    })


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_freq_entropy(
    quarterly: pd.DataFrame,
    *,
    war_shade: bool = True,
    title: str = "References to the Court of Audit over time",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Dual-axis chart: summed relative frequency (filled) + entropy (line).

    Parameters
    ----------
    quarterly : output of quarterly_stats()
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    ax2.fill_between(quarterly["quarter"], quarterly["loess_sum"],
                     color="#D65F5F", alpha=0.25, label="Summed frequency")
    ax.plot(quarterly["quarter"], quarterly["loess_entropy"],
            color="#4878CF", linewidth=1.5, label="Entropy")

    for spine, color in [("left", "#4878CF"), ("right", "#D65F5F")]:
        ax.spines[spine].set_color(color) if spine == "left" else ax2.spines[spine].set_color(color)
    ax.tick_params(axis="y", colors="#4878CF")
    ax2.tick_params(axis="y", colors="#D65F5F")
    ax.yaxis.label.set_color("#4878CF")
    ax2.yaxis.label.set_color("#D65F5F")

    ax.set_xlim(pd.Timestamp("1849-01-01"), pd.Timestamp("2022-01-01"))
    bottom = quarterly["loess_sum"].min() - (quarterly["loess_sum"].max() - quarterly["loess_sum"].min()) * 0.1
    ax2.set_ylim(bottom, quarterly["loess_sum"].max())

    if war_shade:
        ax.axvspan(pd.Timestamp("1940-01-01"), pd.Timestamp("1946-01-01"),
                   color="grey", alpha=0.15, zorder=0)

    ax.set_title(title, loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weekly_heatmap(
    hits: pd.DataFrame,
    *,
    year_start: int = 1940,
    period_size: int = 5,
    figsize: tuple = (14, 6),
    title: str = "Hit distribution by week-of-year (5-year bins)",
) -> plt.Figure:
    """Heatmap of weekly hit distribution, binned into 5-year periods."""
    monthly = (
        hits
        .groupby([hits.date.dt.year // period_size * period_size,
                  hits.date.dt.isocalendar().week])
        .size()
        .unstack()
        .fillna(0)
    )
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    subset = monthly[monthly.index > year_start]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(subset, cmap="Blues", vmax=0.1, vmin=0.01, cbar=False,
                ax=ax, linewidths=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Week of year")
    ax.set_ylabel(f"{period_size}-year period")
    fig.tight_layout()
    return fig


def plot_coalition_area(
    hits: pd.DataFrame,
    *,
    year_start: int = 1945,
    figsize: tuple = (12, 4),
    title: str = "Who asks about the Court of Audit?",
) -> plt.Figure:
    """Stacked area chart of coalition status shares over time."""
    dfs = hits[(hits["house"] == "commons") & (hits["date"].dt.year > year_start)]
    shares = (
        dfs.groupby(dfs["date"].dt.year)["coalition_status"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=figsize)
    shares.plot(kind="area", ax=ax, alpha=0.75)
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.legend(title="", loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_topic_timeseries(
    resp: pd.DataFrame,
    dist_averaged: pd.DataFrame,
    topic_id: int,
    ks: dict,
    *,
    ewm_span: int = 20,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot partial-corr with hits (blue) vs. overall topic prevalence (red) for one topic.
    """
    label = f"{topic_id} {ks[topic_id]}"
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    series = resp[label].ewm(span=ewm_span).mean() if label in resp.columns else resp.get(str(topic_id))
    if series is None:
        raise KeyError(f"Topic {topic_id} not found in resp columns")

    series.index = pd.PeriodIndex(series.index, freq="Q").to_timestamp()
    series.plot(ax=ax, color="#4878CF", linewidth=1.5)

    dist_series = dist_averaged[topic_id].ewm(span=ewm_span).mean()
    dist_series.plot(ax=ax2, color="#D65F5F", linewidth=1.5, linestyle="--")

    ax.set_title(f"Topic {topic_id}: {ks[topic_id][:60]}", loc="left")
    ax.set_ylabel("Partial corr with hits", color="#4878CF")
    ax2.set_ylabel("Topic prevalence (corpus)", color="#D65F5F")
    ax.tick_params(axis="y", colors="#4878CF")
    ax2.tick_params(axis="y", colors="#D65F5F")
    fig.tight_layout()
    return fig


def plot_topic_bubble(
    scatter_df: pd.DataFrame,
    ks: dict,
    *,
    top_n_labels: int = 10,
    figsize: tuple = (12, 8),
    title: str = "Topic landscape: variance vs. mean partial correlation with hits",
) -> plt.Figure:
    """
    Bubble chart: x=variance, y=mean partial-corr, size=max partial-corr,
    colour=corpus prominence (topic_z).  Labels the top_n_labels topics.
    """
    df = scatter_df.copy()
    df["topic_id"] = df.index
    df["label"] = df["topic_id"].map(ks).str[:40]
    size_norm = (df["max_corr"] - df["max_corr"].min()) / (df["max_corr"].max() - df["max_corr"].min())
    df["bubble"] = 10 + 200 * size_norm

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        df["variance"], df["mean_corr"],
        s=df["bubble"], c=df["topic_z"],
        cmap="Blues_r", alpha=0.7, linewidths=0.4, edgecolors="grey"
    )
    plt.colorbar(sc, ax=ax, label="Corpus prominence (log2-softmax z)")

    top = df.nlargest(top_n_labels, "max_corr")
    for _, row in top.iterrows():
        ax.annotate(row["label"], (row["variance"], row["mean_corr"]),
                    fontsize=7, alpha=0.9,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Variance of partial correlation over time")
    ax.set_ylabel("Mean partial correlation with hits")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def plot_topic_heatmap(
    resp: pd.DataFrame,
    topic_ids: list[int],
    ks: dict,
    *,
    ewm_span: int = 12,
    figsize: tuple = (14, 6),
    title: str = "Partial correlation heatmap (selected topics)",
) -> plt.Figure:
    """
    Heatmap of EWM-smoothed partial correlations for selected topics over time.
    """
    labels = [f"{t} {ks[t]}" for t in topic_ids if f"{t} {ks[t]}" in resp.columns]
    subset = resp[labels].ewm(span=ewm_span).mean()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(subset.T, ax=ax, cmap="RdBu_r", center=0,
                xticklabels=max(1, len(subset) // 20),
                yticklabels=[l.split(",")[0] for l in labels],
                cbar_kws={"shrink": 0.6})
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig
