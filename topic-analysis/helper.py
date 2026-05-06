"""
helper.py — Data loading and visualisation helpers for analysis.ipynb.

Keeps notebook cells focused on narrative and figures.  Import with:
    from helper import *
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.special import softmax
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_references(path: str | Path) -> pd.DataFrame:
    """Load references.csv, parse dates, drop rows without a date."""
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"references: {len(df):,} rows  |  {df.date.min().date()} – {df.date.max().date()}")
    return df


def load_lda(lda_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load 300-topic LDA outputs from *lda_dir*.

    Returns (dist, meta, topic_label, ks).
    dist has integer topic columns 0-299 plus a 'date' column.
    ks maps topic_id → comma-joined top-10 keywords.
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


def load_averaged_topics(lda_dir_500: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load averaged_topics.csv from the 500-topic model.

    The CSV has per-speaker rows (member_or_speaker, party-ref, date, 0..499).
    We group by date and average across speakers so dfd is one row per date.
    Returns (dfd, ks500) where dfd is DatetimeIndex × string topic columns ("0".."499").
    """
    lda_dir_500 = Path(lda_dir_500)
    raw = pd.read_csv(lda_dir_500 / "averaged_topics.csv")
    raw["date"] = pd.to_datetime(raw["date"])

    # keep only numeric topic columns (named "0", "1", …)
    topic_cols = [c for c in raw.columns if str(c).isdigit()]
    dfd = raw.groupby("date")[topic_cols].mean()

    keys_path = lda_dir_500 / "topic-keys.txt"
    if keys_path.exists():
        keys = pd.read_csv(keys_path, sep="\t", header=None)
        ks500 = dict(zip(keys[0], keys[2].str.split(" ").str[:10].str.join(", ")))
    else:
        ks500 = {i: str(i) for i in range(500)}

    print(f"averaged_topics: {len(dfd):,} dates  |  {len(topic_cols)} topics")
    return dfd, ks500


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


def load_all(references_path, lda_dir, noun_total_path, cabinet_path=None):
    """Convenience loader for the 300-topic / references analysis.
    Returns (refs, dist, meta, ks, total_nouns, cab).
    """
    refs = load_references(references_path)
    dist, meta, topic_label, ks = load_lda(lda_dir)
    total_nouns = load_noun_totals(noun_total_path)
    cab = load_cabinets(cabinet_path) if cabinet_path else None
    return refs, dist, meta, ks, total_nouns, cab


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def to_half_year(s: pd.Series) -> pd.Series:
    """Map a datetime Series to 'YYYY-H1' / 'YYYY-H2' strings."""
    return s.dt.year.astype(str) + np.where(s.dt.month <= 6, "-H1", "-H2")


def add_periods(refs: pd.DataFrame, dist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add 'period' column to both DataFrames and flag dist rows that appear in refs."""
    refs = refs.copy()
    dist = dist.copy()
    refs["period"] = to_half_year(refs["date"])
    dist["period"] = to_half_year(dist["date"])
    dist["in_refs"] = dist["date"].isin(set(refs["date"]))
    print(f"flagged: {dist['in_refs'].sum():,} / {len(dist):,} dist rows")
    return refs, dist


def relative_frequency(refs: pd.DataFrame, total_nouns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily reference count and normalise by total noun count.
    Returns a daily DataFrame with columns: date, f, rf.
    """
    date_to_count = dict(zip(pd.to_datetime(total_nouns["date"]), total_nouns["count"]))
    daily = refs.groupby("date").size().reset_index(name="f")
    daily = daily[daily["date"].dt.year > 1848]
    daily["rf"] = daily["f"] / daily["date"].map(date_to_count)
    return daily


def quarterly_stats(daily: pd.DataFrame, lowess_frac: float = 0.05) -> pd.DataFrame:
    """Aggregate daily rf to quarters; adds LOESS-smoothed loess_entropy and loess_sum."""
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


def classify_coalition(refs: pd.DataFrame, cab: pd.DataFrame) -> pd.DataFrame:
    """Add 'coalition_status': 'government' | 'coalition party' | 'opposition party'."""
    refs = refs.copy()
    refs["party_code"] = refs["party-ref"].str.split(".").str[-1].str.lower()
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

    refs["coalition_status"] = refs.apply(_classify, axis=1)
    return refs.drop(columns=["party_code"])


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL CORRELATION (references ↔ topics)
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_COLS = list(range(300))
MIN_OBS = 30


def build_dam(dist: pd.DataFrame, refs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align topic distributions with reference counts per date.
    Returns (dam, refs_day).
    """
    daily_refs = refs.groupby("date").size().rename("Y")
    dam = dist.set_index("date")[TOPIC_COLS].copy()
    dam["Y"] = dam.index.map(daily_refs)
    dam = dam.dropna(subset=["Y"])
    refs_day = dam.pop("Y")
    print(f"dam: {len(dam):,} docs  |  {dam.index.nunique():,} unique dates")
    return dam, refs_day


def pcorr_over_time(
    dam: pd.DataFrame,
    refs_day: pd.Series,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """
    Quarterly partial correlations between each topic and reference frequency.
    Returns long DataFrame: period, topic, partial_corr.
    """
    df = dam.copy()
    df["Y"] = refs_day
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

def trending_topics(resp: pd.DataFrame, n: int = 15, ewm_span: int = 12) -> pd.Series:
    """Top-N topics by Mann-Kendall trend slope in their EWM partial-correlation series."""
    try:
        from pymannkendall import original_test as pmk
    except ImportError:
        raise ImportError("pip install pymannkendall")
    return resp.ewm(span=ewm_span).mean().apply(lambda x: pmk(x).slope).nlargest(n)


def topic_scatter_data(resp: pd.DataFrame, dist_averaged: pd.DataFrame, ewm_span: int = 12) -> pd.DataFrame:
    """Bubble-chart data: variance, mean, max partial-corr + corpus prominence z-score."""
    from scipy.stats import zscore as _zscore
    zscores_raw = (
        dist_averaged.ewm(span=ewm_span).mean()
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
# CONTROL TOPICS (500-topic model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_control_topics(
    dfd: pd.DataFrame,
    control_topics: list[int],
    ks500: dict,
    *,
    ewm_span: int = 8,
    figsize: tuple = (12, 4),
    title: str = "Evaluation / oversight topics over time",
) -> plt.Figure:
    """Line chart of EWM-smoothed topic proportions for a set of control topics."""
    annual = dfd.groupby(dfd.index.year).mean().ewm(span=ewm_span).mean()
    fig, ax = plt.subplots(figsize=figsize)
    for t in control_topics:
        col = str(t)
        if col in annual.columns:
            label = f"{t}: {ks500.get(t, '')[:40]}"
            annual[col].ewm(span=ewm_span).mean().plot(ax=ax, label=label)
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def verantwoordingsdag_test(
    dfd: pd.DataFrame,
    topic_ids: list[int],
    year_range: range | None = None,
) -> pd.DataFrame:
    """
    Test whether topic proportions are significantly elevated on Verantwoordingsdag
    (third Wednesday of May) relative to the rest of the year.

    Returns a DataFrame with columns: topic, mean_vdag, mean_other, t_stat, p_value.
    """
    from scipy.stats import ttest_ind

    if year_range is None:
        year_range = range(dfd.index.year.min(), dfd.index.year.max() + 1)

    third_wednesdays = []
    for y in year_range:
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31", freq="D")
        wednesdays = may[may.dayofweek == 2]
        if len(wednesdays) >= 3:
            third_wednesdays.append(wednesdays[2])

    vdag_idx = dfd.index.normalize().isin([d.normalize() for d in third_wednesdays])
    results = []
    for t in topic_ids:
        col = str(t)
        if col not in dfd.columns:
            continue
        v = dfd.loc[vdag_idx, col].dropna()
        o = dfd.loc[~vdag_idx, col].dropna()
        stat, p = ttest_ind(v, o, equal_var=False)
        results.append({"topic": t, "mean_vdag": v.mean(), "mean_other": o.mean(),
                        "t_stat": stat, "p_value": p})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION — REFERENCES OVER TIME
# ─────────────────────────────────────────────────────────────────────────────

def plot_freq_entropy(
    quarterly: pd.DataFrame,
    *,
    war_shade: bool = True,
    title: str = "References to the Court of Audit over time",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Dual-axis chart: summed relative frequency (filled) + entropy (line)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax2.fill_between(quarterly["quarter"], quarterly["loess_sum"],
                     color="#D65F5F", alpha=0.25, label="Summed frequency")
    ax.plot(quarterly["quarter"], quarterly["loess_entropy"],
            color="#4878CF", linewidth=1.5, label="Entropy")
    for spine, a, color in [("left", ax, "#4878CF"), ("right", ax2, "#D65F5F")]:
        a.spines[spine].set_color(color)
        a.tick_params(axis="y", colors=color)
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
    refs: pd.DataFrame,
    *,
    year_start: int = 1940,
    period_size: int = 5,
    figsize: tuple = (14, 6),
    title: str = "Reference distribution by week-of-year (5-year bins)",
) -> plt.Figure:
    """Heatmap of weekly reference distribution, binned into 5-year periods."""
    monthly = (
        refs
        .groupby([refs.date.dt.year // period_size * period_size,
                  refs.date.dt.isocalendar().week])
        .size().unstack().fillna(0)
    )
    monthly = monthly.div(monthly.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(monthly[monthly.index > year_start], cmap="Blues",
                vmax=0.1, vmin=0.01, cbar=False, ax=ax, linewidths=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Week of year")
    ax.set_ylabel(f"{period_size}-year period")
    fig.tight_layout()
    return fig


def plot_coalition_area(
    refs: pd.DataFrame,
    *,
    year_start: int = 1945,
    figsize: tuple = (12, 4),
    title: str = "Who references the Court of Audit?",
) -> plt.Figure:
    """Stacked area chart of coalition status shares over time."""
    dfs = refs[(refs["house"] == "commons") & (refs["date"].dt.year > year_start)]
    shares = (
        dfs.groupby(dfs["date"].dt.year)["coalition_status"]
        .value_counts(normalize=True).unstack().fillna(0)
    )
    fig, ax = plt.subplots(figsize=figsize)
    shares.plot(kind="area", ax=ax, alpha=0.75)
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.legend(title="", loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION — TOPIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_topic_timeseries(
    resp: pd.DataFrame,
    dist_averaged: pd.DataFrame,
    topic_id: int,
    ks: dict,
    *,
    ewm_span: int = 20,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Partial-corr with references (blue) vs. overall topic prevalence (red) for one topic."""
    label = f"{topic_id} {ks[topic_id]}"
    if label not in resp.columns:
        raise KeyError(f"Topic {topic_id} not in resp columns — check ks dict")
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    series = resp[label].ewm(span=ewm_span).mean()
    series.index = pd.PeriodIndex(series.index, freq="Q").to_timestamp()
    series.plot(ax=ax, color="#4878CF", linewidth=1.5)
    dist_averaged[topic_id].ewm(span=ewm_span).mean().plot(ax=ax2, color="#D65F5F",
                                                            linewidth=1.5, linestyle="--")
    ax.set_title(f"Topic {topic_id}: {ks[topic_id][:60]}", loc="left")
    ax.set_ylabel("Partial corr with references", color="#4878CF")
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
    title: str = "Topic landscape: variance vs. mean partial correlation with references",
) -> plt.Figure:
    """Bubble chart: x=variance, y=mean corr, size=peak corr, colour=corpus prominence."""
    df = scatter_df.copy()
    df["topic_id"] = df.index
    df["label"] = df["topic_id"].map(ks).str[:40]
    size_norm = (df["max_corr"] - df["max_corr"].min()) / (df["max_corr"].max() - df["max_corr"].min())
    df["bubble"] = 10 + 200 * size_norm
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(df["variance"], df["mean_corr"], s=df["bubble"], c=df["topic_z"],
                    cmap="Blues_r", alpha=0.7, linewidths=0.4, edgecolors="grey")
    plt.colorbar(sc, ax=ax, label="Corpus prominence (log2-softmax z)")
    for _, row in df.nlargest(top_n_labels, "max_corr").iterrows():
        ax.annotate(row["label"], (row["variance"], row["mean_corr"]),
                    fontsize=7, alpha=0.9, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Variance of partial correlation over time")
    ax.set_ylabel("Mean partial correlation with references")
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
    """Heatmap of EWM-smoothed partial correlations for selected topics over time."""
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
