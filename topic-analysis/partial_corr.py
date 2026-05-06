"""
topic_partial_time_fast_sampling_progress.py

High-dimensional partial correlation estimation for LDA topic proportions
over time using nodewise Lasso regression with optional sampling per year
and progress tracking.

Author: Your Name
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ---------------------------------------------------------------------
# CLR TRANSFORM
# ---------------------------------------------------------------------

def clr_transform(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Centered log-ratio (CLR) transform for compositional data."""
    X = X + eps
    logX = np.log(X)
    row_means = logX.mean(axis=1, keepdims=True)
    return logX - row_means


# ---------------------------------------------------------------------
# NODEWISE RESIDUALS (optimized)
# ---------------------------------------------------------------------

def _nodewise_residuals_fast(X: np.ndarray, target_index: int, alpha: float = 0.05) -> np.ndarray:
    """Compute residuals using Lasso with boolean masks and warm_start."""
    n_topics = X.shape[1]
    mask = np.ones(n_topics, dtype=bool)
    mask[target_index] = False
    X_others = X[:, mask]
    y = X[:, target_index]

    model = Lasso(alpha=alpha, max_iter=5000, warm_start=True)
    model.fit(X_others, y)
    return y - model.predict(X_others)


def compute_residuals_matrix_fast(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Compute residuals for all topics."""
    n_topics = X.shape[1]
    residuals = np.zeros_like(X)
    for idx in range(n_topics):
        residuals[:, idx] = _nodewise_residuals_fast(X, idx, alpha=alpha)
    return residuals


# ---------------------------------------------------------------------
# ROLLING WINDOW PARTIAL CORRELATION (all pairs, progress)
# ---------------------------------------------------------------------

def rolling_partial_corr_fast(
    df: pd.DataFrame,
    date_col: str,
    topic_cols: list,
    topic_pairs: list,
    window: int = 5,
    min_obs: int = 1000,
    alpha: float = 0.05,
    max_speeches_per_year: int = None
) -> pd.DataFrame:
    """Compute rolling partial correlations for multiple pairs with progress bars."""
    dates = pd.to_datetime(df[date_col]).values
    years_all = pd.Series(dates).dt.year.values
    X_all = df[topic_cols].values
    years = np.sort(np.unique(years_all))
    half_window = window // 2
    results = []

    for year in tqdm(years, desc="Years", unit="yr"):
        mask = (years_all >= year - half_window) & (years_all <= year + half_window)
        X_window = X_all[mask]
        n_obs_window = X_window.shape[0]
        if n_obs_window < min_obs:
            continue

        if max_speeches_per_year is not None and n_obs_window > max_speeches_per_year:
            indices = np.random.choice(n_obs_window, size=max_speeches_per_year, replace=False)
            X_window = X_window[indices]
            n_obs_window = X_window.shape[0]

        X_clr = clr_transform(X_window)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clr)
        residuals = compute_residuals_matrix_fast(X_scaled, alpha=alpha)

        for t1, t2 in tqdm(topic_pairs, desc=f"Pairs in {year}", leave=False):
            i = topic_cols.index(t1)
            j = topic_cols.index(t2)
            score = np.corrcoef(residuals[:, i], residuals[:, j])[0, 1]
            results.append({
                "year": year,
                "topic_i": t1,
                "topic_j": t2,
                "partial_corr": score,
                "n_obs": n_obs_window
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# ROLLING WINDOW PARTIAL CORRELATION (single pair, progress)
# ---------------------------------------------------------------------

def rolling_partial_corr_pair(
    df: pd.DataFrame,
    date_col: str,
    topic_cols: list,
    pair: tuple,
    window: int = 5,
    min_obs: int = 1000,
    alpha: float = 0.05,
    max_speeches_per_year: int = None
) -> pd.DataFrame:
    """Compute rolling partial correlation for a single pair with year progress."""
    dates = pd.to_datetime(df[date_col]).values
    years_all = pd.Series(dates).dt.year.values
    X_all = df[topic_cols].values
    years = np.sort(np.unique(years_all))
    half_window = window // 2
    t1, t2 = pair
    i = topic_cols.index(t1)
    j = topic_cols.index(t2)
    results = []

    for year in tqdm(years, desc="Years", unit="yr"):
        mask = (years_all >= year - half_window) & (years_all <= year + half_window)
        X_window = X_all[mask]
        n_obs_window = X_window.shape[0]
        if n_obs_window < min_obs:
            continue

        if max_speeches_per_year is not None and n_obs_window > max_speeches_per_year:
            indices = np.random.choice(n_obs_window, size=max_speeches_per_year, replace=False)
            X_window = X_window[indices]
            n_obs_window = X_window.shape[0]

        X_clr = clr_transform(X_window)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clr)

        def residual(col_idx):
            n_topics = X_scaled.shape[1]
            mask_col = np.ones(n_topics, dtype=bool)
            mask_col[col_idx] = False
            X_others = X_scaled[:, mask_col]
            y = X_scaled[:, col_idx]
            model = Lasso(alpha=alpha, max_iter=5000, warm_start=True)
            model.fit(X_others, y)
            return y - model.predict(X_others)

        r_i = residual(i)
        r_j = residual(j)
        score = np.corrcoef(r_i, r_j)[0, 1]

        results.append({
            "year": year,
            "topic_i": t1,
            "topic_j": t2,
            "partial_corr": score,
            "n_obs": n_obs_window
        })

    return pd.DataFrame(results)
