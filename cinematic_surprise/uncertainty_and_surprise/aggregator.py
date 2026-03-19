"""
uncertainty_and_surprise/aggregator.py

Post-hoc aggregation of raw per-second surprise and uncertainty values.

Aggregation strategy (from design document):
    - Z-score within channel, within film, within participant
    - Interaction per channel = z(surprise_c) * z(uncertainty_c)
    - Aggregate = z_film( (1/C) * sum_c[ z_film(signal_c) ] )
    - interaction_combined uses Option A: per-channel interaction first,
      then average and z-score. NOT equivalent to z(S_combined)*z(U_combined).

Z-scoring convention:
    mean and std computed across ALL seconds of the film for each column.
    This answers: "relative to how this channel behaved during this film,
    how surprising/uncertain was this second?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cinematic_surprise.config import CHANNELS


def zscores_film(series: pd.Series) -> pd.Series:
    """
    Z-score a series across the film (all seconds).
    Returns zero-series if std is zero (constant signal).
    """
    mu  = series.mean()
    std = series.std()
    if std < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / std


def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-channel interaction columns.

    interaction_c = z_film(surprise_c) * z_film(uncertainty_c)

    This is computed before any cross-channel averaging, preserving
    the channel-level co-occurrence structure (Option A from design doc).
    Not equivalent to z(surprise_combined) * z(uncertainty_combined).
    """
    for ch in CHANNELS:
        s_col = f"surprise_{ch}"
        u_col = f"uncertainty_{ch}"
        i_col = f"interaction_{ch}"

        if s_col in df.columns and u_col in df.columns:
            z_s = zscores_film(df[s_col])
            z_u = zscores_film(df[u_col])
            df[i_col] = z_s * z_u
        else:
            df[i_col] = np.nan

    return df


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three aggregate columns.

    Formula (from design document, Table 14):

        surprise_combined    = z_film( (1/C) * sum_c[ z_film(surprise_c) ] )
        uncertainty_combined = z_film( (1/C) * sum_c[ z_film(uncertainty_c) ] )
        interaction_combined = z_film( (1/C) * sum_c[ z_film(S_c) * z_film(U_c) ] )

    The double z-score ensures:
        1. Inner z-score: each channel contributes equally (scale-invariant)
        2. Outer z-score: the combined signal is normalised to std=1 across
           the film, ready for comparison with BOLD data without further
           normalisation.

    C = 12 channels (all entries in config.CHANNELS).
    """
    # ── surprise_combined ─────────────────────────────────────────────────
    s_cols = [f"surprise_{ch}" for ch in CHANNELS if f"surprise_{ch}" in df.columns]
    if s_cols:
        z_surprises = pd.concat(
            [zscores_film(df[c]) for c in s_cols], axis=1
        )
        df["surprise_combined"] = zscores_film(z_surprises.mean(axis=1))
    else:
        df["surprise_combined"] = np.nan

    # ── uncertainty_combined ──────────────────────────────────────────────
    u_cols = [f"uncertainty_{ch}" for ch in CHANNELS if f"uncertainty_{ch}" in df.columns]
    if u_cols:
        z_uncertainties = pd.concat(
            [zscores_film(df[c]) for c in u_cols], axis=1
        )
        df["uncertainty_combined"] = zscores_film(z_uncertainties.mean(axis=1))
    else:
        df["uncertainty_combined"] = np.nan

    # ── interaction_combined (Option A) ───────────────────────────────────
    i_cols = [f"interaction_{ch}" for ch in CHANNELS if f"interaction_{ch}" in df.columns]
    if i_cols:
        # interaction_c already = z(S_c) * z(U_c), computed in compute_interactions
        interaction_mean = df[i_cols].mean(axis=1)
        df["interaction_combined"] = zscores_film(interaction_mean)
    else:
        df["interaction_combined"] = np.nan

    return df


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full post-hoc aggregation pipeline on a raw results DataFrame.

    Steps:
        1. Compute per-channel interaction columns
        2. Compute three aggregate columns

    Args:
        df : DataFrame with raw surprise_* and uncertainty_* columns,
             one row per second.

    Returns:
        DataFrame with interaction_* and aggregate columns added.
        Column order: metadata | surprise×12 | uncertainty×12 |
                      interaction×12 | 3 aggregates = 48 columns total
                      (plus any additional metadata columns).
    """
    df = compute_interactions(df)
    df = compute_aggregates(df)
    return df
