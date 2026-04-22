# utils_feature_selection.py
# =====================================================================
# Variable selection utilities used before model training:
#
# - Config-driven variable selection (fetch_series_config_for_variable_selection)
# - Missing value handling (drop_high_missing, time_interpolate)
# - Simple correlation-based filtering (corr_based_filter)
# - Multicollinearity handling via VIF (compute_vif, vif_based_filter)
# - Variable-ranking mechanisms:
#    * Mutual Information (MI)
#    * ElasticNetCV coefficients
#    * XGBoost + SHAP
#
# These rankings are later combined into a composite score in the main
# feature selection script.
# =====================================================================

import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

from settings import RANDOM_STATE, N_JOBS


# =====================================================================
# Missing value handling
# =====================================================================

def drop_high_missing(
    df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """
    Drop variables with missingness above a specified threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (independent variables only).
    threshold : float
        Fraction (0–1). Columns with missing percentage > threshold are dropped.

    Returns
    -------
    (pd.DataFrame, list, dict)
        - DataFrame with high-missing columns removed.
        - List of dropped column names.
        - Dict of {column_name: missing_percentage} for all columns.
    """
    miss_pct = df.isna().mean() * 100.0
    threshold_pct = threshold * 100.0

    to_drop = miss_pct[miss_pct > threshold_pct].index.tolist()
    df_clean = df.drop(columns=to_drop)

    return df_clean, to_drop, miss_pct.to_dict()


def time_interpolate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform time-based interpolation to fill missing values smoothly along
    the time axis.

    Assumes a DatetimeIndex or PeriodIndex with monotonic order.
    """
    df_interpolated = df.interpolate(method="time")
    df_interpolated = df_interpolated.ffill()
    df_interpolated = df_interpolated.bfill()
    return df_interpolated

# =====================================================================
# Correlation based filter
# =====================================================================

def corr_based_filter(
    df: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple correlation-based filter.

    Selection rule:
      - Keep variables where abs(corr(variable, target)) >= threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame including target and feature columns (numeric).
    target_col : str
        Target column name in df.
    threshold : float
        Minimum absolute correlation required (0–1).

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        - df_keep:
            ["Variable_Name", "Correlation_Value"]
        - df_reject:
            same schema, variables that did not pass the threshold.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    corr_series = df.corr(numeric_only=True)[target_col]
    corr_series = corr_series.drop(labels=[target_col], errors="ignore")

    keep_records = []
    reject_records = []

    for var_name, corr_val in corr_series.items():
        if pd.isna(corr_val):
            continue
        record = [var_name, float(round(corr_val, 3))]
        if abs(corr_val) >= threshold:
            keep_records.append(record)
        else:
            reject_records.append(record)

    cols = ["Variable_Name", "Correlation_Value"]
    df_keep = pd.DataFrame(keep_records, columns=cols)
    df_reject = pd.DataFrame(reject_records, columns=cols)

    return df_keep, df_reject

# =====================================================================
# VIF utilities
# =====================================================================

def compute_vif(
    X: pd.DataFrame,
) -> pd.Series:
    """
    Compute Variance Inflation Factor (VIF) for each numeric column.

    Returns
    -------
    pd.Series
        Series with column names as index and VIF values as floats.
    """
    X_j = X.select_dtypes(include=[np.number]).copy()

    if X_j.shape[1] == 0:
        return pd.Series(dtype=float)

    for c in X_j.columns:
        if X_j[c].std(ddof=0) == 0:
            X_j[c] = X_j[c] + np.random.normal(0, 1e-8, size=len(X_j))

    vals = []
    for i in range(X_j.shape[1]):
        try:
            v = variance_inflation_factor(X_j.values, i)
            if not np.isfinite(v):
                v = np.inf
        except Exception:
            v = np.inf
        vals.append(v)

    vif = pd.Series(vals, index=X_j.columns, dtype=float)
    return vif


def vif_based_filter(
    X: pd.DataFrame,
    vif_max: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Iteratively drop variables with the highest VIF until all remaining
    variables have VIF <= vif_max (or until only one remains).
    """
    X = X.select_dtypes(include=[np.number]).copy()

    if X.shape[1] <= 1:
        return X, compute_vif(X).sort_values(ascending=True)

    max_iter = max(2 * X.shape[1], 10)
    iters = 0

    while True:
        iters += 1
        vif = compute_vif(X)

        if vif.empty or X.shape[1] <= 1:
            return X, vif.sort_values(ascending=True)

        worst = vif.idxmax()
        worst_v = vif.loc[worst]

        if np.isfinite(worst_v) and worst_v <= vif_max:
            return X, vif.sort_values(ascending=True)

        X = X.drop(columns=[worst])

        if iters >= max_iter:
            return X, compute_vif(X).sort_values(ascending=True)

# =====================================================================
# Ranking utilities
# =====================================================================

def mi_rank(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Rank variables using Mutual Information (MI) with the target.
    """
    scores = mutual_info_regression(X, y.values, random_state=RANDOM_STATE)
    mi = pd.Series(scores, index=X.columns)
    return mi.sort_values(ascending=False)


def elasticnet_rank(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Rank variables using an ElasticNetCV linear model.
    """
    scaler = None
    # Standardize X for stable ElasticNet optimisation
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    enet = ElasticNetCV(
        cv=TimeSeriesSplit(n_splits=5),
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_iter=20_000,
    )

    enet.fit(Xs, y.values)
    coefs = pd.Series(np.abs(enet.coef_), index=X.columns)
    return coefs.sort_values(ascending=False)


def xgb_shap_rank(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Rank variables using XGBoost + SHAP.
    """
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X, y.values)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_importance = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(shap_importance, index=X.columns)
    return imp.sort_values(ascending=False)


def normalize_importance(
    series: pd.Series,
    feature_names: List[str],
) -> pd.Series:
    """
    Reindex and normalize an importance/score series to the range [0, 1].
    """
    s = series.reindex(feature_names).fillna(0.0)
    max_val = s.max()

    if max_val is None or max_val <= 0 or not np.isfinite(max_val):
        return pd.Series(0.0, index=feature_names)

    return s / max_val
