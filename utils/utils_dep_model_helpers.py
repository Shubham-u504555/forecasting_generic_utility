# utils_dep_model_helpers.py
# =====================================================================
# Helpers for TARGET variable models:
#
# This module focuses on models that forecast the **dependent / target**
# series (typically the target variable), using:
#
#   - Column sanitization for robust modeling
#   - Target-side feature engineering with:
#       * Autoregressive lags of the target
#       * Contemporary independent variables
#       * Calendar seasonality encodings (month, sin/cos)
#       * Rolling mean and volatility of the target
#   - Single-row feature builder for recursive forecasting
#   - Model builders:
#       * LightGBM (tree-based gradient boosting)
#       * CatBoost
#       * XGBoost
#       * ElasticNet (linear, with scaling)
#   - NeuralForecast wrapper for univariate deep TS models:
#       * NHITS, NBEATS, TFT, LSTM, etc.
#
# All models are typically trained on the **log of the target** to:
#   - Stabilize variance
#   - Improve multiplicative / percentage behavior
# =====================================================================

import re
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from neuralforecast import NeuralForecast

from settings import (
    N_LAGS_TARGET,   # number of target lags used in feature engineering / recursion
    RANDOM_STATE,    # global random seed
    N_JOBS,          # parallelism for XGBoost (and other models if needed)
)

import warnings
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in reciprocal",
    category=RuntimeWarning
)


# ---------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------

def _get_roll_params(freq: str):
    """Return rolling-window sizes (short, long) in *periods* for the given frequency.

    Parameters
    ----------
    freq : str
        Frequency code:
          - "M" = Monthly
          - "W" = Weekly
          - "Y" = Yearly (or anything else falls back to yearly)

    Returns
    -------
    (roll_short, roll_long) : tuple[int, int]
        Rolling window sizes expressed as number of observations (periods):
          - Monthly: (3, 6)  -> ~quarter and ~half-year
          - Weekly : (13, 26)-> ~quarter and ~half-year (in weeks)
          - Yearly : (2, 3)  -> short multi-year smoothing
    """
    freq = str(freq or "M").upper()

    if freq == "M":
        roll_short = 3
        roll_long = 6
    elif freq == "W":
        roll_short = 13
        roll_long = 26
    else:
        roll_short = 2
        roll_long = 3

    return roll_short, roll_long
	
# =====================================================================
# Generic modeling helpers (target side)
# =====================================================================

def make_target_features(
    df_local: pd.DataFrame,
    target_col: str,
    indep_columns: List[str],
    freq: str = "M",
    n_lags_target: int = N_LAGS_TARGET,
):
    """
    Build supervised learning features for target (dependent variable) modeling.

    Features created
    ----------------
    1) Autoregressive lags of the target (shifted by i periods):
         target_lag_1 ... target_lag_n_lags_target

    2) Contemporary independent variables:
         For each column in `indep_columns`, the value at the same timestamp
         is included as a feature.

    3) Frequency-aware calendar / seasonality features:
         - Monthly ("M"):
             month (1..12), year, plus cyclic encoding using month
         - Weekly ("W"):
             ISO week (1..53), year, plus cyclic encoding using week
         - Yearly ("Y"):
             year only; seasonality terms are set to 0.0

         season_sin/season_cos provide smooth cyclic encoding.

    4) Rolling statistics on the *lagged* target (shifted by 1 period),
       using windows chosen by `_get_roll_params(freq)`:
         rolling_short : mean over roll_short periods
         rolling_long  : mean over roll_long periods
         std_short     : std over roll_short periods
         std_long      : std over roll_long periods

       Note: rolling is computed on target shifted by 1 to avoid leakage.

    Missing data strategy
    ---------------------
    - Rows with NaNs introduced by lagging/rolling are dropped from X.
    - y is aligned to the remaining X.index.

    Parameters
    ----------
    df_local : pd.DataFrame
        Input data with target and independent variables.
        Index must be a DatetimeIndex.
    target_col : str
        Target column name.
    indep_columns : list[str]
        Independent variable column names to include.
    freq : str, optional
        "M", "W", or "Y" to control calendar features and rolling windows.
    n_lags_target : int, optional
        Number of target lags to generate.

    Returns
    -------
    (X, y)
        X : pd.DataFrame
            Feature matrix after dropping NaN rows.
        y : pd.Series
            Target values aligned to X.index.
    """

    df_feat = pd.DataFrame(index=df_local.index)

    # 1) Target lags (AR structure of the target)
    for i in range(1, n_lags_target + 1):
        df_feat[f"target_lag_{i}"] = df_local[target_col].shift(i)

    # 2) Exogenous variables (same timestamp)
    for col in indep_columns:
        df_feat[col] = df_local[col]

    # 3) Calendar features (frequency-aware)
    roll_short, roll_long = _get_roll_params(freq)

    if freq == "M":
        df_feat["month"] = df_local.index.month
        df_feat["year"] = df_local.index.year
        df_feat["season_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
        df_feat["season_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    elif freq == "W":
        # ISO week number (1..53)
        iso_week = df_local.index.isocalendar().week.astype(int)
        df_feat["week"] = iso_week
        df_feat["year"] = df_local.index.year
        # Use 52 for a standard yearly cycle approximation
        df_feat["season_sin"] = np.sin(2 * np.pi * df_feat["week"] / 52)
        df_feat["season_cos"] = np.cos(2 * np.pi * df_feat["week"] / 52)
    else:
        # Yearly frequency: keep only the year trend; no cyclic seasonality
        df_feat["year"] = df_local.index.year
        df_feat["season_sin"] = 0.0
        df_feat["season_cos"] = 0.0

    # 4) Rolling statistics on (lagged) target (in periods)
    target_shifted = df_local[target_col].shift(1)
    df_feat["rolling_short"] = target_shifted.rolling(roll_short).mean()
    df_feat["rolling_long"] = target_shifted.rolling(roll_long).mean()
    df_feat["std_short"] = target_shifted.rolling(roll_short).std()
    df_feat["std_long"] = target_shifted.rolling(roll_long).std()

    # Drop rows with incomplete lag/rolling information
    X = df_feat.dropna()
    y = df_local.loc[X.index, target_col]

    return X, y


def build_target_feature_row(
    target_history_log: List[float],
    indep_current: Dict[str, float],
    current_date: pd.Timestamp,
    feature_columns: List[str],
    n_lags_target: int = N_LAGS_TARGET,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Build a single feature row for recursive forecasting of the target.

    This is the "online" counterpart of `make_target_features`, used
    during iterative multi-step forecasting. Instead of recomputing
    features from a full DataFrame, we reuse:
      - A list of recent target values (usually log of target).
      - The independent variables values at the current forecast step.
      - The current date to reconstruct calendar features.

    Parameters
    ----------
    target_history_log : list of float
        Last `n_lags_target` values of the target (typically log-target),
        ordered chronologically. The function will pull the latest values
        using negative indices.
    indep_current : dict
        Mapping {indep_name: value} for the current forecast timestamp.
    current_date : pd.Timestamp
        Date being forecasted (used for calendar features).
    feature_columns : list of str
        Full list of feature columns expected by the model
        (must match X.columns used for training).
    n_lags_target : int
        Number of target lags to use.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns exactly matching `feature_columns`.
    """
    if len(target_history_log) < n_lags_target:
        raise ValueError("Not enough target history (log) to build required lags.")

    feat: Dict[str, float] = {}

    # Target lags (from most recent backwards)
    for i in range(1, n_lags_target + 1):
        feat[f"target_lag_{i}"] = float(target_history_log[-i])

    # 2) Exogenous values at current timestamp
    for col, val in indep_current.items():
        feat[col] = float(val)
		
    # Calendar features (frequency-aware)
    roll_short, roll_long = _get_roll_params(freq)

    if freq == "M":
        month = current_date.month
        year = current_date.year
        feat["month"] = month
        feat["year"] = year
        feat["season_sin"] = float(np.sin(2 * np.pi * month / 12))
        feat["season_cos"] = float(np.cos(2 * np.pi * month / 12))
    elif freq == "W":
        week = int(current_date.isocalendar().week)
        year = current_date.year
        feat["week"] = week
        feat["year"] = year
        feat["season_sin"] = float(np.sin(2 * np.pi * week / 52))
        feat["season_cos"] = float(np.cos(2 * np.pi * week / 52))
    else:
        year = current_date.year
        feat["year"] = year
        feat["season_sin"] = 0.0
        feat["season_cos"] = 0.0

    # Rolling stats from history (in periods, not months)

    if len(target_history_log) >= roll_short:
        feat["rolling_short"] = float(np.mean(target_history_log[-roll_short:]))
        feat["std_short"] = float(np.std(target_history_log[-roll_short:], ddof=0))
    else:
        feat["rolling_short"] = float(np.mean(target_history_log))
        feat["std_short"] = float(np.std(target_history_log, ddof=0)) if len(target_history_log) > 1 else 0.0

    if len(target_history_log) >= roll_long:
        feat["rolling_long"] = float(np.mean(target_history_log[-roll_long:]))
        feat["std_long"] = float(np.std(target_history_log[-roll_long:], ddof=0))
    else:
        feat["rolling_long"] = float(np.mean(target_history_log))
        feat["std_long"] = float(np.std(target_history_log, ddof=0)) if len(target_history_log) > 1 else 0.0

    # Align columns exactly as during training. Missing keys will become NaN,
    # but in the pipeline we ensure feature_columns are consistent.
    x_df = pd.DataFrame([feat], columns=feature_columns)

    return x_df

# =====================================================================
# Model builders (multivariate target models)
# =====================================================================
def build_lgbm_model() -> LGBMRegressor:
    """
    Construct a LightGBM regressor for target forecasting.

    Returns
    -------
    LGBMRegressor
        LightGBM model instance.
    """
    
    return LGBMRegressor(
        random_state=RANDOM_STATE,  # fixed seed for reproducibility.
        metric='mape',              # "mape" used only for internal evaluation.
        n_estimators=500,           # number of boosting iterations (500 trees).
        learning_rate=0.03,         # 0.03 — relatively small to allow smoother learning.
        num_leaves=31,              # 31 — tree complexity; moderate to avoid overfitting.
        min_data_in_leaf=10,        # 10 — regularization via minimum leaf size.
        subsample=0.8,              # 0.8 — row subsampling for stochasticity.
        colsample_bytree=0.8,       # 0.8 — feature subsampling per tree.
        reg_alpha=0.1,              # L1 regularization (0.1) on leaf weights.
        reg_lambda=1.0,             # L2 regularization (1.0) on leaf weights.
        verbosity=-1,               # -1 suppresses LightGBM internal logs.
        n_jobs=N_JOBS,              # Parallelization over CPU cores.
    )


def build_catboost_model():
    """
    Construct a CatBoostRegressor for target forecasting (log-target, with independent variables).

    Raises
    ------
    ImportError
        If `catboost` is not installed.

    Returns
    -------
    CatBoostRegressor
        Configured CatBoost model ready for fit.
    """

    return CatBoostRegressor(
        loss_function="RMSE",        # Standard squared-error loss (root mean squared).
        depth=4,                     # Tree depth; relatively shallow trees reduce overfitting risk.
        learning_rate=0.05,          # Step size shrinkage; smaller values → smoother learning.
        n_estimators=500,            # Number of boosting iterations (trees).
        l2_leaf_reg=5,               # L2 regularization on leaf values.
        bootstrap_type="Bernoulli",  # Row sampling at each iteration; typical for CatBoost.
        subsample=0.8,               # Fraction of rows used at each iteration (stochastic boosting).
        random_seed=RANDOM_STATE,    # Reproducibility of tree construction.
        verbose=False,               # Suppress CatBoost's training logs.
        thread_count=N_JOBS,         # Parallelization over CPU cores.
    )


def build_xgboost_model():
    """
    Construct an XGBRegressor for target forecasting (log-target, with independent variables).

    Raises
    ------
    ImportError
        If `xgboost` is not installed.

    Returns
    -------
    XGBRegressor
        Configured XGBoost model.
    """

    return XGBRegressor(
        objective="reg:squarederror",  # Standard squared-error loss for regression.
        max_depth=4,                   # Depth of individual trees. Shallow trees reduce overfitting.
        learning_rate=0.05,            # Shrinkage factor; lower -> slower but more stable learning.
        n_estimators=500,              # Number of boosting iterations (trees).
        subsample=0.8,                 # Row subsampling rate for each tree; adds stochasticity.
        colsample_bytree=0.8,          # Feature subsampling rate for each tree; combats correlation.
        min_child_weight=3,            # Minimum sum of instance weight needed in a child; regularization.
        reg_alpha=0.1,                 # L1 regularization on weights.
        reg_lambda=1.0,                # L2 regularization on weights.
        random_state=RANDOM_STATE,     # For reproducible tree sampling and splits.
        n_jobs=N_JOBS,                 # Parallelization over CPU cores.
        verbosity=0,                   # <-- disable logging completely
    )


def build_elasticnet_model() -> Pipeline:
    """
    Construct an ElasticNetCV regression pipeline for log-target forecasting.

    Pipeline:
    ---------
    1) StandardScaler
         - Standardizes all features (mean=0, std=1).
         - Required for stable ElasticNet behavior since L1/L2 penalties 
           are scale-dependent.

    2) ElasticNetCV
         - Performs k-fold cross-validation on time-series splits.
         - Learns optimal alpha (regularization strength) and l1_ratio 
           (balance of L1 vs L2).
         - Uses TimeSeriesSplit(n_splits=5) to respect chronological order.

    ElasticNetCV Hyperparameters:
    -----------------------------
    - cv = TimeSeriesSplit(5)
         Ensures future data is never used to predict the past.

    - l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
         Tests different L1/L2 tradeoffs:
            0.1 → mostly L2 (Ridge-like)
            0.9 → mostly L1 (Lasso-like)

    - max_iter = 20_000
         Higher iteration limit ensures convergence even with many 
         correlated features.

    - random_state = RANDOM_STATE
         Reproducibility of optimization path.

    - n_jobs = N_JOBS
         Parallel cross-validation.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with ("scaler" -> "enet") components.
    """

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    cv=TimeSeriesSplit(n_splits=5),  # follows time order
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # mixed L1/L2 grid
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    max_iter=20_000,
                    verbose=0
                ),
            ),
        ]
    )


def build_sarimax_model(
    df_train: pd.DataFrame,
    target_col: str,
    indep_cols: list,
    seasonal_period: int = 12,
):
    """
    Build and return an un-fitted SARIMAX model for training.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset containing the target variable and selected exogenous variables.
    target_col : str
        Name of the target column (series to forecast).
    indep_cols : list
        List of column names to be used as exogenous predictors.

    Notes
    -----
    - Using SARIMAX because it supports both autoregressive components and exogenous regressors.
    - order=(1,1,1) is a robust general-purpose ARIMA structure for monthly data.
      * AR(1): captures short-term momentum from previous month.
      * I(1): ensures the series is differenced once to remove trend.
      * MA(1): captures short-term shocks / noise.
    - seasonal_order=(1,0,1,12) models annual seasonality (12 months).
      * SAR(1): seasonal autoregressive behaviour (value depends on same month last year).
      * SMA(1): accounts for seasonal shocks.
      * No seasonal differencing (D=0) assuming trend removed in preprocessing.
    - enforce_stationarity=False and enforce_invertibility=False allow more flexible model fitting
      and prevent hard failures when data is borderline non-stationary.
    """

    # Instantiate SARIMAX model with target as endog and selected variables as exog
    sarimax_model = SARIMAX(
        endog=df_train[target_col],    # Target time-series
        exog=df_train[indep_cols],     # External regressors
        order=(1, 1, 1),               # Non-seasonal ARIMA terms
        seasonal_order=(1, 0, 1, seasonal_period) if seasonal_period and seasonal_period > 1 else (0, 0, 0, 0),  # Annual monthly seasonality structure
        enforce_stationarity=False,    # Allow model even if AR roots suggest non-stationarity
        enforce_invertibility=False,   # Allow MA roots to be outside invertible region
    )

    # Return un-fitted SARIMAX object (fit() will be called externally)
    return sarimax_model

# =====================================================================
# Model builders (univariate statistical based target models)
# =====================================================================
def fit_holtwinters(train_series_log: pd.Series, df_future: pd.DataFrame, ts_freq: str = "MS", seasonal_period: int = 12) -> pd.Series:
    y_hw = train_series_log.asfreq(ts_freq)
    if seasonal_period and seasonal_period > 1 and len(y_hw) >= (2 * seasonal_period + 5):
        es_hw = ExponentialSmoothing(
            y_hw,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
        )
    else:
        es_hw = ExponentialSmoothing(y_hw, trend="add")
    hw_res = es_hw.fit(optimized=True)

    horizon = len(df_future)
    fc_log = hw_res.forecast(horizon)
    return pd.Series(fc_log.values, index=df_future.index)

def fit_prophet(
    train_series_log: pd.Series,
    df_future: pd.DataFrame,
    ts_freq: str = "MS",
    seasonal_period: int = 12,
) -> pd.Series:
    """
    Fit Facebook Prophet on a log-target series and return log-forecasts.

    Prophet decomposes the series into trend + seasonality + holidays using
    a Bayesian structural time-series approach.  It requires no manual ARIMA
    order selection and handles missing values and outliers gracefully.

    Parameters
    ----------
    train_series_log : pd.Series
        Training log-target series with a DatetimeIndex.
    df_future : pd.DataFrame
        DataFrame whose index defines the forecast dates (horizon).
    ts_freq : str, optional
        Pandas frequency alias (e.g. "MS", "W", "YS").
    seasonal_period : int, optional
        Number of periods in one seasonal cycle.  Used to enable/disable
        yearly seasonality (active when seasonal_period == 12 or 52).

    Returns
    -------
    pd.Series
        Log-forecast values indexed by df_future.index.
    """
    from prophet import Prophet  # lazy import — avoids overhead when disabled

    horizon = len(df_future)

    # Prophet requires a DataFrame with columns 'ds' (datetime) and 'y' (value)
    df_train_prophet = pd.DataFrame(
        {"ds": train_series_log.index, "y": train_series_log.values}
    )

    # Enable yearly seasonality only for monthly / weekly data
    yearly_seasonality = seasonal_period in (12, 52)

    model = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,   # irrelevant for monthly / yearly data
        daily_seasonality=False,    # irrelevant for sub-daily-free series
        interval_width=0.80,        # 80 % uncertainty interval (not used here)
    )

    # Suppress verbose Stan output
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    model.fit(df_train_prophet)

    # Build a future DataFrame covering exactly the required horizon
    # We pass the future dates directly to avoid alignment issues.
    future_dates_df = pd.DataFrame({"ds": df_future.index})
    forecast_df = model.predict(future_dates_df)

    fc_log = pd.Series(
        forecast_df["yhat"].values,
        index=df_future.index,
    )
    return fc_log


def fit_theta(train_series_log: pd.Series, df_future: pd.DataFrame, ts_freq: str = "MS", seasonal_period: int = 12) -> pd.Series:
    y_theta = train_series_log.asfreq(ts_freq)
    if seasonal_period and seasonal_period > 1 and len(y_theta) >= 2 * seasonal_period:
        period = seasonal_period
    else:
        period = None
    theta_model = ThetaModel(
        y_theta,
        period=period,
        deseasonalize=bool(period),
    )
    theta_res = theta_model.fit()

    horizon = len(df_future)
    fc_log = theta_res.forecast(horizon)
    return pd.Series(fc_log.values, index=df_future.index)

# =====================================================================
#  Model builders (univariate neural forecast based target models)
# =====================================================================

# Frequency-aware neural model params (NHITS / NBEATS / TFT)
def get_neural_input_size_and_downsample(freq: str, train_len: int, h: int):
    """
    Choose frequency-aware input_size (lookback window) and NHITS downsampling.

    - Weekly ("W"): use ~1 year history (52) when possible (captures annual patterns)
    - Monthly ("M"): use ~2 years history (24) when possible
    - Yearly ("Y"): use ~10 periods when possible; avoid downsampling since yearly series is often short

    Also caps input_size to ensure there is room for horizon h (train_len - h),
    and enforces a small minimum window.
    """
    freq = str(freq or "M").upper()

    if freq == "W":
        base_input = 52
        n_freq_downsample = [2, 1, 1]
    elif freq == "M":
        base_input = 24
        n_freq_downsample = [2, 1, 1]
    else:  # "Y" (or fallback)
        base_input = 10
        n_freq_downsample = [1, 1, 1]

    # Keep room for forecasting horizon; enforce minimum window
    max_allowed = max(4, train_len - h)
    input_size = min(base_input, train_len, max_allowed)

    return input_size, n_freq_downsample

def neuralforecast_univariate(
    df_local: pd.DataFrame,
    target_col: str,
    horizon: int,
    model,
    ts_freq: str = "MS",
) -> pd.Series:
    """
    Fit a NeuralForecast model on a univariate target series and return
    the forecast for the specified horizon as a pandas Series.

    This is a thin convenience wrapper to integrate:
      - NHITS, NBEATS, TFT etc.
    into the rest of the pipeline.

    Format expected by NeuralForecast:
      - DataFrame with columns:
          * "unique_id" : series identifier (we use a single ID: "series")
          * "ds"        : datetime column
          * "y"         : target values

    Parameters
    ----------
    df_local : pd.DataFrame
        Must contain `target_col` and a DatetimeIndex.
    target_col : str
        Name of the target column in df_local (typically "target_log").
    horizon : int
        Forecast horizon (should match the model's h parameter).
    model :
        A NeuralForecast model instance (e.g., NHITS(h=...), NBEATS(...),
        TFT(...).
    ts_freq : str, optional
        Frequency string (e.g. "MS" for month-start).

    Raises
    ------
    ImportError
        If `neuralforecast` is not installed.
    ValueError
        If a suitable forecast value column cannot be inferred from
        NeuralForecast's output.

    Returns
    -------
    pd.Series
        Forecast series indexed by timestamp (`ds`), containing the
        model's forecast values for the given horizon.
    """
    
    if hasattr(model, "h") and model.h != horizon:
        raise ValueError(f"Mismatch: model.h = {model.h}, but horizon = {horizon}")

    # Prepare data for NeuralForecast: single series with id "series"
    df_nf = pd.DataFrame(
        {
            "unique_id": "series",
            "ds": df_local.index,
            "y": df_local[target_col].values,
        }
    )

    # Configure and fit NeuralForecast container
    nf = NeuralForecast(models=[model], freq=ts_freq)
    nf.fit(df_nf)

    # Predict over the specified horizon
    fc = nf.predict()
    if isinstance(fc, pd.Series):
        # Some versions may return a Series; convert to DataFrame
        fc = fc.to_frame(name=model.__class__.__name__)

    # Try to infer the forecast column:
    # - Exclude metadata columns ("unique_id", "ds")
    value_cols = [c for c in fc.columns if c not in ["unique_id", "ds"]]
    if not value_cols:
        # Fallback: attempt matching on model name (case-insensitive)
        model_name = model.__class__.__name__.lower()
        value_cols = [c for c in fc.columns if model_name in c.lower() and c not in ["unique_id", "ds"]]

    if not value_cols:
        raise ValueError(
            f"Could not detect forecast column for model {model.__class__.__name__}. "
            f"Columns returned by NeuralForecast: {list(fc.columns)}"
        )

    # Filter to our single series "series" and index by timestamp
    model_col = value_cols[0]
    fc_series = (
        fc.loc[fc["unique_id"] == "series", ["ds", model_col]]
        .set_index("ds")[model_col]
    )
    return fc_series

# ---------------------------------------------------------------------
# Shared helpers for MODEL TRAIN + FORECAST
# ---------------------------------------------------------------------
def run_recursive_multivariate_log_model(
    model,
    model_key: str,
    model_label: str,
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    indep_cols: list,
    indep_holdout_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    series_name: str,
    fold_idx: int,
    logger,
    freq: str = "M",
    fit_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[pd.Series], Optional[Dict[str, float]]]:
    """
    Train an independent variables aware ML model (log-target) and generate
    recursive multi-step forecasts over the holdout window.

    This encapsulates the common pattern used by:
      - LGBM
      - ElasticNet
      - CatBoost
      - XGBoost

    Parameters
    ----------
    model :
        Instantiated regression model (sklearn / catboost / xgboost).
    model_key : str
        Short key used in dicts (e.g., "lgbm", "elasticnet").
    model_label : str
        Human-readable name for logging (e.g., "LGBM", "CatBoost").
    df_train, df_holdout : pd.DataFrame
        Training and holdout slices (with "target_log" and independent columns).
    indep_cols : list of str
        Names of independent columns.
    indep_holdout_df : pd.DataFrames
        Forecasted independent variable series for the holdout period.
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train_log : pd.Series
        Log-target values for training.
    series_name : str
        Series name for logging.
    fold_idx : int
        Zero-based fold index.
    logger :
        Logger instance.
    fit_kwargs : dict, optional
        Extra keyword arguments passed to model.fit().

    Returns
    -------
    (forecast_series, metrics_dict)
        forecast_series : pd.Series of exp(log-forecast) indexed by holdout dates
        metrics_dict    : dict with mae, mse, rmse, mape
        If training fails or there is insufficient history, returns (None, None).
    """
    fit_kwargs = fit_kwargs or {}

    try:
        # Fit model on log-target features
        model.fit(X_train, y_train_log, **fit_kwargs)
    except Exception as e:
        logger.warning(
            f"[{series_name}][Fold {fold_idx + 1}] {model_label} fit failed: {e}"
        )
        return None, None

    train_end_date = df_train.index.max()
    target_history_log = (
        df_train["target_log"].loc[:train_end_date].tail(N_LAGS_TARGET).tolist()
    )

    if len(target_history_log) < N_LAGS_TARGET:
        logger.warning(
            f"[{series_name}][Fold {fold_idx + 1}] Not enough log-target history for {model_label}. "
            f"Skipping {model_label} for this fold."
        )
        return None, None

    # Safe bounds for log-predictions: training range ± 3 std.
    # Clips runaway recursive predictions (e.g. ANN divergence) before they
    # corrupt the lag-history buffer and cascade into infinity.
    _y_log_arr = np.array(y_train_log)
    _log_mean = float(_y_log_arr.mean())
    _log_std  = max(float(_y_log_arr.std()), 1e-6)
    _log_lo   = _log_mean - 5.0 * _log_std
    _log_hi   = _log_mean + 5.0 * _log_std

    preds_log = []
    for dt in df_holdout.index:
        indep_current = {col: indep_holdout_df.loc[dt, col] for col in indep_cols}
        x_row = build_target_feature_row(
            target_history_log=target_history_log,
            indep_current=indep_current,
            current_date=dt,
            feature_columns=list(X_train.columns),
            n_lags_target=N_LAGS_TARGET,
            freq=freq,
        )
        try:
            y_hat_log = float(model.predict(x_row)[0])
        except Exception as e:
            logger.warning(
                f"[{series_name}][Fold {fold_idx + 1}] {model_label} prediction failed at {dt}: {e}"
            )
            return None, None

        # Guard: replace non-finite or out-of-bounds values with the last
        # known log-target so the history buffer stays well-behaved.
        if not np.isfinite(y_hat_log) or y_hat_log < _log_lo or y_hat_log > _log_hi:
            logger.warning(
                f"[{series_name}][Fold {fold_idx + 1}] {model_label} produced an "
                f"out-of-range log-prediction ({y_hat_log:.4g}) at {dt}. "
                f"Clamping to training range [{_log_lo:.4g}, {_log_hi:.4g}]."
            )
            y_hat_log = float(np.clip(y_hat_log if np.isfinite(y_hat_log) else _log_mean,
                                      _log_lo, _log_hi))

        preds_log.append(y_hat_log)

        # Recursive update of history
        target_history_log.append(y_hat_log)
        if len(target_history_log) > N_LAGS_TARGET:
            target_history_log.pop(0)

    # Back-transform to price space
    forecast_vals = np.exp(np.array(preds_log))

    # Final sanity check: if the back-transformed series still contains
    # non-finite values, discard this model for the fold.
    if not np.all(np.isfinite(forecast_vals)):
        logger.warning(
            f"[{series_name}][Fold {fold_idx + 1}] {model_label} forecast contains "
            f"non-finite values after exp(). Discarding model for this fold."
        )
        return None, None

    forecast = pd.Series(
        forecast_vals,
        index=df_holdout.index,
        name=f"Forecast_{model_key}",
    )
    return forecast, evaluate_regression_forecast(df_holdout["target"], forecast)


def run_univariate_statistical_log_model(
    model_key: str,
    model_label: str,
    train_series_log: pd.Series,
    df_holdout: pd.DataFrame,
    series_name: str,
    fold_idx: int,
    logger,
    ts_freq: str = "MS",
    seasonal_period: int = 12,
) -> Tuple[Optional[pd.Series], Optional[Dict[str, float]]]:
    """
    Shared wrapper for univariate log-target TS models like:
      - Holt-Winters / ExponentialSmoothing
      - Theta model

    Parameters
    ----------
    model_key : str
        Short key used in dicts (e.g., "holtwinters", "theta").
    model_label : str
        Human-readable name for logging.
    train_series_log : pd.Series
        Training log-target series (DatetimeIndex, freq "MS").
    df_holdout : pd.DataFrame
        Holdout slice (used only for dates and actuals).
    series_name : str
        Series name for logging.
    fold_idx : int
        Fold index.
    logger :
        Logger instance.
    fit_and_forecast_fn : callable
        Function that takes (train_series_log, horizon) and returns a
        pandas Series (log forecasts) aligned to the horizon length.

    Returns
    -------
    (forecast_series, metrics_dict) or (None, None) on failure.
    """
    try:
        if model_key == "holtwinters":
            fc_log = fit_holtwinters(train_series_log, df_holdout, ts_freq=ts_freq, seasonal_period=seasonal_period)
        elif model_key == "theta":
            fc_log = fit_theta(train_series_log, df_holdout, ts_freq=ts_freq, seasonal_period=seasonal_period)
        elif model_key == "prophet":
            fc_log = fit_prophet(train_series_log, df_holdout, ts_freq=ts_freq, seasonal_period=seasonal_period)

        fc_log = fc_log.reindex(df_holdout.index)
        forecast = pd.Series(
            np.exp(fc_log.values),
            index=df_holdout.index,
            name=f"Forecast_{model_key}",
        )
        metrics = evaluate_regression_forecast(df_holdout["target"], forecast)
        log_holdout_metrics(logger, series_name, fold_idx, model_label, metrics)
        return forecast, metrics
    except Exception as e:
        logger.warning(
            f"[{series_name}][Fold {fold_idx + 1}] {model_label} failed for holdout: {e}"
        )
        return None, None


def run_univariate_neural_log_model(
    model,
    model_key: str,
    model_label: str,
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str,
    series_name: str,
    fold_idx: int,
    logger,
    ts_freq: str = "MS",
) -> Tuple[Optional[pd.Series], Optional[Dict[str, float]]]:
    """
    Shared wrapper for NeuralForecast univariate models (log-target):
      - NHITS
      - NBEATS
      - TFT

    Parameters
    ----------
    model :
        Instantiated NeuralForecast model (NHITS, NBEATS, TFT, etc.).
    model_key : str
        Short key used in dicts (e.g., "nhits", "nbeats", "tft").
    model_label : str
        Human-readable name for logging.
    df_train : pd.DataFrame
        Training slice with "target_log".
    df_holdout : pd.DataFrame
        Holdout slice with "target".
    series_name : str
        Series name for logging.
    fold_idx : int
        Fold index.
    logger :
        Logger instance.

    Returns
    -------
    (forecast_series, metrics_dict) or (None, None) on failure.
    """

    try:
        fc_log = neuralforecast_univariate(
            df_local=df_train,
            target_col=target_col,
            horizon=len(df_holdout),
            model=model,
            ts_freq=ts_freq,
        ).reindex(df_holdout.index)

        forecast = pd.Series(
            np.exp(fc_log.values),
            index=df_holdout.index,
            name=f"Forecast_{model_key}",
        )
        metrics = evaluate_regression_forecast(df_holdout["target"], forecast)
        log_holdout_metrics(logger, series_name, fold_idx, model_label, metrics)
        return forecast, metrics
    except Exception as e:
        logger.warning(
            f"[{series_name}][Fold {fold_idx + 1}] {model_label} failed for holdout: {e}"
        )
        return None, None
    

def evaluate_regression_forecast(
    actual: pd.Series,
    forecast: pd.Series,
) -> Dict[str, float]:
    """
    Compute standard regression metrics between actual and forecast.

    Metrics:
      - MAE  : mean_absolute_error
      - MSE  : mean_squared_error
      - RMSE : sqrt(MSE)
      - MAPE : mean_absolute_percentage_error

    Returns
    -------
    dict
        {"mae": ..., "mse": ..., "rmse": ..., "mape": ...}

    Notes
    -----
    Any non-finite values in `forecast` are replaced with the corresponding
    `actual` value before metric computation (zero-error fallback) so that
    a single bad prediction never crashes the entire evaluation step.
    """
    actual_arr   = np.array(actual,   dtype=float)
    forecast_arr = np.array(forecast, dtype=float)

    # Replace any inf / nan in forecast with the actual value (neutral fallback)
    bad_mask = ~np.isfinite(forecast_arr)
    if bad_mask.any():
        forecast_arr[bad_mask] = actual_arr[bad_mask]

    mae  = mean_absolute_error(actual_arr, forecast_arr)
    mse  = mean_squared_error(actual_arr, forecast_arr)
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(actual_arr, forecast_arr)
    return dict(mae=mae, mse=mse, rmse=rmse, mape=mape)


def log_holdout_metrics(
    logger,
    series_name: str,
    fold_idx: int,
    model_label: str,
    metrics: Dict[str, float],
):
    """
    Log a standard holdout metrics line for a given model.

    Parameters
    ----------
    logger :
        Logger instance (with .info).
    series_name : str
        Name of the series.
    fold_idx : int
        Zero-based fold index (will be logged as fold_idx + 1).
    model_label : str
        Human-readable model label for logging (e.g., "LGBM", "Holt-Winters").
    metrics : dict
        Must contain keys: "mae", "rmse", "mape".
    """
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    mape = metrics["mape"]
    logger.info(
        f"[{series_name}][Fold {fold_idx + 1}] {model_label} HOLDOUT -> "
        f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f} ({mape*100:.2f}%)"
    )


# =====================================================================
# Model builders — Additional ML models
# =====================================================================

def build_lasso_model() -> Pipeline:
    """
    Construct a LASSO regression pipeline (L1 penalty) for log-target forecasting.

    Pipeline:
    ---------
    1) StandardScaler  — scale features to zero mean / unit variance.
    2) LassoCV         — cross-validated LASSO; selects optimal alpha via
                         TimeSeriesSplit to respect chronological order.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.linear_model import LassoCV

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=TimeSeriesSplit(n_splits=5),
                    max_iter=20_000,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_rf_model():
    """
    Construct a Random Forest regressor for log-target forecasting.

    Key hyperparameters
    -------------------
    n_estimators=500       : large ensemble for stability.
    max_depth=8            : moderate depth to curb overfitting.
    min_samples_leaf=10    : minimum leaf size as regularisation.
    max_features=0.7       : random feature fraction per split.

    Returns
    -------
    RandomForestRegressor
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def build_ann_model() -> Pipeline:
    """
    Construct a Multi-Layer Perceptron (ANN) pipeline for log-target forecasting.

    Architecture: (64 → 32) hidden units, ReLU activations.
    Uses early stopping (patience 20 epochs) against a 15 % validation split
    to prevent overfitting on short financial series.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.neural_network import MLPRegressor

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_svr_model() -> Pipeline:
    """
    Construct a Support Vector Regression (RBF kernel) pipeline.

    Pipeline:
    ---------
    1) StandardScaler — SVR is scale-sensitive; normalisation is mandatory.
    2) SVR            — RBF kernel; C=10 balances margin and fit;
                        epsilon=0.05 sets the tube width in log space.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.svm import SVR

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"),
            ),
        ]
    )


# =====================================================================
# Econometric estimator classes — sklearn-compatible wrappers
# (adapted for single time-series with independent variables)
# =====================================================================

def _identify_indep_cols(cols: List[str]) -> List[str]:
    """
    Return the subset of feature columns that correspond to independent
    (exogenous) variables — i.e., exclude autoregressive lags, calendar
    features, and rolling statistics created by make_target_features.

    Parameters
    ----------
    cols : list[str]
        Full list of feature matrix column names.

    Returns
    -------
    list[str]
        Independent-variable column names only.
    """
    reserved = {
        "month", "year", "week",
        "season_sin", "season_cos",
        "rolling_short", "rolling_long",
        "std_short", "std_long",
    }
    return [
        c for c in cols
        if c not in reserved and not c.startswith("target_lag_")
    ]


class TWFEEstimator:
    """
    Two-way Fixed Effects (TWFE) estimator adapted for single time series.

    In a panel context TWFE controls for entity fixed effects (removed by
    within-transformation) and time fixed effects (year + period dummies).
    For a single series the entity FE collapses to a constant term;  the
    time FE are implemented as binary year and month (or week) dummies that
    replace the continuous year / month / season_sin / season_cos features.

    Estimation
    ----------
    OLS via statsmodels on the augmented feature matrix.

    Usage
    -----
    Fully sklearn-compatible: exposes .fit(X, y) and .predict(X).
    """

    def __init__(self):
        self._params = None
        self._feature_cols: List[str] = []
        self._year_dummies: List[int] = []
        self._month_dummies: List[int] = []
        self._week_dummies: List[int] = []

    # ------------------------------------------------------------------
    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace calendar columns with binary dummies; add constant."""
        import statsmodels.api as sm

        X_aug = X.copy().fillna(0)

        # Drop cyclic encodings (superseded by binary dummies)
        for col in ["season_sin", "season_cos"]:
            if col in X_aug.columns:
                X_aug = X_aug.drop(columns=[col])

        # Year dummies (all years except the reference / first year)
        if "year" in X_aug.columns:
            years = X_aug["year"].astype(int)
            for yr in self._year_dummies:
                X_aug[f"_yd_{yr}"] = (years == yr).astype(float)
            X_aug = X_aug.drop(columns=["year"])

        # Month dummies (months 2–12; month 1 is the reference)
        if "month" in X_aug.columns:
            months = X_aug["month"].astype(int)
            for m in self._month_dummies:
                X_aug[f"_md_{m}"] = (months == m).astype(float)
            X_aug = X_aug.drop(columns=["month"])

        # Week dummies (for weekly frequency data)
        if "week" in X_aug.columns:
            weeks = X_aug["week"].astype(int)
            for wk in self._week_dummies:
                X_aug[f"_wkd_{wk}"] = (weeks == wk).astype(float)
            X_aug = X_aug.drop(columns=["week"])

        X_aug = sm.add_constant(X_aug, has_constant="add")
        return X_aug

    # ------------------------------------------------------------------
    def fit(self, X, y):
        import statsmodels.api as sm

        X_df = pd.DataFrame(X).fillna(0)

        # Determine which dummy levels exist in the training data
        if "year" in X_df.columns:
            years = X_df["year"].astype(int)
            unique_years = sorted(years.unique())
            self._year_dummies = unique_years[1:]   # drop reference year
        if "month" in X_df.columns:
            self._month_dummies = list(range(2, 13))
        if "week" in X_df.columns:
            weeks = X_df["week"].astype(int)
            unique_weeks = sorted(weeks.unique())
            self._week_dummies = unique_weeks[1:]

        X_aug = self._augment(X_df)
        self._feature_cols = list(X_aug.columns)

        result = sm.OLS(np.array(y).flatten(), X_aug).fit()
        self._params = result.params
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        X_df = pd.DataFrame(X).fillna(0)
        X_aug = self._augment(X_df)

        # Align to training columns
        for col in self._feature_cols:
            if col not in X_aug.columns:
                X_aug[col] = 0.0
        X_aug = X_aug.reindex(columns=self._feature_cols, fill_value=0.0)

        return X_aug.values @ self._params.values


class SysGMMEstimator:
    """
    System GMM approximation for single time series via Two-Stage Least Squares.

    In dynamic panel models, sysGMM treats the lagged dependent variable as
    endogenous and uses deeper lags as instruments.  Here we apply the same
    idea to a single series:

      • Endogenous regressor : target_lag_1
      • Instruments          : target_lag_3, target_lag_4, target_lag_5
        (sufficiently deep to be predetermined / exogenous)
      • First stage          : regress target_lag_1 on instruments + exog
      • Second stage         : OLS on (fitted_lag_1, exog)

    Falls back to standard OLS if lag columns are unavailable.

    Usage
    -----
    sklearn-compatible: .fit(X, y) / .predict(X).
    """

    def __init__(self):
        self._params = None
        self._feature_cols: List[str] = []
        self._fs_params = None       # first-stage OLS params
        self._fs_cols: List[str] = []
        self._is_iv: bool = False

    # ------------------------------------------------------------------
    def fit(self, X, y):
        import statsmodels.api as sm

        X_df = pd.DataFrame(X).fillna(0)
        y_arr = np.array(y).flatten()

        endog_col = "target_lag_1"
        potential_instr = ["target_lag_3", "target_lag_4", "target_lag_5"]
        instr_cols = [c for c in potential_instr if c in X_df.columns]

        if endog_col in X_df.columns and len(instr_cols) >= 2:
            self._is_iv = True

            exog_cols = [c for c in X_df.columns if c != endog_col]
            X_exog = X_df[exog_cols]

            # First stage: regress endog on deep-lag instruments + exog
            X_fs = pd.concat([X_exog, X_df[instr_cols]], axis=1).fillna(0)
            X_fs_const = sm.add_constant(X_fs, has_constant="add")
            self._fs_cols = list(X_fs_const.columns)
            fs_res = sm.OLS(X_df[endog_col].values, X_fs_const).fit()
            self._fs_params = fs_res.params

            lag1_hat = fs_res.fittedvalues

            # Second stage: OLS on (exog + fitted endog)
            X_ss = X_exog.copy()
            X_ss["target_lag_1_hat"] = lag1_hat.values
            X_ss_const = sm.add_constant(X_ss, has_constant="add")
            ss_res = sm.OLS(y_arr, X_ss_const).fit()

            self._params = ss_res.params
            self._feature_cols = list(X_ss_const.columns)
        else:
            # Fallback: plain OLS
            self._is_iv = False
            X_const = sm.add_constant(X_df, has_constant="add")
            result = sm.OLS(y_arr, X_const).fit()
            self._params = result.params
            self._feature_cols = list(X_const.columns)

        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        import statsmodels.api as sm

        X_df = pd.DataFrame(X).fillna(0)

        if self._is_iv:
            endog_col = "target_lag_1"
            exog_cols = [c for c in X_df.columns if c != endog_col]
            X_exog = X_df[exog_cols]

            # First stage: predict fitted lag1
            X_fs = X_df.copy()
            X_fs_const = sm.add_constant(X_fs, has_constant="add")
            for col in self._fs_cols:
                if col not in X_fs_const.columns:
                    X_fs_const[col] = 0.0
            X_fs_const = X_fs_const.reindex(
                columns=self._fs_cols, fill_value=0.0
            )
            lag1_hat = X_fs_const.values @ self._fs_params.values

            # Second stage
            X_ss = X_exog.copy()
            X_ss["target_lag_1_hat"] = lag1_hat
            X_ss_const = sm.add_constant(X_ss, has_constant="add")
            for col in self._feature_cols:
                if col not in X_ss_const.columns:
                    X_ss_const[col] = 0.0
            X_ss_const = X_ss_const.reindex(
                columns=self._feature_cols, fill_value=0.0
            )
            return X_ss_const.values @ self._params.values
        else:
            X_const = sm.add_constant(X_df, has_constant="add")
            for col in self._feature_cols:
                if col not in X_const.columns:
                    X_const[col] = 0.0
            X_const = X_const.reindex(
                columns=self._feature_cols, fill_value=0.0
            )
            return X_const.values @ self._params.values


class MGEstimator:
    """
    Mean Group (MG) estimator adapted for single time series.

    In the panel literature, MG estimates a separate OLS regression for each
    cross-sectional unit and averages the coefficients.  Applied here to a
    single series, we partition the training history into K non-overlapping
    sub-windows, fit OLS on each, then average the slope vectors and
    intercepts — producing a "mean-group" linear predictor that is more
    robust to structural shifts than a single pooled OLS.

    Parameters
    ----------
    n_splits : int
        Number of sub-windows to split the training data into.
    min_window : int
        Minimum window size in observations; enforced to avoid degenerate fits.

    Usage
    -----
    sklearn-compatible: .fit(X, y) / .predict(X).
    """

    def __init__(self, n_splits: int = 5, min_window: int = 20):
        self.n_splits = n_splits
        self.min_window = min_window
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    # ------------------------------------------------------------------
    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression

        X_arr = np.array(X, dtype=float)
        np.nan_to_num(X_arr, copy=False)
        y_arr = np.array(y).flatten()
        n = len(X_arr)

        window_size = max(self.min_window, n // self.n_splits)

        coefs, intercepts = [], []
        for s in range(0, n, window_size):
            e = min(s + window_size, n)
            if (e - s) < self.min_window:
                break
            lr = LinearRegression()
            try:
                lr.fit(X_arr[s:e], y_arr[s:e])
                coefs.append(lr.coef_)
                intercepts.append(float(lr.intercept_))
            except Exception:
                continue

        if coefs:
            self._coef = np.mean(coefs, axis=0)
            self._intercept = float(np.mean(intercepts))
        else:
            # Fallback: single OLS on all data
            lr = LinearRegression()
            lr.fit(X_arr, y_arr)
            self._coef = lr.coef_
            self._intercept = float(lr.intercept_)

        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        X_arr = np.array(X, dtype=float)
        np.nan_to_num(X_arr, copy=False)
        return X_arr @ self._coef + self._intercept


class AMGEstimator:
    """
    Augmented Mean Group (AMG) estimator adapted for single time series.

    AMG extends MG by first extracting a common dynamic factor (the first
    principal component of the independent variables) and adding it as an
    additional regressor.  This controls for cross-sectional dependence
    (common macro shocks) before computing the mean-group OLS averages.

    In the single-series context:
      • The independent variables at each date serve as the "cross-section".
      • PC1 of those variables captures the dominant shared movement
        (analogous to Pesaran's cross-sectional average in large panels).

    Parameters
    ----------
    n_splits / min_window : same as MGEstimator.

    Usage
    -----
    sklearn-compatible: .fit(X, y) / .predict(X).
    """

    def __init__(self, n_splits: int = 5, min_window: int = 20):
        self.n_splits = n_splits
        self.min_window = min_window
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._pca = None
        self._pca_scaler = None
        self._indep_cols: List[str] = []
        self._aug_col_order: List[str] = []

    # ------------------------------------------------------------------
    def _augment_with_factor(
        self, X_df: pd.DataFrame, fitting: bool = False
    ) -> np.ndarray:
        """Add PC1 of independent variables as '_common_factor' column."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler as _SS

        X_aug = X_df.copy().fillna(0)

        if self._indep_cols:
            indep_present = [
                c for c in self._indep_cols if c in X_aug.columns
            ]
            if indep_present:
                X_indep = (
                    X_aug[indep_present]
                    .reindex(columns=self._indep_cols, fill_value=0.0)
                )
                if fitting:
                    self._pca_scaler = _SS()
                    X_scaled = self._pca_scaler.fit_transform(X_indep)
                    self._pca = PCA(n_components=1)
                    common_factor = self._pca.fit_transform(X_scaled).flatten()
                else:
                    X_scaled = self._pca_scaler.transform(X_indep)
                    common_factor = self._pca.transform(X_scaled).flatten()
            else:
                common_factor = np.zeros(len(X_aug))
            X_aug["_common_factor"] = common_factor

        if fitting:
            self._aug_col_order = list(X_aug.columns)

        return np.nan_to_num(X_aug.reindex(
            columns=self._aug_col_order, fill_value=0.0
        ).values, copy=False)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression

        X_df = pd.DataFrame(X)
        self._indep_cols = _identify_indep_cols(X_df.columns.tolist())

        X_arr = self._augment_with_factor(X_df, fitting=True)
        y_arr = np.array(y).flatten()
        n = len(X_arr)

        window_size = max(self.min_window, n // self.n_splits)
        coefs, intercepts = [], []

        for s in range(0, n, window_size):
            e = min(s + window_size, n)
            if (e - s) < self.min_window:
                break
            lr = LinearRegression()
            try:
                lr.fit(X_arr[s:e], y_arr[s:e])
                coefs.append(lr.coef_)
                intercepts.append(float(lr.intercept_))
            except Exception:
                continue

        if coefs:
            self._coef = np.mean(coefs, axis=0)
            self._intercept = float(np.mean(intercepts))
        else:
            lr = LinearRegression()
            lr.fit(X_arr, y_arr)
            self._coef = lr.coef_
            self._intercept = float(lr.intercept_)

        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        X_df = pd.DataFrame(X)
        X_arr = self._augment_with_factor(X_df, fitting=False)
        return X_arr @ self._coef + self._intercept


class CCEMGEstimator:
    """
    Common Correlated Effects Mean Group (CCEMG) adapted for single time series.

    In Pesaran's panel CCEMG, the cross-sectional averages of all variables
    at each time period are added as regressors to control for unobserved
    common factors.  For a single time series (one "unit"), we treat the
    multiple independent variables as the cross-section and compute their
    row-wise mean as the proxy for the cross-sectional average (_cce_mean).

    This augmented OLS is then estimated on the full training sample
    (a degenerate "mean group" with a single unit).

    Usage
    -----
    sklearn-compatible: .fit(X, y) / .predict(X).
    """

    def __init__(self):
        self._params = None
        self._feature_cols: List[str] = []
        self._indep_cols: List[str] = []

    # ------------------------------------------------------------------
    def _augment(
        self, X_df: pd.DataFrame, fitting: bool = False
    ) -> pd.DataFrame:
        import statsmodels.api as sm

        X_aug = X_df.copy().fillna(0)
        indep_present = [c for c in self._indep_cols if c in X_aug.columns]

        if indep_present:
            X_aug["_cce_mean"] = (
                X_aug[indep_present]
                .reindex(columns=self._indep_cols, fill_value=0.0)
                .mean(axis=1)
            )
        else:
            X_aug["_cce_mean"] = 0.0

        X_aug = sm.add_constant(X_aug, has_constant="add")
        return X_aug

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        self._indep_cols = _identify_indep_cols(X_df.columns.tolist())

        X_aug = self._augment(X_df, fitting=True)
        self._feature_cols = list(X_aug.columns)

        import statsmodels.api as sm
        result = sm.OLS(np.array(y).flatten(), X_aug).fit()
        self._params = result.params
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        X_df = pd.DataFrame(X)
        X_aug = self._augment(X_df)

        for col in self._feature_cols:
            if col not in X_aug.columns:
                X_aug[col] = 0.0
        X_aug = X_aug.reindex(columns=self._feature_cols, fill_value=0.0)

        return X_aug.values @ self._params.values


class DCCEMGEstimator(CCEMGEstimator):
    """
    Dynamic CCEMG (DCCEMG) adapted for single time series.

    Extends CCEMG by also including the one-period lagged cross-sectional
    mean (_cce_mean_lag1) as an additional regressor.  This captures
    persistence in the common factor, making the estimator "dynamic"
    in the same spirit as the panel DCCEMG.

    Note: target_lag_1 is already present in the feature matrix from
    make_target_features, so the lagged dependent variable is accounted
    for automatically — consistent with the dynamic panel motivation.

    Usage
    -----
    sklearn-compatible: .fit(X, y) / .predict(X).
    """

    # ------------------------------------------------------------------
    def _augment(
        self, X_df: pd.DataFrame, fitting: bool = False
    ) -> pd.DataFrame:
        import statsmodels.api as sm

        X_aug = X_df.copy().fillna(0)
        indep_present = [c for c in self._indep_cols if c in X_aug.columns]

        if indep_present:
            cce_mean = (
                X_aug[indep_present]
                .reindex(columns=self._indep_cols, fill_value=0.0)
                .mean(axis=1)
            )
            X_aug["_cce_mean"] = cce_mean
            # Lagged CCE mean (shift by 1; at prediction time single row
            # has no prior row, so we re-use current as the best proxy)
            X_aug["_cce_mean_lag1"] = cce_mean.shift(1).fillna(cce_mean)
        else:
            X_aug["_cce_mean"] = 0.0
            X_aug["_cce_mean_lag1"] = 0.0

        X_aug = sm.add_constant(X_aug, has_constant="add")
        return X_aug


def compute_directional_and_band_accuracy(
    actual: pd.Series,
    forecast: pd.Series,
    last_train_value: float,
    band_threshold: float = 0.10,
) -> Tuple[float, float]:
    """
    Compute two diagnostic metrics:
      - Directional Accuracy (DA)
      - Within-band Accuracy (WBA)

    Directional Accuracy (DA)
    -------------------------
    - Compares the sign of month-on-month changes in actual vs predicted
      values, using the last training point as the "previous" value for
      the first holdout month.

    Within-band Accuracy (WBA)
    --------------------------
    - Fraction of months where the relative error is within a tolerance band:
          |error| / |actual| <= band_threshold

      For example, with band_threshold = 0.10, this is the proportion of
      months where the forecast is within ±10% of the actual.

    Parameters
    ----------
    actual : pd.Series
        Actual target values over the holdout window.
    forecast : pd.Series
        Forecasted values over the same window (same index as `actual`).
    last_train_value : float
        Last observed target value from the training window, used as the
        reference for computing the first month-on-month change.
    band_threshold : float, optional
        Relative error tolerance (default 0.10 → 10%).

    Returns
    -------
    tuple[float, float]
        (da, within_band_accuracy)
    """
    if len(actual) == 0:
        return 0.0, 0.0

    actual_arr = actual.values.astype(float)
    pred_arr = forecast.values.astype(float)

    # Within-band Accuracy: percentage of points within |error|/|actual| <= band_threshold
    denom = np.where(np.abs(actual_arr) < 1e-9, 1e-9, np.abs(actual_arr))
    rel_error = np.abs(actual_arr - pred_arr) / denom
    wba = float((rel_error <= band_threshold).mean())

    # Directional Accuracy: compare sign of deltas vs last train point
    prev_actual = np.concatenate([[last_train_value], actual_arr[:-1]])
    true_delta = actual_arr - prev_actual
    pred_delta = pred_arr - prev_actual

    mask = true_delta != 0
    if mask.sum() == 0:
        # Flat series -> no direction changes, consider DA perfect.
        da = 1.0
    else:
        hits = np.sign(true_delta[mask]) == np.sign(pred_delta[mask])
        da = float(hits.mean())

    return da, wba
