# utils_indep_model_helpers.py
# =====================================================================
# Helpers for independent variable forecasting:
#
# This module focuses on models that predict independent variable
# time series independently of the main target. These forward paths
# for independent variables are later used as inputs to the target
# variable forecasting pipeline.
#
# Included:
#   - Classical time-series models (per independent variable series):
#       * ETS (Holt-Winters Exponential Smoothing)
#       * Auto-ARIMA
#       * Theta model
#   - A simple ensemble of ETS + ARIMA + Theta for independent variable 
#     forecasting
#   - Anomaly / risk scoring for future independent paths vs historical
#     distributions
#   - Future-path generation for independent variables (Step-3)
#     that is:
#       * series-agnostic
#       * frequency-aware (M/W/Y)
# =====================================================================

from typing import Dict, List, Tuple

import os
import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from pmdarima import auto_arima

# Custom utilities
from utils.utils_common import (
    load_excel_file,
    sanitize_column_names,
    sanitize_column_values
)

from utils.utils_feature_selection import time_interpolate

from settings import (
    RANDOM_STATE,   # used to make Auto-ARIMA search reproducible
    N_JOBS,         # used to parallelize Auto-ARIMA search where possible,
    INDEP_ENSEMBLE_WEIGHTS_SUFFIX,
    FUTURE_FORECAST_SUFFIX,
    HIST_INDEP_NUM_MONTHS,
    CSV_FLOAT_FMT
)


# =====================================================================
# Frequency helpers
# =====================================================================

def ts_freq_from_code(freq: str) -> str:
    """
    Map config frequency code to a pandas frequency string and
    a default seasonal period.

        "M" -> ("MS", 12)
        "W" -> ("W",  52)
        "Y" -> ("YS", 1)

    Returns
    -------
    (ts_freq, seasonal_period)
    """

    if freq == "M":
        ts_freq = "MS"
        seasonal_period = 12
    elif freq == "W":
        ts_freq = "W-MON"
        seasonal_period = 52
    else:  # "Y"
        ts_freq = "YS"
        seasonal_period = 1

    return ts_freq, seasonal_period

def standardize_series_to_freq(series: pd.Series, freq: str) -> pd.Series:
    """
    Aggregate an arbitrary datetime-indexed series to one value per
    period at the configured frequency, using the LAST value in each
    period, and return a regular TimestampIndex.

    Example:
        freq = "M" → one value per month (month-start timestamps).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex.")
    
    p = series.index.to_period(freq)
    s = series.groupby(p).last()
    s.index = s.index.to_timestamp()
    return s

def _compute_hist_periods(freq: str) -> int:
    """
    Convert HIST_INDEP_NUM_MONTHS (interpreted as ~2 years of history)
    into an appropriate number of periods for the configured frequency.

    For example, if HIST_INDEP_NUM_MONTHS = 24:
        - Monthly ("M"): 24 periods (~24 months ≈ 2 years)
        - Weekly  ("W"): ~104 periods (52 * 2 weeks ≈ 2 years)
        - Yearly  ("Y"): 2 periods (~2 years)
    """
    years_hist = HIST_INDEP_NUM_MONTHS / 12.0  # e.g., 24 months -> 2 years

    if freq == "M":
        hist_periods = int(round(12 * years_hist))   # ~24
    elif freq == "W":
        hist_periods = int(round(52 * years_hist))   # ~104
    else:  # "Y"
        hist_periods = max(1, int(round(1 * years_hist)))  # ~2

    return max(1, hist_periods)

# =====================================================================
# Classical TS models for independent variables
# =====================================================================

def forecast_holt_winters(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = 12,
    ts_freq: str = "MS",
) -> np.ndarray:
    """
    Holt-Winters / Exponential Smoothing forecast for an independent 
    variable series.

    Logic:
      - Standardize the input series to given frequency (ts_freq).
      - If sufficient data length is available (≥ 2 * seasonal_period + 5),
        fit a full additive trend + additive seasonality model:
            trend="add", seasonal="add", seasonal_periods=seasonal_period
      - Otherwise, fall back to a trend-only model:
            trend="add" (no explicit seasonal component).
    """

    # Ensure a regular monthly-start frequency
    y = pd.Series(train.values, index=train.index).asfreq(ts_freq)

    # Decide whether to include a seasonal component based on data length
    if seasonal_period and seasonal_period >= 2 and len(y) >= (2 * seasonal_period + 5):
        es = ExponentialSmoothing(
            y,
            trend="add", # Allows a linear (additive) trend component.
            seasonal="add", # Uses an additive seasonal pattern when enabled.
            seasonal_periods=seasonal_period, # Typical choice for monthly data is 12 (1-year seasonality).
        )
    else:
        # Not enough data to reliably estimate seasonality → trend-only
        es = ExponentialSmoothing(y, trend="add")

    # Fit model and get predictions
    model = es.fit(optimized=True)
    preds = np.array(model.forecast(horizon))

    return preds


def forecast_auto_arima(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = 12,
    ts_freq: str = "MS",
) -> np.ndarray:
    """
    Auto-ARIMA forecast for an independent variable series.

    Uses pmdarima.auto_arima to automatically select an appropriate
    (p, d, q)(P, D, Q)m model based on information criteria (AIC).
    """

    # Standardize to configured frequency
    y = pd.Series(train.values, index=train.index).asfreq(ts_freq)

    seasonal = bool(seasonal_period and seasonal_period > 1)
    m = seasonal_period if seasonal else 1

    # ✅ Weekly safety: if m is big (52), reduce complexity
    if m >= 52:
        stepwise = True
        n_jobs = 1
        max_p, max_q, max_d = 3, 3, 1
        max_P, max_Q, max_D = 1, 1, 1
    else:
        stepwise = True
        n_jobs = N_JOBS
        max_p, max_q, max_d = 5, 5, 2
        max_P, max_Q, max_D = 2, 2, 1

    model = auto_arima(
        y,
        seasonal=seasonal,              # bool: Whether to fit a seasonal ARIMA (SARIMA) vs non-seasonal ARIMA.
        m=m,                            # int: Seasonal period (e.g., 12 for monthly data with yearly seasonality).
        max_p=max_p,                    # int: Maximum non-seasonal AR order to consider.
        max_q=max_q,                    # int: Maximum non-seasonal MA order to consider.
        max_d=max_d,                    # int: Maximum differencing orders (non-seasonal).
        max_P=max_P,                    # int: Maximum seasonal AR orders
        max_Q=max_Q,                    # int: Maximum seasonal MA orders
        max_D=max_D,                    # int: Maximum differencing orders (seasonal).
        information_criterion="aic",    # ARIMA order selection based on AIC.
        seasonal_test="ocsb",           # Test used to decide whether seasonal differencing is needed.
        with_intercept=True, 
        stepwise=stepwise,              # Use full search (more expensive but more thorough) instead of stepwise heuristic
        suppress_warnings=True,         # Robust to numerical issues / model failures during search.
        error_action="ignore",
        n_jobs=n_jobs,                  # Parallelizes the model search where possible.
        random_state=RANDOM_STATE,      # For reproducibility of search heuristics.
    )

    # Get predictions
    preds = np.array(model.predict(n_periods=horizon))
    return preds


def forecast_theta(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = 12,
    ts_freq: str = "MS",
) -> np.ndarray:
    """
    Theta model forecast for an independent variable series.

    Logic:
      - Standardize the input series to given frequency (ts_freq).
      - If enough data is available, uses a seasonal period; otherwise
        fits a non-seasonal theta model.
    """

    # Standardize to configured frequency
    y = pd.Series(train.values, index=train.index).asfreq(ts_freq)

    # Seasonal period; e.g., 12 for yearly cycle in monthly data.
    # If None, the model is non-seasonal.
    period = (
        seasonal_period
        if seasonal_period
        and seasonal_period >= 2
        and len(y) >= 2 * seasonal_period
        else None
    )

    # Deseasonalize: Bool If True, removes seasonality before fitting.
    tm = ThetaModel(y, period=period, deseasonalize=bool(period))
    model = tm.fit()

    # Get predictions
    preds = np.array(model.forecast(horizon))
    return preds


def forecast_indep_variables(
    series: pd.Series,
    horizon: int,
    freq: str = "M",
) -> Dict[str, np.ndarray]:
    """
    Forecast an independent variable series with three base models:

        - ETS   (Holt-Winters / ExponentialSmoothing)
        - ARIMA (auto_arima)
        - Theta (ThetaModel)

    It returns the raw outputs of each model.
    """
    
    # Drop missing values — models require clean series
    series_clean = series.dropna()
    if len(series_clean) == 0:
        raise ValueError(f"No non-NaN history available for {series.name} to forecast.")

    if not isinstance(series_clean.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex for independent variable forecasting.")

    # Standardize to configured frequency
    ts_freq, seasonal_period  = ts_freq_from_code(freq)
    series_clean = standardize_series_to_freq(series_clean, freq).asfreq(ts_freq)
    
    # Run each base model and get forecast
    fc_ets = forecast_holt_winters(
        series_clean,
        horizon=horizon,
        seasonal_period=seasonal_period,
        ts_freq=ts_freq,
    )

    fc_arima = forecast_auto_arima(
        series_clean,
        horizon=horizon,
        seasonal_period=seasonal_period,
        ts_freq=ts_freq,
    )

    fc_theta = forecast_theta(
        series_clean,
        horizon=horizon,
        seasonal_period=seasonal_period,
        ts_freq=ts_freq,
    )

    return {
        "ets": fc_ets,
        "arima": fc_arima,
        "theta": fc_theta,
    }


def forecast_indep_ensemble(
    series: pd.Series,
    last_date: pd.Timestamp,
    horizon: int,
    weights: Dict[str, float] | None = None,
    freq: str = "M",
) -> pd.Series:
    """
    Forecast an independent variable series using an ensemble of ETS, Auto-ARIMA, and Theta.

    Ensemble logic:
      - First, obtain per-model forecasts via `forecast_indep_variables`.
      - If `weights` is None:
            assign each model equal weight (1/3).
      - If `weights` is provided:
            * allowed keys: "ets", "arima", "theta"
            * unspecified keys default to 0.0
            * if the sum of weights <= 0, fallback to equal weights
            * otherwise, normalize weights so they sum to 1.0
      - The final ensemble forecast is:
            fc_final = w_ets * ets + w_arima * arima + w_theta * theta

    Result is indexed by future dates at the configured frequency.
    """

    # Get independent variable forecast using different model
    forecast_dict = forecast_indep_variables(
        series=series,
        horizon=horizon,
        freq=freq
    )

    # Handle default or custom weights
    if weights is None:
        # Equal weights when no explicit preference is specified
        weights = {"ets": 1.0 / 3.0, "arima": 1.0 / 3.0, "theta": 1.0 / 3.0}
    else:
        # Ensure we have all keys; missing ones default to 0.0
        for k in ["ets", "arima", "theta"]:
            weights.setdefault(k, 0.0)
        total_w = sum(weights.values())
        if total_w <= 0:
            # Degenerate case → fallback to equal weights
            weights = {"ets": 1.0 / 3.0, "arima": 1.0 / 3.0, "theta": 1.0 / 3.0}
        else:
            # Normalize so that weights sum to 1
            weights = {k: v / total_w for k, v in weights.items()}

    # Weighted combination of each model's forecast
    forecast_final = (
        weights["ets"] * forecast_dict["ets"]
        + weights["arima"] * forecast_dict["arima"]
        + weights["theta"] * forecast_dict["theta"]
    )

    # Build future index starting from the period after last_date
    last_p = last_date.to_period(freq)
    future_p = pd.period_range(start=last_p + 1, periods=horizon, freq=freq)
    future_index = future_p.to_timestamp()

    return pd.Series(forecast_final, index=future_index, name=series.name)

# =====================================================================
# Independent variable risk labelling & anomaly summary for future forecasts
# =====================================================================
def indep_risk_label(
    score: float
) -> str:
    """
    Map a numeric risk score (0-100) to a qualitative bucket.

    Buckets:
      - 0  to 20  → "Low"
      - >20 to 40 → "Moderate"
      - >40 to 70 → "High"
      - >70       → "Critical"

    Parameters
    ----------
    score : float
        Risk score (typically on 0–100 scale).

    Returns
    -------
    str
        Label describing the risk severity.
    """
    if score <= 20:
        return "Low"
    elif score <= 40:
        return "Moderate"
    elif score <= 70:
        return "High"
    else:
        return "Critical"


def summarize_indep_future_anomalies(
    df_hist: pd.DataFrame,
    df_future: pd.DataFrame,
    z_threshold: float = 3.0,
    range_margin: float = 0.20,
    flat_std_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    Summarize potential anomalies in future independent variable forecasts by comparing
    them against historical distributions and behavior.

    For each variable (column):
      1) Compute historical stats:
           - hist_mean, hist_std, hist_min, hist_max
      2) Compute future stats:
           - future_mean, future_std, future_min, future_max
      3) Check the relative jump at horizon start:
           - rel_jump_first_future_vs_last_hist
             = (future[0] - last_hist) / max(|last_hist|, 1e-9)
      4) Range-based anomaly:
           - Expand [hist_min, hist_max] by ±range_margin * (hist_max - hist_min)
             and count how many future points fall outside this expanded range.
             If count > 0 → flag_out_of_range = True.
      5) Flatness:
           - Compare future_std vs hist_std.
             * If hist_std > 0:
                 flag_flat = future_std < flat_std_ratio * hist_std
             * If hist_std == 0:
                 flag_flat = (future_std == 0)
      6) Large jump:
           - flag_large_jump = (|rel_jump| > 0.5), i.e., > 50% relative move.

      7) Composite risk score (0–100):
           risk_score =
               40 * flag_out_of_range
             + 40 * flag_large_jump
             + 20 * flag_flat

         This is a heuristic that emphasizes:
           - Large distribution shift (going outside historical range).
           - Big jump at the beginning of the horizon.
           - Unnaturally flat future paths vs historical volatility.

    Parameters
    ----------
    df_hist : pd.DataFrame
        Historical independent variable data (DateIndex or similar), columns = variables.
    df_future : pd.DataFrame
        Future independent variable forecasts; same columns as df_hist.
    z_threshold : float, optional
        Currently unused placeholder; kept for potential future z-score logic.
    range_margin : float, optional
        Margin (as a fraction of historical range) used to expand
        [hist_min, hist_max] when checking for out-of-range future points.
    flat_std_ratio : float, optional
        If future_std < flat_std_ratio * hist_std → treated as "flat".

    Returns
    -------
    pd.DataFrame
        One row per variable with columns:
          - variable
          - hist_mean, hist_std, hist_min, hist_max
          - future_mean, future_std, future_min, future_max
          - rel_jump_first_future_vs_last_hist
          - n_future_points
          - n_future_out_of_hist_range
          - flag_out_of_range
          - flag_large_jump
          - flag_flat
          - risk_score
    """

    rows = []
    for col in df_future.columns:
        hist_series = df_hist[col].dropna()
        fut_series = df_future[col].dropna()

        # Skip variables with no usable history or future values
        if hist_series.empty or fut_series.empty:
            continue

        # Historical distribution
        hist_mean = hist_series.mean()
        hist_std = hist_series.std(ddof=0)
        hist_min = hist_series.min()
        hist_max = hist_series.max()

        # Future distribution
        fut_mean = fut_series.mean()
        fut_std = fut_series.std(ddof=0)
        fut_min = fut_series.min()
        fut_max = fut_series.max()

        # Relative jump between last historical and first future point
        # Large jump detection: > 50% relative move at horizon start
        last_hist = hist_series.iloc[-1]
        first_future = fut_series.iloc[0]
        denom = max(1e-9, abs(last_hist))
        rel_jump = (first_future - last_hist) / denom
        flag_large_jump = abs(rel_jump) > 0.5


        # Range-based anomaly check with margin
        hist_range = hist_max - hist_min
        margin = range_margin * hist_range if hist_range > 0 else 0.0
        lower_bound = hist_min - margin
        upper_bound = hist_max + margin

        out_of_range_mask = (fut_series < lower_bound) | (fut_series > upper_bound)
        n_out_of_range = int(out_of_range_mask.sum())
        flag_out_of_range = n_out_of_range > 0

        # Flatness detection: future volatility much lower than historical
        if hist_std > 0:
            flag_flat = fut_std < flat_std_ratio * hist_std
        else:
            # If historical std is zero, treat future as flat only if it's exactly constant
            flag_flat = fut_std == 0.0

        # Heuristic risk score on 0–100 scale
        risk_score = (
            40.0 * float(flag_out_of_range)
            + 40.0 * float(flag_large_jump)
            + 20.0 * float(flag_flat)
        )

        rows.append(
            {
                "variable": col,
                "hist_mean": hist_mean,
                "hist_std": hist_std,
                "hist_min": hist_min,
                "hist_max": hist_max,
                "future_mean": fut_mean,
                "future_std": fut_std,
                "future_min": fut_min,
                "future_max": fut_max,
                "rel_jump_first_future_vs_last_hist": rel_jump,
                "n_future_points": int(len(fut_series)),
                "n_future_out_of_hist_range": n_out_of_range,
                "flag_out_of_range": flag_out_of_range,
                "flag_large_jump": flag_large_jump,
                "flag_flat": flag_flat,
                "risk_score": risk_score,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# For each independent variable: backtest ETS / ARIMA / Theta and
# derive ensemble weights from inverse-MAPE.
# =====================================================================

def get_indep_ensemble_weights(
    input_folder_name: str,
    series_name: str,
    series_config: dict,
    variables_selected_folder_name: str,
    model_sel_dir: str,
    logger,
):
    """
    Compute ensemble weights for each selected independent variable (driver)
    across three univariate time-series models: ETS, ARIMA, Theta.

    For each independent variable:
      - Clean and interpolate its historical series.
      - Run up to 2 rolling holdout folds with horizon = Model_Selection_Period.
      - Collect absolute percentage errors (APE) per method across folds.
      - Convert APE to MAPE per method and derive inverse-MAPE weights.

    Results are saved per series to INDEP_ENSEMBLE_WEIGHTS_SUFFIX CSV.

    Frequency-awareness:
      - The backtest horizon (Model_Selection_Period) is in *periods* at
        the configured Frequency (M/W/Y).
      - Index is standardized to one observation per period via
        `standardize_series_to_freq`.
    """

    # =====================================================================
    # 1) Load Independent Data
    # =====================================================================
    indep_file = series_config.get("Input_File_Name")
    if not indep_file:
        raise ValueError("Input file name missing for independent variables.")

    indep_path = os.path.join(input_folder_name, indep_file)
    if not os.path.isfile(indep_path):
        raise FileNotFoundError(
            f"Required file '{indep_file}' does not exist in '{input_folder_name}'."
        )

    # Load all independent variables from the "Data" sheet
    df_indep = load_excel_file(indep_path, "Data")

    # Ensure the configured date column exists and use it as the DateTimeIndex
    date_col = series_config["Date_Column_Name"]
    if date_col not in df_indep.columns:
        raise ValueError(f"Date column '{date_col}' not found in independent data.")
    
    df_indep[date_col] = pd.to_datetime(df_indep[date_col])
    df_indep = df_indep.set_index(date_col).sort_index()

    # Standardize to configured frequency (one row per period)
    freq = str(series_config["Frequency"]).upper()
    df_indep = standardize_series_to_freq(df_indep, freq)

    # Sanitize column names
    df_indep, col_map =  sanitize_column_names(df_indep)
    
    # =====================================================================
    # 2) Load Selected Features (from variable selection step)
    # =====================================================================
    variables_selected_file_path = os.path.join(
        variables_selected_folder_name,
        "selected_features.csv",
    )
    if not os.path.isfile(variables_selected_file_path):
        raise FileNotFoundError(
            f"Required file '{variables_selected_file_path}' does not exist."
        )
    
    df_variables_selected = pd.read_csv(variables_selected_file_path)
    logger.info("Total variables for backtesting: %d", df_variables_selected.shape[0])

    # List of independent variables (drivers) to evaluate
    indep_cols = list(df_variables_selected["Variable_Name"])

    # =====================================================================
    # Data structures for indep variable models-ensemble error tracking
    # =====================================================================
    # For each variable and method (ETS, ARIMA, Theta), track:
    #   - cumulative absolute percentage error (err)
    #   - number of points contributing to that error (count)
    indep_error_sums = {
        col: {
            "ets":   {"err": 0.0, "count": 0},
            "arima": {"err": 0.0, "count": 0},
            "theta": {"err": 0.0, "count": 0},
        }
        for col in indep_cols
    }

    # =====================================================================
    # 3) Backtesting for each independent variable and calculate 
    # weights across three univariate time-series models: ETS, ARIMA, Theta
    # =====================================================================
    indep_weights_rows = []

    # Horizon in *periods* at configured frequency
    model_sel_period = int(series_config.get("Model_Selection_Period"))

    for col in indep_cols:

        col = str(col).strip()
        logger.info("-" * 80)
        logger.info("# Independent Variable: %s", col)

        # =====================================================================
        # 3.1) Sanity check: variable must exist in indep data
        # =====================================================================
        if col not in df_indep.columns:
            logger.warning(
                "Variable '%s' not present in independent dataset. Skipping.", col
            )
            continue
        
        # Extract historical series for this variable
        df_col = df_indep[[col]]
    
        # Identify first and last non-null indices
        var_first_valid = df_col[col].first_valid_index()
        if var_first_valid is None:
            logger.warning(
                "Selected variable '%s' has no non-null values. Skipping.", col
            )
            continue
        logger.info("# First valid index: %s", var_first_valid)

        var_last_valid = df_col[col].last_valid_index()
        logger.info("# Last valid index: %s", var_last_valid)
    
        # =====================================================================
        # 3.2) Restrict to valid range and interpolate gaps
        # =====================================================================
        # Keep only the segment between first and last non-null values
        df_col_clean = time_interpolate(df_col.loc[var_first_valid:var_last_valid])
    
        # Convert to numeric and drop any remaining NaNs
        var_hist_series = pd.to_numeric(df_col_clean[col], errors="coerce").dropna()
        var_hist_series = standardize_series_to_freq(var_hist_series, freq)
    
        # Total number of usable observations for this variable
        total_len = len(var_hist_series)
        logger.info("# Total observations (periods): %d", total_len)

        # =====================================================================
        # 3.3) Rolling holdout backtest for this variable
        # =====================================================================

        # Horizon = model-selection period
        n_holdout = model_sel_period
       
        # Use at most 2 folds to bound compute
        n_folds = 2

        # Start index for earliest holdout segment
        start_idx = total_len - n_folds * n_holdout

        if start_idx <= 0:
            logger.warning(
                "Not enough history for rolling holdout for variable '%s'. Skipping.", col
            )
            continue

        for fold_idx in range(n_folds):
            # Indices that define the holdout slice
            fold_holdout_start = start_idx + fold_idx * n_holdout
            fold_holdout_end = fold_holdout_start + n_holdout

            # Split into train and holdout portions
            series_train = var_hist_series.iloc[:fold_holdout_start]
            series_holdout = var_hist_series.iloc[fold_holdout_start:fold_holdout_end]
            train_end_date = series_train.index.max()

            logger.info(f"[{col}] Fold {fold_idx + 1}/{n_folds}")
            logger.info(
                f"[{col}] Train period:   {series_train.index.min().date()} "
                f"-> {series_train.index.max().date()}"
            )
            logger.info(
                f"[{col}] Holdout period: {series_holdout.index.min().date()} "
                f"-> {series_holdout.index.max().date()}"
            )

            # =====================================================================
            # 3.4) Train ETS/ARIMA/Theta on TRAIN and forecast HOLDOUT
            # =====================================================================
            try:
                # Fit & forecast each component model. Returns:
                #   {"ets": np.ndarray, "arima": np.ndarray, "theta": np.ndarray}
                forecast_indep_dict = forecast_indep_variables(
                    series=series_train,
                    horizon=n_holdout,
                    freq=freq,
                )

                
                actual_indep = series_holdout

                # Build method-wise forecast series and accumulate errors
                for method_name, arr in forecast_indep_dict.items():
                    s_fc = pd.Series(
                        arr,
                        index=pd.period_range(start=train_end_date.to_period(freq) + 1, periods=n_holdout, freq=freq).to_timestamp(),
                        name=f"{col}_{method_name}",
                    ).reindex(series_holdout.index)

                    # Only compute APE where both actual and forecast are valid and actual != 0
                    mask = actual_indep.notna() & s_fc.notna() & (actual_indep != 0)
                    if mask.any():
                        ape = (actual_indep[mask] - s_fc[mask]).abs() / actual_indep[mask].abs()

                        # Update cumulative error + count for this variable & method
                        indep_error_sums[col][method_name]["err"] += ape.sum()
                        indep_error_sums[col][method_name]["count"] += mask.sum()

            except Exception as e:
                # If ETS/ARIMA/Theta fail for this fold, just log it.
                # We still compute weights from whatever folds/methods succeed.
                logger.warning(
                    f"[{series_name}][Fold {fold_idx + 1}] Could not forecast independent variable {col} "
                    f"with ensemble components: {e}"
                )

        # =====================================================================
        # 3.5) After all folds for this variable: derive MAPE and weights
        # =====================================================================
        info = indep_error_sums.get(col, {})

        # Cumulative errors + counts across all folds for this variable
        ets_err,   ets_count   = info["ets"]["err"],   info["ets"]["count"]
        arima_err, arima_count = info["arima"]["err"], info["arima"]["count"]
        theta_err, theta_count = info["theta"]["err"], info["theta"]["count"]

        # Convert cumulative APE to MAPE per method (protect against division by zero)
        mape_ets   = ets_err   / ets_count   if ets_count   > 0 else np.nan
        mape_arima = arima_err / arima_count if arima_count > 0 else np.nan
        mape_theta = theta_err / theta_count if theta_count > 0 else np.nan

        # Inverse-MAPE weighting: lower MAPE => larger inverse => higher weight
        inv = {}
        for name, m in [("ets", mape_ets), ("arima", mape_arima), ("theta", mape_theta)]:
            inv[name] = 1.0 / m if (m is not None and not np.isnan(m) and m > 0) else 0.0

        total_inv = inv["ets"] + inv["arima"] + inv["theta"]
        
        # If we cannot compute any sensible inverse-MAPE, fall back to equal weights
        if total_inv <= 0:
            w_ets = w_arima = w_theta = 1.0 / 3.0
        else:
            w_ets = inv["ets"]   / total_inv
            w_arima = inv["arima"] / total_inv
            w_theta = inv["theta"] / total_inv

        # Append row for this variable to the final weights table
        indep_weights_rows.append({
            "Indep_Var": col,
            "MAPE_ETS": mape_ets,
            "MAPE_ARIMA": mape_arima,
            "MAPE_THETA": mape_theta,
            "Weight_ETS": w_ets,
            "Weight_ARIMA": w_arima,
            "Weight_THETA": w_theta,
            "Count_ETS": ets_count,
            "Count_ARIMA": arima_count,
            "Count_THETA": theta_count,
        })

    # ======================================================================
    # 4) Save indep variables ensemble weights per variable (all variables)
    # ======================================================================
    indep_weights_df = pd.DataFrame(indep_weights_rows)
    indep_weights_file = os.path.join(model_sel_dir, INDEP_ENSEMBLE_WEIGHTS_SUFFIX)
    indep_weights_df.to_csv(indep_weights_file, index=False, float_format=CSV_FLOAT_FMT)
    logger.info(
        f"[{series_name}] Exogenous ensemble weights saved to {indep_weights_file}"
    )

    return indep_weights_df


# =====================================================================
# Future forecast for independent variables (generic + W/M/Y)
# =====================================================================

def get_indep_future_forecast(
    input_folder_name: str, 
    series_name: str, 
    series_config: dict, 
    variables_selected_folder_name: str, 
    model_sel_dir: str, 
    df_series: pd.DataFrame,
    logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Generate future forecasts for each selected independent variable (driver)
    using previously computed ensemble weights (ETS/ARIMA/Theta).

    Frequency-aware & series-agnostic:
      - Works for Frequency ∈ {M, W, Y}.
      - Interprets Forecasting_Period as number of *periods*.
      - The "historical + future" window is anchored so that the last
        historical period always coincides with the latest dependent date.

    Outputs
    -------
    indep_future_df : pd.DataFrame
        Index:
            future periods only (length = Forecasting_Period)
        Columns:
            selected independent variables

    indep_historical_future_df : pd.DataFrame
        Index:
            [history_window ... latest dependent date] + [future horizon]
        Length:
            hist_periods + Forecasting_Period
        Columns:
            same as indep_future_df

        The historical window spans approximately 2 years (based on
        HIST_INDEP_NUM_MONTHS), converted to periods in the configured
        frequency:
            - Monthly: ~24 months
            - Weekly : ~104 weeks
            - Yearly : ~2 years
        and always ends at the latest dependent-date period.
    """

    # =====================================================================
    # 1) Load independent variables ensemble weights (if available)
    # =====================================================================
    indep_weights_path = os.path.join(model_sel_dir, INDEP_ENSEMBLE_WEIGHTS_SUFFIX)
    indep_weights_dict: Dict[str, Dict[str, float]] = {}

    if not os.path.isfile(indep_weights_path):
        raise FileNotFoundError(
            f"Independent-variable weights file not found at: {indep_weights_path}"
        )

    indep_w_df = pd.read_csv(indep_weights_path)
    for _, row in indep_w_df.iterrows():
        var = str(row["Indep_Var"])
        indep_weights_dict[var] = {
            "ets": row.get("Weight_ETS", np.nan),
            "arima": row.get("Weight_ARIMA", np.nan),
            "theta": row.get("Weight_THETA", np.nan),
        }
    logger.info(
        "[%s] Loaded independent-variable ensemble weights from %s",
        series_name,
        indep_weights_path,
    )

    # =====================================================================
    # 2) Load independent data (from master input) and standardize freq
    # =====================================================================
    indep_file = series_config.get("Input_File_Name")
    if not indep_file:
        raise ValueError("Input file name missing for independent variables.")

    indep_path = os.path.join(input_folder_name, indep_file)
    if not os.path.isfile(indep_path):
        raise FileNotFoundError(
            f"Required file '{indep_file}' does not exist in '{input_folder_name}'."
        )

    df_indep = load_excel_file(indep_path, "Data")

    date_col = series_config["Date_Column_Name"]
    if date_col not in df_indep.columns:
        raise ValueError(f"Date column '{date_col}' not found in independent data.")
    
    df_indep[date_col] = pd.to_datetime(df_indep[date_col])
    df_indep = df_indep.set_index(date_col).sort_index()

    # Config frequency + metadata
    freq = str(series_config.get("Frequency", "M")).upper()
    ts_freq, seasonal_period = ts_freq_from_code(freq)
    forecasting_periods = int(series_config["Forecasting_Period"])
    hist_periods = _compute_hist_periods(freq)

    # Standardize independent data to one row per period
    df_indep = standardize_series_to_freq(df_indep, freq)

    # Sanitize column names (important!)
    df_indep, col_map = sanitize_column_names(df_indep)

    # =====================================================================
    # 3) Load Dependent Data (for max date reference)
    # =====================================================================
    if not isinstance(df_series.index, pd.DatetimeIndex):
        raise TypeError("df_series index must be a DatetimeIndex.")

    df_dep_idx = df_series.sort_index()
    df_dep_idx = standardize_series_to_freq(df_dep_idx, freq)

    dep_latest_date = df_dep_idx.index.max()
    dep_latest_period = dep_latest_date.to_period(freq)
    logger.info("Latest dependent-date period available: %s", str(dep_latest_period))
    
    # =====================================================================
    # 4) Load Selected Features (from variable selection step)
    # =====================================================================
    variables_selected_file_path = os.path.join(
        variables_selected_folder_name,
        "selected_features.csv",
    )

    if not os.path.isfile(variables_selected_file_path):
        raise FileNotFoundError(
            f"Required file '{variables_selected_file_path}' does not exist."
        )
    
    df_variables_selected = pd.read_csv(variables_selected_file_path)
    logger.info(
        "Total independent variables for future forecasting: %d",
        df_variables_selected.shape[0],
    )

    # List of independent variables (drivers) to forecast
    indep_cols = [str(x).strip() for x in df_variables_selected["Variable_Name"]]

    # =====================================================================
    # 5) Prepare containers for per-variable forecasts
    # =====================================================================

    # Container A: future-only forecasts for requested horizon
    indep_future_forecasts: Dict[str, pd.Series] = {}
    
    # Container B: latest history (ending at dep_latest_date)
    # + future values for requested horizon
    indep_hist_future_forecasts: Dict[str, pd.Series] = {}

    # Pre-compute common future index (based on latest dependent period)
    future_periods = pd.period_range(
        start=dep_latest_period + 1,
        periods=forecasting_periods,
        freq=freq,
    )

    future_idx = future_periods.to_timestamp()

    # History index: last `hist_periods` periods ending at dep_latest_period
    hist_start_period = dep_latest_period - (hist_periods - 1)
    hist_period_range = pd.period_range(
        start=hist_start_period,
        end=dep_latest_period,
        freq=freq,
    )
    hist_idx = hist_period_range.to_timestamp()
    hist_future_idx = hist_idx.append(future_idx)

    logger.info(
        "History window length (periods): %d | Future horizon (periods): %d",
        hist_periods,
        forecasting_periods,
    )

    # =====================================================================
    # 6) Per-variable future forecasting
    # =====================================================================
    for col in indep_cols:

        col = str(col).strip()
        logger.info("-" * 80)
        logger.info("# Independent variable: %s", col)

        if col not in df_indep.columns:
            logger.warning(
                "Variable '%s' not present in independent dataset (after sanitization). Skipping.",
                col,
            )
            continue
        
        # Extract historical series for this variable
        df_col = df_indep[[col]]

        # Identify first and last non-null indices
        var_first_valid = df_col[col].first_valid_index()
        if var_first_valid is None:
            logger.warning(
                "Selected variable '%s' has no non-null values. Skipping.", col
            )
            continue
        logger.info("# First valid index: %s", var_first_valid)

        var_last_valid = df_col[col].last_valid_index()
        logger.info("# Last valid index: %s", var_last_valid)

        # Restrict to valid range and interpolate gaps
        df_col_clean = time_interpolate(df_col.loc[var_first_valid:var_last_valid])
        var_hist_series = pd.to_numeric(df_col_clean[col], errors="coerce").dropna()

        if var_hist_series.empty:
            logger.warning(
                "Selected variable '%s' has no numeric history after cleaning. Skipping.",
                col,
            )
            continue

        # Standardize to configured frequency once more (safety)
        var_hist_series = standardize_series_to_freq(var_hist_series, freq)

        hist_series_end_date = var_hist_series.index.max()
        hist_series_end_period = hist_series_end_date.to_period(freq)

        # Compute effective forecast horizon
        gap_periods = dep_latest_period.ordinal - hist_series_end_period.ordinal
        logger.info(
            "Gap (dep_latest_period - indep_last_period) for %s: %d period(s).",
            col,
            gap_periods,
        )

        # We need coverage at least up to:
        #   last_needed_period = dep_latest_period + forecasting_periods
        last_needed_period = dep_latest_period + forecasting_periods
        steps_to_forecast = last_needed_period.ordinal - hist_series_end_period.ordinal
        
        # Build future forecasts (or reuse existing coverage)
        if steps_to_forecast <= 0:
            logger.info(
                "No additional model-based forecast required for %s. "
                "Existing series already covers up to required last period.",
                col,
            )
            final_series = var_hist_series.copy()

        else:
            logger.info(
                "Forecasting %d future period(s) for %s to cover gap + horizon.",
                steps_to_forecast,
                col,
            )
            try:
                w = indep_weights_dict.get(col, None)

                fc_future = forecast_indep_ensemble(
                    series=var_hist_series,
                    last_date=hist_series_end_date,
                    horizon=steps_to_forecast,
                    weights=w,
                    freq=freq,
                )
            except Exception as e:
                logger.warning(
                    "[%s] Could not forecast future independent variable %s with ensemble: %s. "
                    "Using flat (last-value) forecast.",
                    series_name,
                    col,
                    e,
                )

                future_p = pd.period_range(
                    start=hist_series_end_period + 1,
                    periods=steps_to_forecast,
                    freq=freq,
                )
                future_index = future_p.to_timestamp()

                last_val = var_hist_series.iloc[-1]
                fc_future = pd.Series(
                    [last_val] * steps_to_forecast,
                    index=future_index,
                    name=col,
                )
            
            final_series = pd.concat([var_hist_series, fc_future])
            final_series = final_series[~final_series.index.duplicated(keep="last")]

        # Future-only slice (always forecasting_periods rows)
        future_slice = final_series.reindex(future_idx)

        # History + future window, anchored at dep_latest_period
        hist_future_series = final_series.reindex(hist_future_idx)

        indep_future_forecasts[col] = future_slice
        indep_hist_future_forecasts[col] = hist_future_series

    # =====================================================================
    # 6) Build final DataFrames and align columns with df_series
    # =====================================================================
    indep_future_df = pd.DataFrame(indep_future_forecasts, index=future_idx)
    indep_historical_future_df = pd.DataFrame(
        indep_hist_future_forecasts,
        index=hist_future_idx,
    )

    logger.info(
        "[%s] Independent-variable future forecasts prepared "
        "(future-only + history+future window).",
        series_name,
    )

    # Align column order to df_series where possible
    common_cols = [c for c in df_series.columns if c in indep_future_df.columns]
    indep_future_df = indep_future_df[common_cols]
    indep_historical_future_df = indep_historical_future_df[common_cols]

    return indep_future_df, indep_historical_future_df
