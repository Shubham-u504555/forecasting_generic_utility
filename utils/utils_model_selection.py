# utils_model_selection.py
# =====================================================================
# Target model selection (dependent variable) + supporting utilities
# for reading config and selected variables.
#
# This module covers:
#
# 1) Config / feature selection helpers:
#    - fetch_commodities_for_model_selection
#    - fetch_selected_variables
#
# 2) Target model selection for a time series:
#    - Rolling-origin multi-fold holdout evaluation
#    - Trains multiple target models 
#       - Multivariate (LGBM, ElasticNet, CatBoost, XGBoost, SARIMAX)
#       - Univariate (Holt-Winters, Theta, NHITS, NBEATS, TFT)
#    - Uses Forecast Quality Score (FQS) based on:
#          DA, MAPEScore, WBA
#    - Selects top-K models and builds an FQS-weighted ensemble
#    - Computes independent models ensemble weights (ETS/ARIMA/Theta)
#      per variable
#    - Saves metrics, holdout forecasts, plots, and model weights
# =====================================================================

from typing import Dict, Any

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error

from utils.utils_common import (
    sanitize_column_names,
    sanitize_column_values
)

from settings import (
    WEIGHT_DA,
    WEIGHT_MAPESCORE,
    WEIGHT_WBA,
    TOP_K_ENSEMBLE,
    HOLDOUT_RESULTS_SUFFIX,
    HOLDOUT_METRICS_SUFFIX,
    HOLDOUT_INDEP_SUFFIX,
    HOLDOUT_PLOT_SUFFIX,
    SELECTED_MODELS_SUFFIX,
    PER_FOLD_HOLDOUT_METRICS_SUFFIX,
    N_LAGS_TARGET,
    RANDOM_STATE,
    # Target model enable flags (multivariate / univariate)
    ENABLE_TARGET_LGBM,
    ENABLE_TARGET_ELASTICNET,
    ENABLE_TARGET_CATBOOST,
    ENABLE_TARGET_XGBOOST,
    ENABLE_TARGET_SARIMAX,
    ENABLE_TARGET_HOLTWINTERS,
    ENABLE_TARGET_THETA,
    ENABLE_TARGET_NHITS,
    ENABLE_TARGET_NBEATS,
    ENABLE_TARGET_TFT,
    ENABLE_TARGET_LSTM,
    ENABLE_TARGET_PROPHET,
    # Additional ML models
    ENABLE_TARGET_LASSO,
    ENABLE_TARGET_RF,
    ENABLE_TARGET_ANN,
    ENABLE_TARGET_SVR,
    # Econometric panel models
    ENABLE_TARGET_TWFE,
    ENABLE_TARGET_SYSGMM,
    ENABLE_TARGET_MG,
    ENABLE_TARGET_AMG,
    ENABLE_TARGET_CCEMG,
    ENABLE_TARGET_DCCEMG,
)

from utils.utils_dep_model_helpers import (
    make_target_features,
    build_lgbm_model,
    build_elasticnet_model,
    build_catboost_model,
    build_xgboost_model,
    build_sarimax_model,
    # Additional ML model builders
    build_lasso_model,
    build_rf_model,
    build_ann_model,
    build_svr_model,
    # Econometric estimator classes
    TWFEEstimator,
    SysGMMEstimator,
    MGEstimator,
    AMGEstimator,
    CCEMGEstimator,
    DCCEMGEstimator,
    # Prophet
    fit_prophet,
    # Shared runner / metric helpers
    evaluate_regression_forecast,
    log_holdout_metrics,
    compute_directional_and_band_accuracy,
    run_recursive_multivariate_log_model,
    run_univariate_statistical_log_model,
    run_univariate_neural_log_model,
    get_neural_input_size_and_downsample,
)

from utils.utils_indep_model_helpers import (
    get_indep_ensemble_weights,
    forecast_indep_ensemble,
    ts_freq_from_code
)

from neuralforecast.losses.pytorch import MAPE
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM

warnings.filterwarnings("ignore")


def fetch_selected_variables(
    df: pd.DataFrame,
    variables_selected_folder_name: str,
) -> pd.DataFrame:
    """Filter a DataFrame to only the variables selected in Step-1.

    Expected file:
        <variables_selected_folder_name>/selected_features.csv

    This CSV must contain at least a column "Variable_Name".
    """
    variables_selected_file_path = os.path.join(
        variables_selected_folder_name,
        "selected_features.csv",
    )
    if not os.path.isfile(variables_selected_file_path):
        raise FileNotFoundError(
            f"Required file '{variables_selected_file_path}' does not exist."
        )

    df_variables_selected = pd.read_csv(variables_selected_file_path, encoding="utf-8-sig")
    df_variables_selected, _ = sanitize_column_values(df_variables_selected, "Variable_Name")
    list_variables_selected = [x.strip() for x in df_variables_selected["Variable_Name"].tolist()]

    # Keep intersection in case config and dataframe are slightly misaligned
    df_selected = df[df.columns.intersection(list_variables_selected)]
    return df_selected

# =====================================================================
# Target model selection helpers and core routine
# =====================================================================

# Target variable models that explicitly use independent variables (multivariate)
MULTIVARIATE_TARGET_MODELS = {
    "lgbm", "elasticnet", "sarimax", "catboost", "xgboost",
    # Additional ML models
    "lasso", "rf", "ann", "svr",
    # Econometric models (all use independent variables)
    "twfe", "sysgmm", "mg", "amg", "ccemg", "dccemg",
}

def execute_model_selection(
    df_series,
    input_folder_name,
    series_name,
    series_config,
    variables_selected_folder_name,
    output_folder_name,
    logger,
):
    """
    Rolling-origin multi-holdout evaluation for a time series.

    Uses up to 2 non-overlapping holdout windows of length
    Model_Selection_Period. Aggregates metrics across folds and selects top
    target forecast models using a Forecast Quality Score (FQS):

        MAPEScore_m = 1 / (1 + MAPE_m)

        FQS_m       = WEIGHT_DA * DA_m               (Directional Accuracy)
                    + WEIGHT_MAPESCORE * MAPEScore_m  (Mape Based Accuracy)
                    + WEIGHT_WBA * WBA_m         (Within Band 10% Accuracy)     

    The final ensemble uses FQS-based weights over the top-K models
    (with at least one multivariate target model forced into the ensemble
    if such models are available).

    NOTE
    ----
    Individual model families can be toggled on/off globally via the
    ENABLE_TARGET_* flags in settings.py. Disabled models are completely
    skipped during training and evaluation.
    """
    
    # Get target column name
    target_col_name = series_config.get("Target_Column_Name")

    # Get configured frequency (M = monthly, W = weekly, Y = yearly)
    freq = str(series_config["Frequency"])  # "M", "W", or "Y"

    # Map config frequency to pandas time series freq and seasonal_period
    ts_freq, seasonal_period  = ts_freq_from_code(freq)

    # Create per-series output directory for all artifacts
    model_sel_dir = output_folder_name
    os.makedirs(model_sel_dir, exist_ok=True)
    logger.info(f"Model Selection Output directory: {model_sel_dir} (freq={freq})")

    # ======================================================================
    # 1) For each independent variable, calculate weights across three 
    # univariate time-series models: ETS, ARIMA, Theta
    # ======================================================================
    indep_weights_df = get_indep_ensemble_weights(
        input_folder_name=input_folder_name,
        series_name=series_name,
        series_config=series_config,
        variables_selected_folder_name=variables_selected_folder_name,
        model_sel_dir=model_sel_dir,
        logger=logger,
    )

    indep_weights_dict: Dict[str, Dict[str, float]] = {}
    for _, row in indep_weights_df.iterrows():
        var = row["Indep_Var"]
        indep_weights_dict[var] = {
            "ets": row.get("Weight_ETS", np.nan),
            "arima": row.get("Weight_ARIMA", np.nan),
            "theta": row.get("Weight_THETA", np.nan),
        }

    # =====================================================================
    # 2) Column sanitization and log-target preparation
    # =====================================================================
    df_series, col_map = sanitize_column_names(df_series)
    if target_col_name not in col_map:
        logger.error(
            f"[{series_name}] Target column '{target_col_name}' not found. "
            f"Available: {list(col_map.keys())}"
        )
        return

    sanitized_target = col_map[target_col_name]
    df_series = df_series.rename(columns={sanitized_target: "target"})

    # Work in log space for models
    df_series["target_log"] = np.log(df_series["target"])

    # -----------------------------------------------------------------
    # Standardize index to the configured frequency as canonical period
    # start. We:
    #   - convert timestamps -> PeriodIndex(freq)
    #   - aggregate by period (take last obs if multiple in same period)
    #   - convert back to Timestamp at period start.
    # This works for:
    #   freq = "M" (monthly), "W" (weekly), "Y" (yearly)
    # -----------------------------------------------------------------
    periods = df_series.index.to_period(freq)
    df_series_final = df_series.groupby(periods).last()
    df_series_final.index = df_series_final.index.to_timestamp()

    # Get count of periods to be used for model selection process
    Model_Selection_Period = int(series_config["Model_Selection_Period"])

    # Basic data sufficiency check
    if len(df_series_final) <= Model_Selection_Period + N_LAGS_TARGET + 6:
        logger.warning(
            f"[{series_name}] Not enough data for robust training and "
            f"{Model_Selection_Period} {freq}-period holdout. Rows available: {len(df_series_final)}"
        )
        return
    
    logger.info(
        f"[{series_name}] Date range: {df_series_final.index.min().date()} "
        f"to {df_series_final.index.max().date()}"
    )

    # All non-target columns are treated as independent drivers
    indep_cols = [c for c in df_series_final.columns if c not in ["target", "target_log"]]
    logger.info(f"[{series_name}] Exogenous columns: {indep_cols}")

    # Log which target models are enabled for this run
    logger.info(
        f"[{series_name}] Enabled target models → "
        f"Multivariate: "
        f"LGBM={ENABLE_TARGET_LGBM}, "
        f"ElasticNet={ENABLE_TARGET_ELASTICNET}, "
        f"CatBoost={ENABLE_TARGET_CATBOOST}, "
        f"XGBoost={ENABLE_TARGET_XGBOOST}, "
        f"SARIMAX={ENABLE_TARGET_SARIMAX}, "
        f"LASSO={ENABLE_TARGET_LASSO}, "
        f"RF={ENABLE_TARGET_RF}, "
        f"ANN={ENABLE_TARGET_ANN}, "
        f"SVR={ENABLE_TARGET_SVR} | "
        f"Econometric: "
        f"TWFE={ENABLE_TARGET_TWFE}, "
        f"sysGMM={ENABLE_TARGET_SYSGMM}, "
        f"MG={ENABLE_TARGET_MG}, "
        f"AMG={ENABLE_TARGET_AMG}, "
        f"CCEMG={ENABLE_TARGET_CCEMG}, "
        f"DCCEMG={ENABLE_TARGET_DCCEMG} | "
        f"Univariate-Stat: HoltWinters={ENABLE_TARGET_HOLTWINTERS}, "
        f"Theta={ENABLE_TARGET_THETA} | "
        f"Univariate-Neural: NHITS={ENABLE_TARGET_NHITS}, "
        f"NBEATS={ENABLE_TARGET_NBEATS}, "
        f"TFT={ENABLE_TARGET_TFT}, "
        f"LSTM={ENABLE_TARGET_LSTM} | "
        f"Univariate-StatExtended: Prophet={ENABLE_TARGET_PROPHET}"
    )

    # =====================================================================
    # 3) Rolling-origin multi-fold setup
    # =====================================================================
    n_holdout = Model_Selection_Period
    total_len = len(df_series_final)

    # Maximum possible folds respecting lags and a small training buffer
    max_possible_folds = (total_len - (N_LAGS_TARGET + 6)) // n_holdout
    if max_possible_folds < 1:
        logger.warning(
            f"[{series_name}] Not enough data for even one full holdout window of size {n_holdout} "
            f"with required lags. Rows available: {len(df_series_final)}"
        )
        return

    # Use at most 2 folds to keep compute bounded
    n_folds = min(2, max_possible_folds)
    logger.info(
        f"[{series_name}] Using {n_folds} rolling holdout window(s), "
        f"each of length {n_holdout} {freq}-periods."
    )

    # =====================================================================
    # 4) Loop over folds
    # =====================================================================

    # Aggregated metrics over folds
    aggregated_metrics: dict = {}

    # Keep references to latest fold data for saving plots/CSVs
    latest_df_train = None
    latest_df_holdout = None
    latest_indep_holdout_df = None
    latest_model_holdout_forecasts = None
    latest_actual_holdout_price = None

    # Per-fold metrics rows (for CSV)
    per_fold_rows = []
    
    # Start index of the earliest holdout used in rolling scheme
    start_idx = total_len - n_folds * n_holdout

    for fold_idx in range(n_folds):
        # Determine train/holdout split indices for this fold
        fold_holdout_start = start_idx + fold_idx * n_holdout
        fold_holdout_end = fold_holdout_start + n_holdout

        df_train = df_series_final.iloc[:fold_holdout_start]
        df_holdout = df_series_final.iloc[fold_holdout_start:fold_holdout_end]
        train_end_date = df_train.index.max()

        logger.info("-" * 80)
        logger.info(f"[{series_name}] Fold {fold_idx + 1}/{n_folds}")
        logger.info(
            f"[{series_name}] Train period:   {df_train.index.min().date()} -> {df_train.index.max().date()}"
        )
        logger.info(
            f"[{series_name}] Holdout period: {df_holdout.index.min().date()} -> {df_holdout.index.max().date()}"
        )

        # =====================================================================
        # 4.1) Forecasts independent variables for holdout dataset
        # =====================================================================
        indep_holdout_forecasts = {}

        for col in indep_cols:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Forecasting holdout for independent variable: {col}"
            )
            series_train = df_train[col]
            try:
                # Components: ETS, ARIMA, Theta
                w = indep_weights_dict.get(col, None)

                fc_holdout = forecast_indep_ensemble(
                    series=series_train,
                    last_date=train_end_date,
                    horizon=n_holdout,
                    weights=w,
                    freq=freq
                )
                indep_holdout_forecasts[col] = fc_holdout
            except Exception as e:
                # Fallback: flat (last-value) independent variable forecast
                logger.warning(
                    f"[{series_name}] Could not forecast independent variable {col} "
                    f"with ensemble: {e}. Using flat forecast.",
                )

                # Build holdout index as the next n_holdout periods at the
                # configured frequency, starting right after train_end_date.
                last_period = train_end_date.to_period(freq)
                future_periods = pd.period_range(
                    start=last_period + 1,
                    periods=n_holdout,
                    freq=freq,
                )
                holdout_index = future_periods.to_timestamp()

                last_val = series_train.iloc[-1]
                fc_holdout = pd.Series(
                    [last_val] * n_holdout,
                    index=holdout_index,
                    name=col,
                )
                indep_holdout_forecasts[col] = fc_holdout


        # DataFrame of independent variable forecasts for this fold
        indep_holdout_df = pd.DataFrame(indep_holdout_forecasts)
        logger.info(
            f"[{series_name}][Fold {fold_idx + 1}] Exogenous holdout forecasts prepared."
        )

        # =====================================================================
        # 4.2) Create features for multivariate target models (log space)
        # =====================================================================
        X_train, y_train_log = make_target_features(
            df_train,
            target_col="target_log",
            indep_columns=indep_cols,
            freq=freq,   # now calendar and rolling windows align with M/W/Y
        )
        logger.info(
            f"[{series_name}][Fold {fold_idx + 1}] Target training samples: {X_train.shape[0]}, "
            f"features: {X_train.shape[1]}"
        )

        actual_holdout_price = df_holdout["target"]
        model_holdout_forecasts: Dict[str, pd.Series] = {}
        model_metrics_fold: Dict[str, Dict[str, float]] = {}

        # ======================================================================
        # 4.3) Multivariate target models (LGBM, ENet, CatBoost, XGBoost, SARIMAX)
        # ======================================================================

        # ------------------------------------------
        # Train LGBM and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_LGBM:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training LGBM model (log-target)..."
            )

            lgbm_model = build_lgbm_model()
            
            fc_lgbm, mt_lgbm = run_recursive_multivariate_log_model(
                model=lgbm_model,
                model_key="lgbm",
                model_label="LGBM",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,       # <-- new
                fit_kwargs={"eval_metric": "mape"},
            )

            if fc_lgbm is None:
                # If LGBM fails completely (when enabled), the fold is not reliable
                logger.warning(
                    f"[{series_name}][Fold {fold_idx + 1}] LGBM did not produce forecast. Skipping fold."
                )
                continue

            model_holdout_forecasts["lgbm"] = fc_lgbm
            model_metrics_fold["lgbm"] = mt_lgbm
            log_holdout_metrics(logger, series_name, fold_idx, "LGBM", mt_lgbm)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping LGBM (ENABLE_TARGET_LGBM=False)."
            )

        # ------------------------------------------
        # Train ElasticNet and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_ELASTICNET:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training ElasticNet model (log-target)..."
            )

            enet_model = build_elasticnet_model()
            
            fc_enet, mt_enet = run_recursive_multivariate_log_model(
                model=enet_model,
                model_key="elasticnet",
                model_label="ElasticNet",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,       # <-- new
            )
            
            if fc_enet is not None:
                model_holdout_forecasts["elasticnet"] = fc_enet
                model_metrics_fold["elasticnet"] = mt_enet
                log_holdout_metrics(logger, series_name, fold_idx, "ElasticNet", mt_enet)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping ElasticNet (ENABLE_TARGET_ELASTICNET=False)."
            )

        # ------------------------------------------
        # Train CatBoost and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_CATBOOST:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training CatBoostRegressor (log-target)..."
            )

            cb_model = build_catboost_model()
            
            fc_cb, mt_cb = run_recursive_multivariate_log_model(
                model=cb_model,
                model_key="catboost",
                model_label="CatBoost",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,       # <-- new
            )

            if fc_cb is not None:
                model_holdout_forecasts["catboost"] = fc_cb
                model_metrics_fold["catboost"] = mt_cb
                log_holdout_metrics(logger, series_name, fold_idx, "CatBoost", mt_cb)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping CatBoost (ENABLE_TARGET_CATBOOST=False)."
            )

        # ------------------------------------------
        # Train XGBoost and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_XGBOOST:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training XGBRegressor (log-target)..."
            )

            xgb_model = build_xgboost_model()

            fc_xgb, mt_xgb = run_recursive_multivariate_log_model(
                model=xgb_model,
                model_key="xgboost",
                model_label="XGBoost",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,       # <-- new
            )

            if fc_xgb is not None:
                model_holdout_forecasts["xgboost"] = fc_xgb
                model_metrics_fold["xgboost"] = mt_xgb
                log_holdout_metrics(logger, series_name, fold_idx, "XGBoost", mt_xgb)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping XGBoost (ENABLE_TARGET_XGBOOST=False)."
            )

        # ------------------------------------------
        # Train SARIMAX and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_SARIMAX:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training SARIMAX model (log-target, with independent variables)..."
            )

            try:
                sarimax_model = build_sarimax_model(
                    df_train=df_train, 
                    target_col="target_log", 
                    indep_cols=indep_cols,
                    seasonal_period=seasonal_period,   # <- pass here
                )

                sarimax_res = sarimax_model.fit(disp=False)

                sarimax_forecast_log = sarimax_res.get_forecast(
                    steps=n_holdout,
                    exog=indep_holdout_df,
                ).predicted_mean

                forecast_sarimax = pd.Series(
                    np.exp(sarimax_forecast_log.values),
                    index=df_holdout.index,
                    name="Forecast_sarimax",
                )

                mt_sarimax = evaluate_regression_forecast(actual_holdout_price, forecast_sarimax)
                model_holdout_forecasts["sarimax"] = forecast_sarimax
                model_metrics_fold["sarimax"] = mt_sarimax
                log_holdout_metrics(logger, series_name, fold_idx, "SARIMAX", mt_sarimax)
            except Exception as e:
                logger.warning(f"[{series_name}][Fold {fold_idx + 1}] SARIMAX failed: {e}")
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping SARIMAX (ENABLE_TARGET_SARIMAX=False)."
            )

        # ======================================================================
        # 4.4) Univariate statistical based target models (Holt-Winters, Theta)
        # ======================================================================

        # --------------------------------------------
        # Train Holt-Winters and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_HOLTWINTERS:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training Holt-Winters (ExponentialSmoothing) on target_log..."
            )

            # Index has already been standardized to the configured freq
            y_train_hw_log = df_train["target_log"]
            
            fc_hw, mt_hw = run_univariate_statistical_log_model(
                model_key="holtwinters",
                model_label="Holt-Winters",
                train_series_log=y_train_hw_log,
                df_holdout=df_holdout,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )

            if fc_hw is not None:
                model_holdout_forecasts["holtwinters"] = fc_hw
                model_metrics_fold["holtwinters"] = mt_hw
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping Holt-Winters (ENABLE_TARGET_HOLTWINTERS=False)."
            )

        # --------------------------------------------
        # Train Theta and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_THETA:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training Theta model on target_log..."
            )

            # Index has already been standardized to the configured freq
            y_train_theta_log = df_train["target_log"]
            
            fc_theta, mt_theta = run_univariate_statistical_log_model(
                model_key="theta",
                model_label="Theta",
                train_series_log=y_train_theta_log,
                df_holdout=df_holdout,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )
            if fc_theta is not None:
                model_holdout_forecasts["theta"] = fc_theta
                model_metrics_fold["theta"] = mt_theta
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping Theta (ENABLE_TARGET_THETA=False)."
            )

        # ======================================================================
        # 4.5) Univariate neural based target models (NHITS, NBEATS, TFT)
        # ======================================================================

        # --------------------------------------------
        # Train NHITS and forecasts for HOLDOUT
        # --------------------------------------------

        input_size_neural, freq_downsample = get_neural_input_size_and_downsample(
                freq=freq,
                train_len=len(df_train),
                h=n_holdout,
            )

        if ENABLE_TARGET_NHITS:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training NHITS model (log-target, univariate)..."
            )

            nhits_model = NHITS(
                h=n_holdout,
                input_size=input_size_neural,
                max_steps=300,
                n_freq_downsample=freq_downsample,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_nhits, mt_nhits = run_univariate_neural_log_model(
                model=nhits_model,
                model_key="nhits",
                model_label="NHITS",
                df_train=df_train,
                df_holdout=df_holdout,
                target_col="target_log",
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq, 
            )

            if fc_nhits is not None:
                model_holdout_forecasts["nhits"] = fc_nhits
                model_metrics_fold["nhits"] = mt_nhits
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping NHITS (ENABLE_TARGET_NHITS=False)."
            )

        # --------------------------------------------
        # Train NBEATS and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_NBEATS:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training NBEATS model (log-target, univariate)..."
            )

            nbeats_model = NBEATS(
                h=n_holdout,
                input_size=input_size_neural,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_nbeats, mt_nbeats = run_univariate_neural_log_model(
                model=nbeats_model,
                model_key="nbeats",
                model_label="NBEATS",
                df_train=df_train,
                df_holdout=df_holdout,
                target_col="target_log",
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq, 
            )

            if fc_nbeats is not None:
                model_holdout_forecasts["nbeats"] = fc_nbeats
                model_metrics_fold["nbeats"] = mt_nbeats
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping NBEATS (ENABLE_TARGET_NBEATS=False)."
            )

        # --------------------------------------------
        # Train TFT and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_TFT:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training TFT model (log-target, univariate)..."
            )

            tft_model = TFT(
                h=n_holdout,
                input_size=input_size_neural,
                hidden_size=32,
                dropout=0.2,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_tft, mt_tft = run_univariate_neural_log_model(
                model=tft_model,
                model_key="tft",
                model_label="TFT",
                df_train=df_train,
                df_holdout=df_holdout,
                target_col="target_log",
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq, 
            )

            if fc_tft is not None:
                model_holdout_forecasts["tft"] = fc_tft
                model_metrics_fold["tft"] = mt_tft
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping TFT (ENABLE_TARGET_TFT=False)."
            )

        # --------------------------------------------
        # Train LSTM and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_LSTM:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training LSTM model (log-target, univariate)..."
            )

            lstm_model = LSTM(
                h=n_holdout,
                input_size=input_size_neural,
                encoder_n_layers=2,
                encoder_hidden_size=64,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_lstm, mt_lstm = run_univariate_neural_log_model(
                model=lstm_model,
                model_key="lstm",
                model_label="LSTM",
                df_train=df_train,
                df_holdout=df_holdout,
                target_col="target_log",
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq,
            )

            if fc_lstm is not None:
                model_holdout_forecasts["lstm"] = fc_lstm
                model_metrics_fold["lstm"] = mt_lstm
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping LSTM (ENABLE_TARGET_LSTM=False)."
            )

        # --------------------------------------------
        # Train Prophet and forecasts for HOLDOUT
        # --------------------------------------------
        if ENABLE_TARGET_PROPHET:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training Prophet model on target_log..."
            )

            y_train_prophet_log = df_train["target_log"]

            fc_prophet, mt_prophet = run_univariate_statistical_log_model(
                model_key="prophet",
                model_label="Prophet",
                train_series_log=y_train_prophet_log,
                df_holdout=df_holdout,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )

            if fc_prophet is not None:
                model_holdout_forecasts["prophet"] = fc_prophet
                model_metrics_fold["prophet"] = mt_prophet
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping Prophet (ENABLE_TARGET_PROPHET=False)."
            )

        # ======================================================================
        # 4.5b) Additional ML models (LASSO, RF, ANN, SVR)
        # ======================================================================

        # ------------------------------------------
        # Train LASSO and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_LASSO:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training LASSO model (log-target)..."
            )
            lasso_model = build_lasso_model()
            fc_lasso, mt_lasso = run_recursive_multivariate_log_model(
                model=lasso_model,
                model_key="lasso",
                model_label="LASSO",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_lasso is not None:
                model_holdout_forecasts["lasso"] = fc_lasso
                model_metrics_fold["lasso"] = mt_lasso
                log_holdout_metrics(logger, series_name, fold_idx, "LASSO", mt_lasso)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping LASSO (ENABLE_TARGET_LASSO=False)."
            )

        # ------------------------------------------
        # Train Random Forest and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_RF:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training Random Forest model (log-target)..."
            )
            rf_model = build_rf_model()
            fc_rf, mt_rf = run_recursive_multivariate_log_model(
                model=rf_model,
                model_key="rf",
                model_label="RF",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_rf is not None:
                model_holdout_forecasts["rf"] = fc_rf
                model_metrics_fold["rf"] = mt_rf
                log_holdout_metrics(logger, series_name, fold_idx, "RF", mt_rf)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping RF (ENABLE_TARGET_RF=False)."
            )

        # ------------------------------------------
        # Train ANN and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_ANN:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training ANN (MLPRegressor) model (log-target)..."
            )
            ann_model = build_ann_model()
            fc_ann, mt_ann = run_recursive_multivariate_log_model(
                model=ann_model,
                model_key="ann",
                model_label="ANN",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_ann is not None:
                model_holdout_forecasts["ann"] = fc_ann
                model_metrics_fold["ann"] = mt_ann
                log_holdout_metrics(logger, series_name, fold_idx, "ANN", mt_ann)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping ANN (ENABLE_TARGET_ANN=False)."
            )

        # ------------------------------------------
        # Train SVR and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_SVR:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training SVR model (log-target)..."
            )
            svr_model = build_svr_model()
            fc_svr, mt_svr = run_recursive_multivariate_log_model(
                model=svr_model,
                model_key="svr",
                model_label="SVR",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_svr is not None:
                model_holdout_forecasts["svr"] = fc_svr
                model_metrics_fold["svr"] = mt_svr
                log_holdout_metrics(logger, series_name, fold_idx, "SVR", mt_svr)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping SVR (ENABLE_TARGET_SVR=False)."
            )

        # ======================================================================
        # 4.5c) Econometric panel models (TWFE, sysGMM, MG, AMG, CCEMG, DCCEMG)
        # ======================================================================

        # ------------------------------------------
        # Train TWFE and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_TWFE:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training TWFE model (log-target)..."
            )
            twfe_model = TWFEEstimator()
            fc_twfe, mt_twfe = run_recursive_multivariate_log_model(
                model=twfe_model,
                model_key="twfe",
                model_label="TWFE",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_twfe is not None:
                model_holdout_forecasts["twfe"] = fc_twfe
                model_metrics_fold["twfe"] = mt_twfe
                log_holdout_metrics(logger, series_name, fold_idx, "TWFE", mt_twfe)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping TWFE (ENABLE_TARGET_TWFE=False)."
            )

        # ------------------------------------------
        # Train sysGMM and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_SYSGMM:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training sysGMM (2SLS) model (log-target)..."
            )
            sysgmm_model = SysGMMEstimator()
            fc_sysgmm, mt_sysgmm = run_recursive_multivariate_log_model(
                model=sysgmm_model,
                model_key="sysgmm",
                model_label="sysGMM",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_sysgmm is not None:
                model_holdout_forecasts["sysgmm"] = fc_sysgmm
                model_metrics_fold["sysgmm"] = mt_sysgmm
                log_holdout_metrics(logger, series_name, fold_idx, "sysGMM", mt_sysgmm)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping sysGMM (ENABLE_TARGET_SYSGMM=False)."
            )

        # ------------------------------------------
        # Train MG and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_MG:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training MG (Mean Group) model (log-target)..."
            )
            mg_model = MGEstimator()
            fc_mg, mt_mg = run_recursive_multivariate_log_model(
                model=mg_model,
                model_key="mg",
                model_label="MG",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_mg is not None:
                model_holdout_forecasts["mg"] = fc_mg
                model_metrics_fold["mg"] = mt_mg
                log_holdout_metrics(logger, series_name, fold_idx, "MG", mt_mg)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping MG (ENABLE_TARGET_MG=False)."
            )

        # ------------------------------------------
        # Train AMG and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_AMG:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training AMG (Augmented Mean Group) model (log-target)..."
            )
            amg_model = AMGEstimator()
            fc_amg, mt_amg = run_recursive_multivariate_log_model(
                model=amg_model,
                model_key="amg",
                model_label="AMG",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_amg is not None:
                model_holdout_forecasts["amg"] = fc_amg
                model_metrics_fold["amg"] = mt_amg
                log_holdout_metrics(logger, series_name, fold_idx, "AMG", mt_amg)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping AMG (ENABLE_TARGET_AMG=False)."
            )

        # ------------------------------------------
        # Train CCEMG and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_CCEMG:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training CCEMG model (log-target)..."
            )
            ccemg_model = CCEMGEstimator()
            fc_ccemg, mt_ccemg = run_recursive_multivariate_log_model(
                model=ccemg_model,
                model_key="ccemg",
                model_label="CCEMG",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_ccemg is not None:
                model_holdout_forecasts["ccemg"] = fc_ccemg
                model_metrics_fold["ccemg"] = mt_ccemg
                log_holdout_metrics(logger, series_name, fold_idx, "CCEMG", mt_ccemg)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping CCEMG (ENABLE_TARGET_CCEMG=False)."
            )

        # ------------------------------------------
        # Train DCCEMG and forecasts for HOLDOUT
        # ------------------------------------------
        if ENABLE_TARGET_DCCEMG:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Training DCCEMG model (log-target)..."
            )
            dccemg_model = DCCEMGEstimator()
            fc_dccemg, mt_dccemg = run_recursive_multivariate_log_model(
                model=dccemg_model,
                model_key="dccemg",
                model_label="DCCEMG",
                df_train=df_train,
                df_holdout=df_holdout,
                indep_cols=indep_cols,
                indep_holdout_df=indep_holdout_df,
                X_train=X_train,
                y_train_log=y_train_log,
                series_name=series_name,
                fold_idx=fold_idx,
                logger=logger,
                freq=freq,
            )
            if fc_dccemg is not None:
                model_holdout_forecasts["dccemg"] = fc_dccemg
                model_metrics_fold["dccemg"] = mt_dccemg
                log_holdout_metrics(logger, series_name, fold_idx, "DCCEMG", mt_dccemg)
        else:
            logger.info(
                f"[{series_name}][Fold {fold_idx + 1}] Skipping DCCEMG (ENABLE_TARGET_DCCEMG=False)."
            )

        # ======================================================================
        # 4.6) If no model produced forecasts in this fold, skip it
        # ======================================================================
        if not model_holdout_forecasts:
            logger.warning(
                f"[{series_name}][Fold {fold_idx + 1}] No model forecasts produced for holdout. Skipping fold."
            )
            continue

        # ======================================================================
        # 4.7) Add DA (Directional Accuracy) and WBA (Within 10% Band Accuracy) 
        # to fold metrics for each model
        # ======================================================================
        last_train_value = float(df_train["target"].iloc[-1])
        
        for mname, fc in model_holdout_forecasts.items():
            da, wba = compute_directional_and_band_accuracy(
                actual=actual_holdout_price,
                forecast=fc,
                last_train_value=last_train_value,
                band_threshold=0.10,
            )
            model_metrics_fold[mname]["da"] = da
            model_metrics_fold[mname]["wba"] = wba

        # ======================================================================
        # 4.8) Collect per-fold metrics rows for CSV
        # ======================================================================
        for mname, mt in model_metrics_fold.items():
            per_fold_rows.append(
                {
                    "fold": fold_idx + 1,
                    "n_folds_total": n_folds,
                    "model_name": mname,
                    "train_start": df_train.index.min().date(),
                    "train_end": df_train.index.max().date(),
                    "holdout_start": df_holdout.index.min().date(),
                    "holdout_end": df_holdout.index.max().date(),
                    "mae": mt["mae"],
                    "rmse": mt["rmse"],
                    "mape": mt["mape"],
                    "da": mt["da"],
                    "wba": mt["wba"],
                }
            )

        # ======================================================================
        # 4.9) Aggregate metrics across folds in memory
        # ======================================================================
        for mname, mt in model_metrics_fold.items():
            if mname not in aggregated_metrics:
                aggregated_metrics[mname] = {
                    "mae": [],
                    "rmse": [],
                    "mape": [],
                    "da": [],
                    "wba": [],
                }
            aggregated_metrics[mname]["mae"].append(mt["mae"])
            aggregated_metrics[mname]["rmse"].append(mt["rmse"])
            aggregated_metrics[mname]["mape"].append(mt["mape"])
            aggregated_metrics[mname]["da"].append(mt["da"])
            aggregated_metrics[mname]["wba"].append(mt["wba"])

        # Save latest fold objects for holdout CSV/plotting
        if fold_idx == n_folds - 1:
            latest_df_train = df_train
            latest_df_holdout = df_holdout
            latest_indep_holdout_df = indep_holdout_df
            latest_model_holdout_forecasts = model_holdout_forecasts
            latest_actual_holdout_price = actual_holdout_price

    # ======================================================================
    # 5) After all folds: guard against no valid folds
    # ======================================================================
    if not aggregated_metrics or latest_model_holdout_forecasts is None:
        logger.warning(f"[{series_name}] No valid folds produced metrics. Skipping series.")
        return

    # ======================================================================
    # 5.1) Save per-fold metrics CSV
    # ======================================================================
    if per_fold_rows:
        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_file = os.path.join(model_sel_dir, PER_FOLD_HOLDOUT_METRICS_SUFFIX)
        per_fold_df.to_csv(per_fold_file, index=False)
        logger.info(f"[{series_name}] Per-fold holdout metrics saved to {per_fold_file}")
    else:
        logger.info(f"[{series_name}] No per-fold holdout metrics to save (no successful folds).")

    # ======================================================================
    # 5.2) Save independent variables holdout for latest fold
    # ======================================================================
    indep_holdout_save = latest_indep_holdout_df.copy()
    indep_holdout_save.insert(0, "Date", indep_holdout_save.index)
    indep_holdout_file = os.path.join(model_sel_dir, HOLDOUT_INDEP_SUFFIX)
    indep_holdout_save.to_csv(indep_holdout_file, index=False)
    logger.info(
        f"[{series_name}] Exogenous HOLDOUT forecasts (latest fold) saved to {indep_holdout_file}"
    )

    # ======================================================================
    # 5.3) Aggregate metrics per model across folds and compute FQS
    # ======================================================================
    aggregated_model_metrics = {}
    for mname, vals in aggregated_metrics.items():
        mae_arr = np.array(vals["mae"])
        rmse_arr = np.array(vals["rmse"])
        mape_arr = np.array(vals["mape"])
        da_arr = np.array(vals["da"])
        wba_arr = np.array(vals["wba"])

        mae_mean = float(mae_arr.mean())
        rmse_mean = float(rmse_arr.mean())
        mape_mean = float(mape_arr.mean())
        da_mean = float(da_arr.mean())
        wba_mean = float(wba_arr.mean())

        mape_score = 1.0 / (1.0 + mape_mean)
        fq_score = WEIGHT_DA * da_mean + WEIGHT_MAPESCORE * mape_score + WEIGHT_WBA * wba_mean

        aggregated_model_metrics[mname] = dict(
            mae=mae_mean,
            rmse=rmse_mean,
            mape=mape_mean,
            da=da_mean,
            wba=wba_mean,
            fq_score=fq_score,
            n_folds=len(mape_arr),
        )

    df_train_last = latest_df_train
    df_holdout_last = latest_df_holdout
    actual_holdout_price = latest_actual_holdout_price

    # ======================================================================
    # 5.4) Baseline naive models (latest fold)
    # ======================================================================
    naive_last = pd.Series(
        [df_train_last["target"].iloc[-1]] * len(df_holdout_last),
        index=df_holdout_last.index,
        name="Forecast_Naive_Last",
    )
    mape_naive_last = mean_absolute_percentage_error(actual_holdout_price, naive_last)

    # Naive seasonal baseline (lag by one seasonal cycle) on latest fold
    if len(df_series_final) >= len(df_holdout_last) + seasonal_period:
        holdout_p = df_holdout_last.index.to_period(freq)
        seasonal_p = holdout_p - seasonal_period
        seasonal_index = seasonal_p.to_timestamp()
        seasonal_index = seasonal_index.intersection(df_series_final.index)
        if len(seasonal_index) == len(df_holdout_last):
            naive_seasonal = df_series_final.loc[seasonal_index, "target"]
            naive_seasonal.index = df_holdout_last.index
            naive_seasonal.name = "Forecast_Naive_Seasonal"
            mape_naive_seasonal = mean_absolute_percentage_error(
                actual_holdout_price, naive_seasonal
            )
        else:
            naive_seasonal = None
            mape_naive_seasonal = None
    else:
        naive_seasonal = None
        mape_naive_seasonal = None

    # ======================================================================
    # 5.5) Top-K ensemble using FQS
    # ======================================================================
    sorted_models = sorted(
        aggregated_model_metrics.items(),
        key=lambda kv: kv[1]["fq_score"],
        reverse=True,
    )
    top_k = min(TOP_K_ENSEMBLE, len(sorted_models))
    top_models = sorted_models[:top_k]
    top_model_names = [mname for mname, _ in top_models]

    # Ensure at least one multivariate target model in the ensemble if available
    has_multi = any(mname in MULTIVARIATE_TARGET_MODELS for mname, _ in top_models)
    if not has_multi:
        multi_candidates = [item for item in sorted_models if item[0] in MULTIVARIATE_TARGET_MODELS]
        if multi_candidates:
            best_multi = multi_candidates[0]
            if len(top_models) < TOP_K_ENSEMBLE:
                top_models.append(best_multi)
            else:
                top_models[-1] = best_multi
            top_models = sorted(top_models, key=lambda kv: kv[1]["fq_score"], reverse=True)

    top_model_names = [mname for mname, _ in top_models]

    if aggregated_model_metrics:
        n_folds_total = next(iter(aggregated_model_metrics.values()))["n_folds"]
    else:
        n_folds_total = 0

    logger.info(
        f"[{series_name}] Top-{len(top_models)} models by FQS over {n_folds_total} fold(s) "
        f"(best first): {top_model_names}"
    )

    # Convert FQS into normalized weights
    fq_array = np.array([mt["fq_score"] for _, mt in top_models])
    eps = 1e-6
    fq_array = np.maximum(fq_array, eps)
    weights = fq_array / fq_array.sum()

    # Ensemble forecast on the last fold
    ensemble_matrix = np.vstack(
        [latest_model_holdout_forecasts[mn].values for mn in top_model_names]
    )
    ensemble_vals = np.average(ensemble_matrix, axis=0, weights=weights)

    holdout_forecast_ensemble = pd.Series(
        ensemble_vals,
        index=df_holdout_last.index,
        name="Forecast_Ensemble",
    )

    metrics_ens = evaluate_regression_forecast(
        actual_holdout_price, holdout_forecast_ensemble
    )
    mae_ens = metrics_ens["mae"]
    rmse_ens = metrics_ens["rmse"]
    mape_ens = metrics_ens["mape"]

    # ======================================================================
    # 5.6) Save selected top models + aggregated metrics + weights
    # ======================================================================
    selected_rows = []
    for (mname, mt), w in zip(top_models, weights):
        selected_rows.append(
            {
                "model_name": mname,
                "mae": mt["mae"],
                "rmse": mt["rmse"],
                "mape": mt["mape"],
                "da": mt["da"],
                "wba": mt["wba"],
                "fq_score": mt["fq_score"],
                "n_folds": mt["n_folds"],
                "weight": float(w),
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    selected_file = os.path.join(model_sel_dir, SELECTED_MODELS_SUFFIX)
    selected_df.to_csv(selected_file, index=False)
    logger.info(
        f"[{series_name}] Selected top-{len(top_models)} models (aggregated over folds) and "
        f"ensemble weights saved to {selected_file}"
    )

    # ======================================================================
    # 5.7) Metrics summary (segregated by multivariate vs univariate)
    # ======================================================================
    metrics_lines = [
        f"[{series_name}] Holdout Validation ({Model_Selection_Period} {freq}-periods per fold, {n_folds_total} fold(s))",
        "------------------------------------------------------------",
    ]

    multi_sorted = [(m, mt) for m, mt in sorted_models if m in MULTIVARIATE_TARGET_MODELS]
    uni_sorted = [(m, mt) for m, mt in sorted_models if m not in MULTIVARIATE_TARGET_MODELS]

    metrics_lines.append("Models using independent variables (multivariate):")
    if multi_sorted:
        for mname, mt in multi_sorted:
            metrics_lines.append(
                f"  [EXOG] {mname:>10} -> MEAN over {mt['n_folds']} fold(s): "
                f"MAE: {mt['mae']:.4f}, RMSE: {mt['rmse']:.4f}, "
                f"MAPE: {mt['mape']:.4f} ({mt['mape']*100:.2f}%), "
                f"DA: {mt['da']:.4f}, WBA: {mt['wba']:.4f}, "
                f"FQS: {mt['fq_score']:.4f}"
            )
    else:
        metrics_lines.append("  (none)")

    metrics_lines.append("")
    metrics_lines.append("Univariate models (target-only, no independent variables used):")
    if uni_sorted:
        for mname, mt in uni_sorted:
            metrics_lines.append(
                f"  [UNI ] {mname:>10} -> MEAN over {mt['n_folds']} fold(s): "
                f"MAE: {mt['mae']:.4f}, RMSE: {mt['rmse']:.4f}, "
                f"MAPE: {mt['mape']:.4f} ({mt['mape']*100:.2f}%), "
                f"DA: {mt['da']:.4f}, WBA: {mt['wba']:.4f}, "
                f"FQS: {mt['fq_score']:.4f}"
            )
    else:
        metrics_lines.append("  (none)")

    metrics_lines.append("")
    metrics_lines.append(
        f"Ensemble (top-{len(top_models)}: {', '.join(top_model_names)}) "
        f"-> Latest-fold MAE: {mae_ens:.4f}, RMSE: {rmse_ens:.4f}, "
        f"MAPE: {mape_ens:.4f} ({mape_ens*100:.2f}%)"
    )
    metrics_lines.append("")
    metrics_lines.append(
        f"Naive last-value (latest fold) MAPE : {mape_naive_last:.4f}  "
        f"(or {mape_naive_last*100:.2f}%)"
    )
    if mape_naive_seasonal is not None:
        metrics_lines.append(
            f"Naive seasonal (t-12, latest fold) MAPE : "
            f"{mape_naive_seasonal:.4f}  (or {mape_naive_seasonal*100:.2f}%)"
        )
    else:
        metrics_lines.append("Naive seasonal (t-12, latest fold) MAPE : N/A")
    metrics_lines.append("")
    metrics_lines.append(
        f"Holdout Mean Target (price, latest fold): {actual_holdout_price.mean():.4f}"
    )

    metrics_str = "\n".join(metrics_lines)
    logger.info(metrics_str)

    metrics_file = os.path.join(model_sel_dir, HOLDOUT_METRICS_SUFFIX)
    with open(metrics_file, "w") as f:
        f.write(metrics_str + "\n")
    logger.info(f"[{series_name}] Metrics saved to {metrics_file}")

    # ======================================================================
    # 5.8) Holdout results CSV (latest fold)
    # ======================================================================
    holdout_results = pd.DataFrame(
        {
            "Date": df_holdout_last.index,
            "Actual": actual_holdout_price.values,
        }
    )

    for mname, fc in latest_model_holdout_forecasts.items():
        holdout_results[f"Forecast_{mname}"] = fc.values

    holdout_results["Forecast_Ensemble"] = holdout_forecast_ensemble.values

    def pe_col(pred_col):
        return (
            (holdout_results["Actual"] - holdout_results[pred_col]).abs()
            / holdout_results["Actual"]
        ) * 100

    for mname in latest_model_holdout_forecasts.keys():
        holdout_results[f"PE_{mname}"] = pe_col(f"Forecast_{mname}")

    holdout_results["PE_Ensemble"] = pe_col("Forecast_Ensemble")

    holdout_csv = os.path.join(model_sel_dir, HOLDOUT_RESULTS_SUFFIX)
    holdout_results.to_csv(holdout_csv, index=False)
    logger.info(f"[{series_name}] Holdout results CSV (latest fold) saved to {holdout_csv}")

    # ======================================================================
    # 5.9) Holdout plot (latest fold) - Best model vs Ensemble
    # ======================================================================
    best_model_name = sorted_models[0][0]
    best_model_fc = latest_model_holdout_forecasts[best_model_name]

    second_best_model_name = sorted_models[1][0] if len(sorted_models) > 1 else None
    second_best_model_fc = (
        latest_model_holdout_forecasts[second_best_model_name]
        if second_best_model_name is not None
        else None
    )

    plt.figure(figsize=(14, 7))
    plt.plot(df_train_last["target"].iloc[-60:], label="Training (Recent)")
    plt.plot(
        actual_holdout_price.index,
        actual_holdout_price.values,
        label="Actual Holdout (latest fold)",
        marker="o",
    )
    plt.plot(
        best_model_fc.index,
        best_model_fc.values,
        label=f"Best Model ({best_model_name})",
        linestyle="--",
    )

    if second_best_model_fc is not None:
        plt.plot(
            second_best_model_fc.index,
            second_best_model_fc.values,
            label=f"Second Best Model ({second_best_model_name})",
            linestyle=":",
        )

    plt.plot(
        holdout_forecast_ensemble.index,
        holdout_forecast_ensemble.values,
        label=f"Ensemble (top-{len(top_models)})",
        linestyle="-",
    )

    plt.title(f"{series_name} - Holdout Forecast (Latest Fold): Best Model vs Ensemble")
    plt.ylabel(target_col_name)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plot_file = os.path.join(model_sel_dir, HOLDOUT_PLOT_SUFFIX)
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"[{series_name}] Holdout plot (latest fold) saved to {plot_file}")
