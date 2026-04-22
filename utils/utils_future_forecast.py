# utils_future_forecast.py
# =====================================================================
# Future forecast utilities used by Step 3.
#
# Key behaviour:
#   - Build and return TWO independent-variable forecast dataframes:
#       1) indep_future_df
#            -> exactly Forecasting_Period rows (future periods only)
#       2) indep_historical_future_df
#            -> Historical_Window rows + Forecasting_Period rows
#
# Rules (as per requirement):
#   - Historical window MUST always end at the latest dependent-date period
#   - Future periods MUST always start right after the latest dependent date
#   - If independent variables have actual values beyond latest dependent date,
#     use them first; forecast only remaining periods (if any).
#
# The implementation is:
#   - series-agnostic
#   - frequency-aware (Frequency ∈ {M, W, Y})
# =====================================================================

import os
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from neuralforecast.losses.pytorch import MAPE
from neuralforecast.models import NHITS, NBEATS, TFT, LSTM

# Custom utilities
from utils.utils_common import (
    sanitize_column_names
)

from settings import (
    FUTURE_METRICS_SUFFIX,
    FUTURE_INDEP_SUFFIX,
    FUTURE_FORECAST_SUFFIX,
    INDEP_ANOMALY_SUFFIX,
    N_LAGS_TARGET,
    SELECTED_MODELS_SUFFIX,
    INDEP_ENSEMBLE_WEIGHTS_SUFFIX,
    RANDOM_STATE,
    HIST_FUTURE_INDEP_SUFFIX,
)

from utils.utils_dep_model_helpers import (
    make_target_features,
    build_target_feature_row,
    build_elasticnet_model,
    build_lgbm_model,
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
    fit_holtwinters,
    fit_theta,
    fit_prophet,
    neuralforecast_univariate,
    get_neural_input_size_and_downsample,
)

from utils.utils_indep_model_helpers import (
    get_indep_future_forecast,
    summarize_indep_future_anomalies,
    indep_risk_label,
    ts_freq_from_code,
)

warnings.filterwarnings("ignore")


# =====================================================================
# Core future-forecast routine
# =====================================================================

def execute_future_forecast(
    df_series: pd.DataFrame,
    input_folder_name: str,
    series_name: str,
    series_config: Dict[str, Any],
    variables_selected_folder_name: str,
    models_selected_folder_name: str,
    target_folder_name: str,
    logger,
) -> None:
    
    """
    Run future forecasting for a time series.

    Steps
    -----
    1) Load selected models + weights from Step 2 output
    2) Load independent variable future forecasts
       (ETS / ARIMA / Theta ensemble from utils_indep_model_helpers)
    3) Sanitize column names, build log-target, enforce configured
       time frequency (M/W/Y)
    4) Retrain selected target models on FULL history and forecast future
    5) Build ensemble forecast using saved FQS-based weights
    6) Summarize independent variable future anomalies / risk
    7) Save outputs:
         - Future independent variables forecast CSV
         - Historical + future independent variables CSV
         - Future target forecasts Excel (per model + ensemble)
         - Independent variable anomaly CSV + risk score
         - Future target forecast plot (Best, Second Best, Ensemble)
    """

    # =====================================================================
    # Config / metadata
    # =====================================================================
    target_col_name = series_config["Target_Column_Name"]
    freq = str(series_config.get("Frequency", "M")).upper()
    ts_freq, seasonal_period  = ts_freq_from_code(freq)
    forecast_period = int(series_config["Forecasting_Period"])

    # Model-selection outputs live under a (per-series) folder
    model_sel_dir = models_selected_folder_name
    if not os.path.exists(model_sel_dir):
        logger.warning(
            "[%s] Model-selection folder missing: %s",
            series_name,
            model_sel_dir,
        )
        return
    
    # Future-forecast output directory
    os.makedirs(target_folder_name, exist_ok=True)
    logger.info(
        "Future forecast output directory: %s",
        target_folder_name,
    )

    # ==================================================================
    # 1) Get independent variables future forecasts
    # ==================================================================
    indep_future_df, indep_historical_future_df = get_indep_future_forecast(
        input_folder_name=input_folder_name,
        series_name=series_name,
        series_config=series_config,
        variables_selected_folder_name=variables_selected_folder_name,
        model_sel_dir=model_sel_dir,
        df_series=df_series,
        logger=logger,
    )

    # Save future independent variables values
    indep_future_save = indep_future_df.copy()
    indep_future_save.insert(0, "Date", indep_future_save.index)
    indep_future_file = os.path.join(target_folder_name, FUTURE_INDEP_SUFFIX)
    indep_future_save.to_csv(indep_future_file, index=False)
    logger.info(
        "Independent FUTURE forecasts saved to %s",
        indep_future_file,
    )

    future_dates = indep_future_df.index

    # Save historical + future independent variables values
    indep_hist_future_save = indep_historical_future_df.copy()
    indep_hist_future_save.insert(0, "Date", indep_hist_future_save.index)
    indep_hist_future_file = os.path.join(target_folder_name, HIST_FUTURE_INDEP_SUFFIX)
    indep_hist_future_save.to_csv(indep_hist_future_file, index=False)
    logger.info(
        "Independent HISTORICAL + FUTURE window saved to %s",
        indep_hist_future_file
    )

    # ==================================================================
    # 2) Load target models selected + weights (from Step 2)
    # ==================================================================
    selected_file = os.path.join(model_sel_dir, SELECTED_MODELS_SUFFIX)
    if not os.path.exists(selected_file):
        logger.warning(
            "[%s] Selected models file not found: %s. Skipping series.",
            series_name,
            selected_file,
        )
        return

    selected_df = pd.read_csv(selected_file)
    if selected_df.empty:
        logger.warning(
            "[%s] Selected models file is empty: %s. Skipping series.",
            series_name,
            selected_file,
        )
        return

    model_names: List[str] = selected_df["model_name"].tolist()
    weights = selected_df["weight"].values

    logger.info(
        "[%s] Using top target models for future forecasting: %s",
        series_name,
        model_names,
    )

    # ==================================================================
    # 3) Sanitize columns, log-target, and frequency
    # ==================================================================
    if not isinstance(df_series.index, pd.DatetimeIndex):
        raise TypeError("df_series index must be a DatetimeIndex.")
    
    # Sanitize column names (idempotent if already sanitized)
    df_series, col_map = sanitize_column_names(df_series.copy())

    if target_col_name not in col_map:
        logger.error(
            "[%s] Target column '%s' not found. Available (original): %s",
            series_name,
            target_col_name,
            list(col_map.keys()),
        )
        return

    sanitized_target = col_map[target_col_name]
    df_series = df_series.rename(columns={sanitized_target: "target"})

    # Work in log space internally
    df_series["target_log"] = np.log(df_series["target"])

    # Sort index and rely on continuity checks already performed in Step 3
    df_series_final = df_series.sort_index()

    # All non-target columns treated as independent drivers,
    # but only keep those for which we have future forecasts
    indep_cols = [
        c
        for c in df_series_final.columns
        if c not in ["target", "target_log"] and c in indep_future_df.columns
    ]

    # Ensure indep_future_df has the same column order as in df_ts_final
    indep_future_df = indep_future_df[indep_cols]

    full_start_date = df_series_final.index.min()
    full_end_date = df_series_final.index.max()

    logger.info(
        "[%s] Date range: %s to %s",
        series_name,
        full_start_date.date(),
        full_end_date.date(),
    )
    logger.info("[%s] Independent driver columns: %s", series_name, indep_cols)

    # ==================================================================
    # 4) FULL retrain target models (log-target) and get future forecasts
    # ==================================================================
    logger.info(
        "[%s] Retraining selected target models on FULL data & forecasting "
        "next %d periods (Frequency=%s)...",
        series_name,
        forecast_period,
        freq,
    )

    # Build feature matrix on full history (same as Step 2)
    X_all, y_all_log = make_target_features(
        df_local=df_series_final,
        target_col="target_log",
        indep_columns=indep_cols,
    )

    feature_cols_all = list(X_all.columns)

    future_model_forecasts = {}

    def _recursive_future_multivariate(
        model,
        model_key: str,
    ) -> None:
        """
        Internal utility for recursive multi-step forecasting in log space.

        For each future date in `future_dates`, it:
          - builds a feature row using the last N_LAGS_TARGET log-target values
            and the independent drivers at that date
          - predicts the next log-target
          - appends prediction to history buffer for the next step
        """

        # Initial history buffer: last N_LAGS_TARGET observed log-target values
        target_history_full_log = (
            df_series_final["target_log"]
            .loc[:full_end_date]
            .tail(N_LAGS_TARGET)
            .tolist()
        )

        if len(target_history_full_log) < N_LAGS_TARGET:
            logger.warning(
                "[%s] Not enough log-target history for %s full retrain. "
                "Skipping %s future forecast.",
                series_name,
                model_key,
                model_key,
            )
            return

        preds_log: List[float] = []

        for dt in future_dates:
            indep_current = {col: indep_future_df.loc[dt, col] for col in indep_cols}

            x_row = build_target_feature_row(
                target_history_log=target_history_full_log,
                indep_current=indep_current,
                current_date=dt,
                feature_columns=feature_cols_all,
                n_lags_target=N_LAGS_TARGET,
            )

            y_hat_log = float(model.predict(x_row)[0])
            preds_log.append(y_hat_log)

            # Recursive update of history buffer
            target_history_full_log.append(y_hat_log)
            if len(target_history_full_log) > N_LAGS_TARGET:
                target_history_full_log.pop(0)

        future_series = pd.Series(
            np.exp(np.array(preds_log)),
            index=future_dates,
            name=f"Forecast_{model_key}",
        )
        future_model_forecasts[model_key] = future_series

    # 4.1) LGBM
    if "lgbm" in model_names:
        try:
            logger.info(
                "[%s] Training LGBM model on full data (log-target)...",
                series_name,
            )
            lgbm_full = build_lgbm_model()
            lgbm_full.fit(X_all, y_all_log, eval_metric="mape")
            _recursive_future_multivariate(lgbm_full, "lgbm")
        except Exception as e:
            logger.warning(
                "[%s] LGBM full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.2) ElasticNet
    if "elasticnet" in model_names:
        try:
            logger.info(
                "[%s] Training ElasticNet model on full data (log-target)...",
                series_name,
            )
            enet_full = build_elasticnet_model()
            enet_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(enet_full, "elasticnet")
        except Exception as e:
            logger.warning(
                "[%s] ElasticNet full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.3) CatBoost
    if "catboost" in model_names:
        try:
            logger.info(
                "[%s] Training CatBoost model on full data (log-target)...",
                series_name,
            )
            cb_full = build_catboost_model()
            cb_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(cb_full, "catboost")
        except Exception as e:
            logger.warning(
                "[%s] CatBoost full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.4) XGBoost
    if "xgboost" in model_names:
        try:
            logger.info(
                "[%s] Training XGBoost model on full data (log-target)...",
                series_name,
            )
            xgb_full = build_xgboost_model()
            xgb_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(xgb_full, "xgboost")
        except Exception as e:
            logger.warning(
                "[%s] XGBoost full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.5) SARIMAX
    if "sarimax" in model_names:
        try:
            logger.info(
                "[%s] Training SARIMAX model on full data (log-target)...",
                series_name,
            )

            sarimax_model = build_sarimax_model(
                df_train=df_series_final,
                target_col="target_log",
                indep_cols=indep_cols,
                seasonal_period=seasonal_period,
            )
            
            sarimax_full = sarimax_model.fit(disp=False)
            sarimax_future_log = sarimax_full.get_forecast(
                steps=forecast_period,
                exog=indep_future_df,
            ).predicted_mean

            future_sarimax = pd.Series(
                np.exp(sarimax_future_log.values),
                index=future_dates,
                name="Forecast_sarimax",
            )
            future_model_forecasts["sarimax"] = future_sarimax
        except Exception as e:
            logger.warning(
                "[%s] SARIMAX full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.6) Holt-Winters (univariate)
    if "holtwinters" in model_names:
        try:
            logger.info(
                "[%s] Training Holt-Winters model on full data (target_log)...",
                series_name,
            )
            y_train_hw = df_series_final["target_log"]
            df_future_index = pd.DataFrame(index=future_dates)

            hw_log = fit_holtwinters(
                train_series_log=y_train_hw,
                df_future=df_future_index,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )

            future_hw = pd.Series(
                np.exp(hw_log.values),
                index=future_dates,
                name="Forecast_holtwinters",
            )
            future_model_forecasts["holtwinters"] = future_hw
        except Exception as e:
            logger.warning(
                "[%s] Holt-Winters full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.7) Theta (univariate)
    if "theta" in model_names:
        try:
            logger.info(
                "[%s] Training Theta model on full data (target_log)...",
                series_name,
            )
            y_train_theta = df_series_final["target_log"]
            df_future_index = pd.DataFrame(index=future_dates)

            theta_log = fit_theta(
                train_series_log=y_train_theta,
                df_future=df_future_index,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )

            future_theta = pd.Series(
                np.exp(theta_log.values),
                index=future_dates,
                name="Forecast_theta",
            )
            future_model_forecasts["theta"] = future_theta
        except Exception as e:
            logger.warning(
                "[%s] Theta full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.8) NHITS
    input_size_neural, freq_downsample = get_neural_input_size_and_downsample(
        freq=freq,
        train_len=len(df_series_final),
        h=forecast_period,
    )

    if "nhits" in model_names:
        try:
            logger.info(
                "[%s] Training NHITS model on full data (log-target)...",
                series_name,
            )

            nhits_full = NHITS(
                h=forecast_period,
                input_size=input_size_neural,
                max_steps=300,
                n_freq_downsample=freq_downsample,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_nhits_full = neuralforecast_univariate(
                df_local=df_series_final,
                target_col="target_log",
                horizon=forecast_period,
                model=nhits_full,
                ts_freq=ts_freq,      # <- fix
            ).reindex(future_dates)

            future_nhits = pd.Series(
                np.exp(fc_nhits_full.values),
                index=future_dates,
                name="Forecast_nhits",
            )
            future_model_forecasts["nhits"] = future_nhits
        except Exception as e:
            logger.warning(
                "[%s] NHITS full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.9) NBEATS
    if "nbeats" in model_names:
        try:
            logger.info(
                "[%s] Training NBEATS model on full data (log-target)...",
                series_name,
            )
            nbeats_full = NBEATS(
                h=forecast_period,
                input_size=input_size_neural,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )
            fc_nbeats_full = neuralforecast_univariate(
                df_local=df_series_final,
                target_col="target_log",
                horizon=forecast_period,
                model=nbeats_full,
                ts_freq=ts_freq,      # <- fix
            ).reindex(future_dates)

            future_nbeats = pd.Series(
                np.exp(fc_nbeats_full.values),
                index=future_dates,
                name="Forecast_nbeats",
            )
            future_model_forecasts["nbeats"] = future_nbeats
        except Exception as e:
            logger.warning(
                "[%s] NBEATS full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.10) TFT
    if "tft" in model_names:
        try:
            logger.info(
                "[%s] Training TFT model on full data (log-target)...",
                series_name,
            )
            tft_full = TFT(
                h=forecast_period,
                input_size=input_size_neural,
                hidden_size=32,
                dropout=0.2,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )
            fc_tft_full = neuralforecast_univariate(
                df_local=df_series_final,
                target_col="target_log",
                horizon=forecast_period,
                model=tft_full,
                ts_freq=ts_freq,      # <- fix
            ).reindex(future_dates)

            future_tft = pd.Series(
                np.exp(fc_tft_full.values),
                index=future_dates,
                name="Forecast_tft",
            )
            future_model_forecasts["tft"] = future_tft
        except Exception as e:
            logger.warning(
                "[%s] TFT full-data forecast failed: %s",
                series_name,
                e,
            )

    # 4.10b) LSTM
    if "lstm" in model_names:
        try:
            logger.info("[%s] Training LSTM model on full data (log-target)...", series_name)

            lstm_full = LSTM(
                h=forecast_period,
                input_size=input_size_neural,
                encoder_n_layers=2,
                encoder_hidden_size=64,
                max_steps=300,
                random_seed=RANDOM_STATE,
                loss=MAPE(),
            )

            fc_lstm_full = neuralforecast_univariate(
                df_local=df_series_final,
                target_col="target_log",
                horizon=forecast_period,
                model=lstm_full,
                ts_freq=ts_freq,
            ).reindex(future_dates)

            future_lstm = pd.Series(
                np.exp(fc_lstm_full.values),
                index=future_dates,
                name="Forecast_lstm",
            )
            future_model_forecasts["lstm"] = future_lstm
        except Exception as e:
            logger.warning("[%s] LSTM full-data forecast failed: %s", series_name, e)

    # 4.10c) Prophet
    if "prophet" in model_names:
        try:
            logger.info("[%s] Training Prophet model on full data (target_log)...", series_name)

            y_train_prophet = df_series_final["target_log"]
            df_future_index = pd.DataFrame(index=future_dates)

            prophet_log = fit_prophet(
                train_series_log=y_train_prophet,
                df_future=df_future_index,
                ts_freq=ts_freq,
                seasonal_period=seasonal_period,
            )

            future_prophet = pd.Series(
                np.exp(prophet_log.values),
                index=future_dates,
                name="Forecast_prophet",
            )
            future_model_forecasts["prophet"] = future_prophet
        except Exception as e:
            logger.warning("[%s] Prophet full-data forecast failed: %s", series_name, e)

    # 4.11) LASSO
    if "lasso" in model_names:
        try:
            logger.info("[%s] Training LASSO model on full data (log-target)...", series_name)
            lasso_full = build_lasso_model()
            lasso_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(lasso_full, "lasso")
        except Exception as e:
            logger.warning("[%s] LASSO full-data forecast failed: %s", series_name, e)

    # 4.12) Random Forest
    if "rf" in model_names:
        try:
            logger.info("[%s] Training RF model on full data (log-target)...", series_name)
            rf_full = build_rf_model()
            rf_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(rf_full, "rf")
        except Exception as e:
            logger.warning("[%s] RF full-data forecast failed: %s", series_name, e)

    # 4.13) ANN
    if "ann" in model_names:
        try:
            logger.info("[%s] Training ANN model on full data (log-target)...", series_name)
            ann_full = build_ann_model()
            ann_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(ann_full, "ann")
        except Exception as e:
            logger.warning("[%s] ANN full-data forecast failed: %s", series_name, e)

    # 4.14) SVR
    if "svr" in model_names:
        try:
            logger.info("[%s] Training SVR model on full data (log-target)...", series_name)
            svr_full = build_svr_model()
            svr_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(svr_full, "svr")
        except Exception as e:
            logger.warning("[%s] SVR full-data forecast failed: %s", series_name, e)

    # 4.15) TWFE
    if "twfe" in model_names:
        try:
            logger.info("[%s] Training TWFE model on full data (log-target)...", series_name)
            twfe_full = TWFEEstimator()
            twfe_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(twfe_full, "twfe")
        except Exception as e:
            logger.warning("[%s] TWFE full-data forecast failed: %s", series_name, e)

    # 4.16) sysGMM
    if "sysgmm" in model_names:
        try:
            logger.info("[%s] Training sysGMM (2SLS) model on full data (log-target)...", series_name)
            sysgmm_full = SysGMMEstimator()
            sysgmm_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(sysgmm_full, "sysgmm")
        except Exception as e:
            logger.warning("[%s] sysGMM full-data forecast failed: %s", series_name, e)

    # 4.17) MG
    if "mg" in model_names:
        try:
            logger.info("[%s] Training MG (Mean Group) model on full data (log-target)...", series_name)
            mg_full = MGEstimator()
            mg_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(mg_full, "mg")
        except Exception as e:
            logger.warning("[%s] MG full-data forecast failed: %s", series_name, e)

    # 4.18) AMG
    if "amg" in model_names:
        try:
            logger.info("[%s] Training AMG (Augmented Mean Group) model on full data (log-target)...", series_name)
            amg_full = AMGEstimator()
            amg_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(amg_full, "amg")
        except Exception as e:
            logger.warning("[%s] AMG full-data forecast failed: %s", series_name, e)

    # 4.19) CCEMG
    if "ccemg" in model_names:
        try:
            logger.info("[%s] Training CCEMG model on full data (log-target)...", series_name)
            ccemg_full = CCEMGEstimator()
            ccemg_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(ccemg_full, "ccemg")
        except Exception as e:
            logger.warning("[%s] CCEMG full-data forecast failed: %s", series_name, e)

    # 4.20) DCCEMG
    if "dccemg" in model_names:
        try:
            logger.info("[%s] Training DCCEMG model on full data (log-target)...", series_name)
            dccemg_full = DCCEMGEstimator()
            dccemg_full.fit(X_all, y_all_log)
            _recursive_future_multivariate(dccemg_full, "dccemg")
        except Exception as e:
            logger.warning("[%s] DCCEMG full-data forecast failed: %s", series_name, e)

    # ==================================================================
    # 5) Ensemble over future forecasts using saved FQS weights
    # ==================================================================
    if not future_model_forecasts:
        logger.warning(
            "[%s] No future forecasts produced by selected models. Skipping series.",
            series_name,
        )
        return

    valid_model_names = [mn for mn in model_names if mn in future_model_forecasts]
    if not valid_model_names:
        logger.warning(
            "[%s] None of the selected models produced future forecasts.",
            series_name,
        )
        return

    weights_map = dict(zip(model_names, weights))
    weights_vec = np.array([weights_map[mn] for mn in valid_model_names], dtype=float)

    if weights_vec.sum() <= 0:
        weights_vec = np.ones_like(weights_vec) / len(weights_vec)
    else:
        weights_vec = weights_vec / weights_vec.sum()

    ensemble_matrix_future = np.vstack(
        [future_model_forecasts[mn].values for mn in valid_model_names]
    )

    ensemble_vals_future = np.average(
        ensemble_matrix_future,
        axis=0,
        weights=weights_vec,
    )

    future_forecast_ensemble = pd.Series(
        ensemble_vals_future,
        index=future_dates,
        name="Forecast_Ensemble",
    )

    # ==================================================================
    # 6) Save future forecasts (Excel with forecast + model_selected)
    # ==================================================================
    future_df = pd.DataFrame({"date": future_dates})
    future_df["series_name"] = series_name
    future_df["month_id"] = list(range(1, forecast_period + 1))
    future_df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, mn in enumerate(valid_model_names, start=1):
        future_df[f"forecast_top_{i}"] = future_model_forecasts[mn].values

    future_df["forecast_ensemble"] = future_forecast_ensemble.values

    future_df.set_index("date", inplace=True)

    model_selected_df = pd.DataFrame(
        {"model_rank": range(1, len(valid_model_names) + 1)}
    )
    model_selected_df["model_name"] = valid_model_names
    model_selected_df["series_name"] = series_name

    future_excel = os.path.join(target_folder_name, FUTURE_FORECAST_SUFFIX)
    with pd.ExcelWriter(future_excel, engine="openpyxl") as writer:
        future_df.to_excel(writer, sheet_name="forecast", index=True)
        model_selected_df.to_excel(writer, sheet_name="model_selected", index=False)

    logger.info(
        "[%s] Future %d-period forecast Excel saved to %s",
        series_name,
        forecast_period,
        future_excel,
    )

    # ==================================================================
    # 7) Independent-variable future anomalies & risk scoring
    # ==================================================================
    try:
        # Historical independent-variable data from df_ts_final
        indep_hist_df = df_series_final[indep_cols]

        anomaly_df = summarize_indep_future_anomalies(
            df_hist=indep_hist_df,
            df_future=indep_future_df,
            z_threshold=3.0,
            range_margin=0.20,
            flat_std_ratio=0.05,
        )

        metrics_file = os.path.join(target_folder_name, FUTURE_METRICS_SUFFIX)

        if not anomaly_df.empty:
            anomaly_file = os.path.join(target_folder_name, INDEP_ANOMALY_SUFFIX)
            anomaly_df.to_csv(anomaly_file, index=False)
            logger.info(
                "[%s] Independent-variable future anomaly report saved to %s",
                series_name,
                anomaly_file,
            )

            series_risk_score = float(anomaly_df["risk_score"].mean())
            series_risk_label = indep_risk_label(series_risk_score)

            logger.info(
                "[%s] Independent-variable future risk score (0-100): %.2f (%s)",
                series_name,
                series_risk_score,
                series_risk_label,
            )

            # Try appending risk summary to metrics file
            try:
                with open(metrics_file, "a") as f:
                    f.write("\n")
                    f.write("Independent-variable FUTURE risk assessment\n")
                    f.write("--------------------------------------------\n")
                    f.write(
                        "Series-level independent-variable future forecast "
                        f"risk score (0-100): {series_risk_score:.2f} "
                        f"({series_risk_label})\n"
                    )
            except Exception as e_metrics:
                logger.warning(
                    "[%s] Failed to append independent-variable future forecast "
                    "risk score to metrics file: %s",
                    series_name,
                    e_metrics,
                )

            flagged = anomaly_df[
                anomaly_df[["flag_out_of_range", "flag_large_jump", "flag_flat"]].any(
                    axis=1
                )
            ]

            if not flagged.empty:
                flagged_vars = ", ".join(flagged["variable"].astype(str).tolist())
                logger.warning(
                    "[%s] Independent-variable FUTURE anomalies detected for: %s",
                    series_name,
                    flagged_vars,
                )
        else:
            logger.info(
                "[%s] No independent-variable FUTURE anomaly summary could be computed "
                "(empty anomaly_df).",
                series_name,
            )

    except Exception as e:
        logger.warning(
            "[%s] Failed to compute independent-variable FUTURE anomaly report: %s",
            series_name,
            e,
        )

    # ==================================================================
    # 8) Future plot: Historical + Best Model + Second Best + Ensemble
    # ==================================================================
    best_model_future_name = valid_model_names[0]
    best_future_fc = future_model_forecasts[best_model_future_name]

    second_best_future_name = (
        valid_model_names[1] if len(valid_model_names) > 1 else None
    )

    second_best_future_fc = (
        future_model_forecasts[second_best_future_name]
        if second_best_future_name is not None
        else None
    )

    plt.figure(figsize=(14, 7))
    plt.plot(df_series_final["target"], label="Historical")
    plt.plot(
        best_future_fc.index,
        best_future_fc.values,
        label=f"Future Best Model ({best_model_future_name})",
        linestyle="--",
    )
    if second_best_future_fc is not None:
        plt.plot(
            second_best_future_fc.index,
            second_best_future_fc.values,
            label=f"Future Second Best ({second_best_future_name})",
            linestyle=":",
        )

    plt.plot(
        future_forecast_ensemble.index,
        future_forecast_ensemble.values,
        label=f"Future Ensemble (top-{len(valid_model_names)})",
        linestyle="-",
    )

    plt.title(
        f"{series_name} - Next {forecast_period} periods forecast "
        f"(Best, Second Best & Ensemble)"
    )
    plt.ylabel(target_col_name)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    future_plot = os.path.join(
        target_folder_name,
        f"future_plot_next{forecast_period}.png",
    )
    plt.savefig(future_plot)
    plt.close()

    logger.info(
        "[%s] Future forecast plot saved to %s",
        series_name,
        future_plot,
    )