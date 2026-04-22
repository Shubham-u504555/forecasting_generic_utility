# =============================================================================
# step_1_feature_selection.py
#
# Purpose
# -------
# End-to-end variable selection pipeline for a time series.
# This script:
#   - Reads Config.csv to determine which input file and columns to use.
#   - Loads the input dataset (target + independent variables) from Excel.
#   - Applies trimming, validation, missing-value handling, correlation
#     filtering, VIF pruning, and multiple ranking methods.
#   - Produces a final list of selected variables plus diagnostic outputs.
#
# How It Relates to Input Data & Config.csv
# ------------------------------------------
# 1) Config.csv (in the input folder):
#      Mandatory columns used here:
#        - Input_File_Name
#        - Target_Column_Name
#        - Date_Column_Name
#        - Run_Feature_Selection  (Yes/No)
#        - Frequency              ("M", "W", "Y")
#        - Minimum_Observations   (positive integer; minimum number of periods)
#
#      Behaviour:
#        - The script searches Config.csv and extracts:
#              Input_File_Name      → e.g. "Master_Data_Input_File.xlsx"
#              Target_Column_Name   → e.g. "Target"
#              Date_Column_Name     → e.g. "Date"
#              Frequency            → e.g. "M" / "W" / "Y"
#              Minimum_Observations → e.g. 60
#
# 2) Master Excel file (Input_File_Name from Config.csv):
#      Expectations:
#        - File is located under the input folder passed via --input.
#        - Contains a sheet named "Data".
#        - Sheet "Data" must have at least:
#              Date_Column_Name    → time index (e.g. "Date")
#              Target_Column_Name  → dependent variable (e.g. "Target")
#              All remaining columns are treated as candidate independent
#              variables for feature selection.
#
#      Flow:
#        - Data is loaded and indexed by Date_Column_Name.
#        - History is trimmed to ensure enough observations and decent data coverage.
#        - Target is checked for strict periodic continuity.
#        - High-missing features are dropped; remaining gaps are interpolated.
#        - Correlation filter + VIF pruning + ranking are applied to select
#          the final set of variables.
#
# Outputs
# -------
# - <output_folder>/selected_features.csv
#       Final list of selected variables + scores.
#
# - Diagnostic files (MI, ElasticNet, SHAP scores, correlation rejections, etc.)
#       Useful for debugging and explaining why certain features were kept/dropped.
#
# - feature_selection_summary.json
#       High-level summary of thresholds, counts, and selected variable names.
# =============================================================================


import os
import sys
import time
import json
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Custom utilities
from utils.utils_common import (
    get_logger,
    cleanup_old_logs,
    fetch_series_config,
    load_time_series_data,
    trim_or_take_last_n_observations,
    validate_time_series_strict,
)

from utils.utils_feature_selection import (
    drop_high_missing,
    time_interpolate,
    corr_based_filter,
    vif_based_filter,
    mi_rank,
    elasticnet_rank,
    xgb_shap_rank,
    normalize_importance,
)

# Global settings (thresholds, file formats, etc.)
from settings import (
    LOG_RETENTION_MONTHS,
    ROW_COMPLETENESS_THRESHOLD,
    MIN_OBSERVATIONS,
    COLUMN_MISSING_THRESHOLD,
    CORR_MIN,
    VIF_MAX,
    COMPOSITE_MIN_THRESHOLD,
    MIN_SELECTED_VARIABLES,
    CSV_FLOAT_FMT,
    W_CORR,
    W_MI,
    W_ENET,
    W_SHAP,
)

# =====================================================================
# Core logic for variable selection
# =====================================================================

def run_variable_selection_pipeline(
    input_folder_name: str,
    series_config: dict,
    output_folder_name: str,
    logger,
    data_sheet_name: str = "Data",
) -> pd.DataFrame:
    """
    Execute the full variable selection workflow for a time series.

    Steps:
      1. Load & trim series.
      2. Enforce periodic continuity.
      3. Drop high-missing variables + interpolate.
      4. Correlation-based filtering.
      5. VIF-based multicollinearity pruning.
      6. Rank via Corr, MI, ElasticNet, XGB-SHAP.
      7. Select variables by composite score threshold (with Top-K fallback).
      8. Save all outputs for inspection.

    Parameters
    ----------
    input_folder_name : str
        Root input directory (contains input file and Config.csv).
    series_config : dict
        Configuration (from Config.csv).
    output_folder_name : str
        Output directory for variable selection results.
    logger :
        Logger instance for structured logging.

    Returns
    -------
    pd.DataFrame
        DataFrame of selected variables and associated metadata
        (correlation, VIF, scores, composite score, etc.).
    """

    target_col = series_config["Target_Column_Name"]
    freq = series_config["Frequency"]  # "M", "W", or "Y"

    # Minimum observations are interpreted as *periods* of the configured frequency.
    # (e.g., 60 for monthly means 60 months; 260 for weekly means 260 weeks.)
    min_obs = int(series_config.get("Minimum_Observations", MIN_OBSERVATIONS))

    # =================================================================
    # Step 1: Load time series data (Date index, target+all indep vars)
    # =================================================================
    df_series = load_time_series_data(input_folder_name, data_sheet_name, series_config)
    logger.info(
        "Done 1 - Initial data shape including target and all variables: %s",
        df_series.shape,
    )

    # =================================================================
    # Step 2: Trim rows to a period with decent coverage and sufficient
    #         history (>= Minimum_Observations).
    # =================================================================
    df_trimmed, strategy = trim_or_take_last_n_observations(
        df_series,
        target_col,
        freq=freq,
        threshold=ROW_COMPLETENESS_THRESHOLD,
        min_obs=min_obs,
    )
    logger.info(
        "Done 2 - Trimmed data shape (%s): %s",
        strategy,
        df_trimmed.shape,
    )

    # =================================================================
    # Step 3: Enforce strict periodic continuity on target series
    # =================================================================
    validate_time_series_strict(df_trimmed, target_col, freq=freq)
    logger.info(
        "Done 3 - %s-frequency continuity & target completeness check passed",
        freq,
    )

    # Split target and independent variables for convenience
    y = df_trimmed[target_col]
    X = df_trimmed.drop(columns=[target_col])

    # =================================================================
    # Step 4: Drop variables exceeding missing-value threshold
    # =================================================================
    X_dropped, dropped_cols, missing_map = drop_high_missing(
        X,
        COLUMN_MISSING_THRESHOLD,
    )
    logger.info(
        "Done 4 - Dropped %d columns due to missingness > %.2f",
        len(dropped_cols),
        COLUMN_MISSING_THRESHOLD,
    )

    # =================================================================
    # Step 5: Interpolate remaining gaps along time axis
    # =================================================================
    X_interp = time_interpolate(X_dropped)
    logger.info(
        "Done 5 - Number of independent variables after interpolation: %s",
        X_interp.shape[1],
    )

    # Combine target back for correlation computations
    df_clean = pd.concat([y, X_interp], axis=1)

    # =================================================================
    # Step 6: Simple correlation filter
    # =================================================================
    df_keep_corr, df_reject_corr = corr_based_filter(
        df_clean,
        target_col=target_col,
        threshold=CORR_MIN,
    )
    keep_corr_features = list(df_keep_corr["Variable_Name"])

    if len(keep_corr_features) == 0:
        raise ValueError(
            "No features passed correlation threshold. "
            f"Threshold: {CORR_MIN}"
        )

    X_corr = X_interp[keep_corr_features]
    logger.info(
        "Done 6 - Number of variables after correlation filter: %d",
        X_corr.shape[1],
    )

    # =================================================================
    # Step 7: Standardize variables before VIF so units don't bias VIF
    # =================================================================
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_corr),
        index=X_corr.index,
        columns=X_corr.columns,
    )
    logger.info("Done 7 - Standard scaling applied before VIF computation")

    # =================================================================
    # Step 8: VIF pruning to remove multicollinearity
    # =================================================================
    X_vif, vif_series = vif_based_filter(X_scaled, VIF_MAX)
    df_vif = vif_series.rename_axis("Variable_Name").reset_index(name="VIF_Value")
    logger.info(
        "Done 8 - Number of variables after VIF pruning: %d",
        X_vif.shape[1],
    )

    if X_vif.shape[1] == 0:
        raise ValueError(
            "All features removed by VIF. Please review VIF threshold."
        )

    # Remaining variables after VIF
    feature_names = list(X_vif.columns)

    # =================================================================
    # Step 9: Variable ranking using multiple methods
    # =================================================================
    logger.info("Done 9.1 - Computing MI, ElasticNet, XGB-SHAP rankings")

    y_for_rank = df_clean[target_col].loc[X_corr.index]
    X_for_rank = X_corr[feature_names]

    # Mutual Information (MI)
    # ElasticNet 
    # SHAP values from XGBoost
    mi_scores = mi_rank(X_for_rank, y_for_rank)
    enet_scores = elasticnet_rank(X_for_rank, y_for_rank)
    shap_scores = xgb_shap_rank(X_for_rank, y_for_rank)

    # Absolute correlation for selected variables
    corr_map = df_keep_corr.set_index("Variable_Name")["Correlation_Value"]
    corr_scores = corr_map.reindex(feature_names).abs()

    # Normalize all scores to [0, 1]
    corr_n = normalize_importance(corr_scores, feature_names)
    mi_n = normalize_importance(mi_scores, feature_names)
    enet_n = normalize_importance(enet_scores, feature_names)
    shap_n = normalize_importance(shap_scores, feature_names)

    composite_scores = (
        W_CORR * corr_n
        + W_MI * mi_n
        + W_ENET * enet_n
        + W_SHAP * shap_n
    ).sort_values(ascending=False)

    logger.info(
        "Done 9.2 - Composite score computed for %d features",
        len(composite_scores),
    )

    # =================================================================
    # Step 10: Select final variables based on composite score
    # =================================================================
    selected_by_threshold = composite_scores[composite_scores >= COMPOSITE_MIN_THRESHOLD]

    if len(selected_by_threshold) >= MIN_SELECTED_VARIABLES:
        final_features = list(selected_by_threshold.index)
        logger.info(
            "Done 10 - Selected %d features by composite threshold (>= %.3f)",
            len(final_features),
            COMPOSITE_MIN_THRESHOLD,
        )
    else:
        k = min(MIN_SELECTED_VARIABLES, len(composite_scores))
        final_features = list(composite_scores.head(k).index)
        logger.info(
            "Done 10 - Threshold yielded %d features; fallback to Top-%d by composite score",
            len(selected_by_threshold),
            k,
        )

    # =================================================================
    # Step 11: Build selected_variables DataFrame and enrich with metadata
    # =================================================================
    df_selected = pd.DataFrame(
        {
            "Variable_Name": final_features,
        }
    )

    # Add correlation, VIF and ranking metrics
    df_selected = pd.merge(
        df_selected,
        df_keep_corr[["Variable_Name", "Correlation_Value"]],
        how="left",
        on="Variable_Name",
    )

    df_selected = pd.merge(
        df_selected,
        df_vif[["Variable_Name", "VIF_Value"]],
        how="left",
        on="Variable_Name",
    )

    df_selected["Corr_Abs"] = corr_scores.reindex(final_features).values
    df_selected["MI_Score"] = mi_scores.reindex(final_features).values
    df_selected["ElasticNet_AbsCoef"] = enet_scores.reindex(final_features).values
    df_selected["XGB_SHAP_MeanAbs"] = shap_scores.reindex(final_features).values
    df_selected["Composite_Score"] = composite_scores.reindex(final_features).values

    os.makedirs(output_folder_name, exist_ok=True)

    # =================================================================
    # Step 12: Save artifacts for this series
    # =================================================================
    df_selected.to_csv(
        os.path.join(output_folder_name, "selected_features.csv"),
        index=False,
        encoding="utf-8",
        float_format=CSV_FLOAT_FMT,
    )

    mi_scores.to_csv(
        os.path.join(output_folder_name, "mi_feature_scores.csv"),
        header=["mi_score"],
        encoding="utf-8",
        float_format=CSV_FLOAT_FMT,
    )
    enet_scores.to_csv(
        os.path.join(output_folder_name, "elasticnet_feature_scores.csv"),
        header=["elasticnet_abs_coef"],
        encoding="utf-8",
        float_format=CSV_FLOAT_FMT,
    )
    shap_scores.to_csv(
        os.path.join(output_folder_name, "xgb_shap_feature_scores.csv"),
        header=["xgb_shap_mean_abs"],
        encoding="utf-8",
        float_format=CSV_FLOAT_FMT,
    )

    df_reject_corr.to_csv(
        os.path.join(output_folder_name, "rejected_features_correlation_based.csv"),
        index=False,
        encoding="utf-8",
        float_format=CSV_FLOAT_FMT,
    )

    summary = {
        "rows": int(X_vif.shape[0]),
        "cols": int(len(final_features)),
        "row_completeness_threshold": ROW_COMPLETENESS_THRESHOLD,
        "column_missing_threshold": COLUMN_MISSING_THRESHOLD,
        "corr_min_abs_threshold": CORR_MIN,
        "vif_max_threshold": VIF_MAX,
        "dropped_columns_due_to_missing_value_threshold": dropped_cols,
        "num_selected_features": int(len(final_features)),
        "selected_features": final_features,
        "composite_score_min_threshold": COMPOSITE_MIN_THRESHOLD,
    }

    with open(
        os.path.join(output_folder_name, "feature_selection_summary.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Done 13 - Variable selection complete. Selected %d feature(s): %s",
        len(final_features),
        final_features,
    )
    return df_selected

# =====================================================================
# Argument Parser & Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments for the variable selection script.
    """
    parser = argparse.ArgumentParser(
        description="Variable selection pipeline for a time series"
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Input folder containing master Excel file and Config.csv",
    )
    parser.add_argument(
        "--data-sheet",
        default="Data",
        help="Sheet name in input Excel file (default: 'Data')",
    )
    parser.add_argument(
        "--config-filename",
        default="Config.csv",
        help="Config CSV filename (inside input folder)",
    )
    parser.add_argument(
        "--output_root",
        default="output",
        help="Top-level output folder (will contain subfolders for each step)",
    )
    parser.add_argument(
        "--output",
        default="output_feature_selection",
        help="Output folder for selected variables and diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    """
    Orchestrate the variable selection process for a multivariate time series
    consisting of one target variable and multiple independent variables.

    Returns
    -------
    int
        Exit code: 0=success, 1=runtime failure, 2=setup error.
    """

    args = parse_args()
    
    # Ensure output root exists
    output_root = args.output_root
    logs_dir = os.path.join(output_root, "logs")
    output_feature_selection_dir = os.path.join(output_root, args.output)

    # Create required folders
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_feature_selection_dir, exist_ok=True)

    # Logger now writes under output_root/logs
    logger = get_logger("feature_selection", logs_dir)

    # Cleanup old logs
    cleanup_old_logs(
        logs_dir="logs",
        retention_months=LOG_RETENTION_MONTHS,
        logger=logger,
    )

    logger.info("===== Feature Selection Process Started =====")
    print("===== Feature Selection Process Started =====")
    logger.info(
        "Args: input=%s | data_sheet=%s | config=%s | output_root=%s | output=%s",
        args.input,
        args.data_sheet,
        args.config_filename,
        output_root,
        output_feature_selection_dir,
    )

    # Basic validations
    if not os.path.isdir(args.input):
        logger.error("Input folder missing: %s", args.input)
        return 2

    # Load series config from Config.csv
    try:
        series_config = fetch_series_config(
            args.input, args.config_filename
        )
    except Exception as e:
        logger.exception("Failed to read Config.csv for variable selection: %s", e)
        return 2

    start_time = time.time()
    try:
        df_selected = run_variable_selection_pipeline(
            input_folder_name=args.input,
            series_config=series_config,
            output_folder_name=output_feature_selection_dir,
            logger=logger,
            data_sheet_name=args.data_sheet,
        )
        logger.info(
            "Completed variable selection. Selected %d feature(s).",
            df_selected.shape[0],
        )
        success = True
    except Exception as e:
        logger.exception("Variable selection failed: %s", e)
        success = False

    elapsed = time.time() - start_time
    logger.info(
        "===== Feature Selection Completed in %.2f sec (success=%s) =====",
        elapsed,
        success,
    )
    print(
        f"===== Feature Selection Completed in {elapsed:.2f} sec "
        f"(success={success}) ====="
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
