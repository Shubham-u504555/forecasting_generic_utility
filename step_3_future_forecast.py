# step_3_future_forecast.py
# =====================================================================
# Future forecasting for a SINGLE generic time series (target variable).
#
# This script orchestrates the future forecast pipeline:
#
#   1) Read Config.csv and pick the row with Run_Forecasting = YES
#   2) Load master dataset (target + independent variables) from Excel
#   3) Trim history based on coverage & minimum required observations
#      (frequency-aware, using Frequency from Config.csv)
#   4) Validate time continuity & target completeness
#   5) Keep only variables selected in Step 1 (selected_features.csv)
#   6) Time-interpolate missing values in selected independent variables
#   7) Call future-forecast routine which:
#        - Uses Step 2 model-selection outputs (selected models + weights)
#        - Retrains independent variable models on FULL history and
#          forecasts future (via utils_indep_model_helpers)
#        - Retrains selected target models and generates future forecasts
#        - Builds a weighted ensemble forecast
#        - Computes independent variable future anomalies & risk scores
#
# Output:
#   - Future independent variables forecast CSV
#   - Historical + future independent variables CSV
#   - Future target forecasts Excel (per model + ensemble)
#   - Independent variable future anomalies + risk score
#   - Future target forecast plot
# =====================================================================

import os
import sys
import time
import argparse
from typing import Dict, Any

import pandas as pd

# Custom utilities
from utils.utils_common import (
    get_logger,
    cleanup_old_logs,
    fetch_series_config,
    load_time_series_data,
    trim_or_take_last_n_observations,
    validate_time_series_strict,
    sanitize_column_values,
    delete_training_artifact_folders,
)

from utils.utils_model_selection import fetch_selected_variables
from utils.utils_feature_selection import time_interpolate
from utils.utils_future_forecast import execute_future_forecast

# Global settings (thresholds, file formats, etc.)
from settings import (
    LOG_RETENTION_MONTHS,
    ROW_COMPLETENESS_THRESHOLD,
    MIN_OBSERVATIONS,
)


# =====================================================================
# Core logic for future forecast for a single series
# =====================================================================
def run_future_forecast_pipeline(
    input_folder_name: str,
    series_config: Dict[str, Any],
    variables_selected_folder_name: str,
    models_selected_folder_name: str,
    target_folder_name: str,
    logger,
    data_sheet_name: str = "Data",
) -> None:
    """
    Execute the end-to-end future forecasting time series
    based on input data + Config + Step-1 selected variables +
    Step-2 models selected.

    Parameters
    ----------
    input_folder_name : str
        Folder containing Config.csv and the master Excel file.
    series_config : dict
        Configuration (from Config.csv).
    variables_selected_folder_name : str
        Folder containing Step-1 selected_features.csv.
    models_selected_folder_name : str
        Folder containing Step-2 model selection outputs
    target_folder_name : str
        Output folder for future-forecast artifacts.
    logger : logging.Logger
        Logger instance.
    data_sheet_name : str, optional
        Sheet name in the master Excel file. Defaults to "Data".
    """

    series_name = series_config.get("Series_Name", "Sample Series")
    input_filename = series_config["Input_File_Name"]
    target_col_name = series_config["Target_Column_Name"]
    date_col_name = series_config["Date_Column_Name"]
    freq = series_config.get("Frequency", "M").upper()
    min_obs = int(series_config.get("Minimum_Observations", MIN_OBSERVATIONS))
    forecast_period = int(series_config["Forecasting_Period"])

    logger.info("=" * 80)
    logger.info("Running Future Forecast for series: %s", series_name)
    logger.info(
        "Config → File=%s | Target=%s | Date=%s | Forecasting_Months=%d | Frequency=%s",
        input_filename,
        target_col_name,
        date_col_name,
        forecast_period,
        freq,
    )

    # =================================================================
    # Step 1: Load master dataset from Excel
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
        target_col_name,
        threshold=ROW_COMPLETENESS_THRESHOLD,
        min_obs=min_obs,
        freq=freq,
    )
    logger.info(
        "Done 2 - Trimmed data shape (%s): %s",
        strategy,
        df_trimmed.shape,
    )

    # =================================================================
    # Step 3: Enforce strict periodic continuity on target series
    # =================================================================
    validate_time_series_strict(df_trimmed, target_col_name, freq=freq)
    logger.info(
        "Done 3 - %s-frequency continuity & target completeness check passed",
        freq,
    )

    # Split target and features
    y = df_trimmed[target_col_name]
    X = df_trimmed.drop(columns=[target_col_name])

    # =================================================================
    # Step 4: Keep only the variables shortlisted in Step-1
    # =================================================================
    X_selected = fetch_selected_variables(
        X,
        variables_selected_folder_name,
    )

    logger.info("Done 4 - Number of Independent Variables Selected from Step 1 Output : %s", X_selected.shape[1])
    
    # =====================================================================
    # Step 5: Interpolate missing values in independent variables
    # =====================================================================
    X_interp = time_interpolate(X_selected)
    logger.info("Done 5 - Number of Independent Variables After Interpolation : %s", X_interp.shape[1])

    # Final cleaned dataset = target + selected + interpolated features
    df_clean = pd.concat([y, X_interp], axis=1)

    # =====================================================================
    # Step 6: Call future forecast routine
    # =====================================================================
    execute_future_forecast(
        df_series=df_clean,
        input_folder_name=input_folder_name,
        series_name=series_name,
        series_config=series_config,
        variables_selected_folder_name=variables_selected_folder_name,
        models_selected_folder_name=models_selected_folder_name,
        target_folder_name=target_folder_name,
        logger=logger,
    )


# =====================================================================
# Argument Parser & Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    """
    Define and parse CLI arguments for the future forecasting script.
    """
    parser = argparse.ArgumentParser(
        description="Step 3 - Future forecasting for target variable (single series)"
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Input folder containing Config.csv and master Excel file",
    )
    parser.add_argument(
        "--data-sheet",
        default="Data",
        help="Sheet name in master Excel file (default: 'Data')",
    )
    parser.add_argument(
        "--config-filename",
        default="Config.csv",
        help="Config CSV filename (inside input folder)",
    )
    parser.add_argument(
        "--vars",
        default="output_feature_selection",
        help="Folder with selected features (from Step 1)",
    )
    parser.add_argument(
        "--models",
        default="output_model_selection",
        help="Folder with models selected (from Step 2)",
    )
    parser.add_argument(
        "--output_root",
        default="output",
        help="Top-level output folder (will contain logs and step outputs)",
    )
    parser.add_argument(
        "--output",
        default="output_future_forecast",
        help="Output folder for future forecast artifacts",
    )

    return parser.parse_args()


# =====================================================================
# Main entry point
# =====================================================================
def main() -> int:
    """
    Orchestrate the future forecast process for a multivariate time series
    consisting of one target variable and multiple independent variables.
    """

    args = parse_args()

    # Ensure output root exists
    output_root = args.output_root
    logs_dir = os.path.join(output_root, "logs")
    vars_dir = os.path.join(output_root, args.vars)            # Step-1 outputs
    models_dir = os.path.join(output_root, args.models)        # Step-2 outputs
    output_future_dir = os.path.join(output_root, args.output) # Step-3 outputs

    # Create required folders
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_future_dir, exist_ok=True)

    # Logger under output_root/logs
    logger = get_logger("future_forecast", logs_dir)

    # Cleanup old logs
    cleanup_old_logs(
        logs_dir="logs",
        retention_months=LOG_RETENTION_MONTHS,
        logger=logger,
    )

    logger.info("==== Future Forecast Process Started ====")
    print("==== Future Forecast Process Started ====")
    logger.info(
        "Args: input=%s | data_sheet=%s | config=%s | vars=%s | models=%s | output_root=%s | output=%s ",
        args.input,
        args.data_sheet,
        args.config_filename,
        vars_dir,
        models_dir,
        output_root,
        output_future_dir,
    )

    # Basic validations
    if not os.path.isdir(args.input):
        logger.error("Input folder missing: %s", args.input)
        return 2

    if not os.path.isdir(vars_dir):
        logger.error("Selected variables folder missing: %s", vars_dir)
        return 2

    if not os.path.isdir(models_dir):
        logger.error("Model-selection folder missing: %s", models_dir)
        return 2

    # Load series configuration from Config.csv
    try:
        series_config = fetch_series_config(
            args.input, args.config_filename
        )
    except Exception as e:
        logger.exception("Failed to read Config CSV for future forecasting: %s", e)
        return 2

    # Run the single-series future forecast
    start = time.time()
    try:
        run_future_forecast_pipeline(
            input_folder_name=args.input,
            series_config=series_config,
            variables_selected_folder_name=vars_dir,
            models_selected_folder_name=models_dir,
            target_folder_name=output_future_dir,
            logger=logger,
            data_sheet_name=args.data_sheet,
        )
        success = True
    except Exception as e:
        logger.exception("Future forecasting failed for series: %s", e)
        success = False

    # Delete training logs created (if any)
    delete_training_artifact_folders(logger=logger)

    elapsed = time.time() - start
    logger.info(
        "===== Future Forecast Completed in %.2f sec (success=%d) =====",
        elapsed,
        success,
    )
    print(
        f"===== Future Forecast Completed in {elapsed:.2f} sec "
        f"(success={success}) ====="
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
