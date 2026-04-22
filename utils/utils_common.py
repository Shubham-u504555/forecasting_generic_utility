# =============================================================================
# utils_common.py
#
# Purpose
# -------
# Generic, reusable utilities for:
#   - Logging (get_logger, cleanup_old_logs)
#   - String/column sanitization
#   - Loading input Excel files
#   - Trimming and validating time series
#
# How It Relates to Input Data & Config.csv
# ------------------------------------------
# 1) Input data:
#    - The input dataset is an Excel file (e.g. "Master_Data_Input_File.xlsx")
#      located in the input folder provided at runtime.
#    - It must contain a sheet named "Data" with at least:
#          Date_Column_Name      (from Config.csv)
#          Target_Column_Name    (from Config.csv)
#          All other columns     → treated as candidate independent variables.
#
# 2) Config.csv:
#    - load_time_series_data() uses Config.csv to figure out:
#          • which Excel file to read       → Input_File_Name
#          • which column is the date       → Date_Column_Name
#          • which column is the target     → Target_Column_Name
#    - It then:
#          • reads sheet "Data"
#          • converts Date_Column_Name to datetime
#          • sets it as the index
#          • sorts chronologically
#
# Key Functions Tied to Input Data
# ---------------------------------
# - load_excel_file(path, sheet):
#       Low-level loader for a specific sheet.
#
# - load_time_series_data(input_folder_name, series_config):
#       High-level loader for the master dataset for a single series,
#       driven entirely by Config.csv:
#           • input_folder_name / Input_File_Name
#           • Sheet "Data"
#           • Date_Column_Name as index
#           • Target_Column_Name + all other columns retained.
#
# - trim_or_take_last_n_observations() and validate_time_series_strict():
#       These work on the already-loaded DataFrame and enforce:
#           • minimum history length
#           • strict periodic continuity for the target.
# =============================================================================


import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

import logging
from logging.handlers import RotatingFileHandler

# =====================================================================
# Centralized Logging
# =====================================================================

class IconFormatter(logging.Formatter):
    ICONS = {
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }
    ICON_PREFIXES = tuple(ICONS.values())

    def format(self, record):
        icon = self.ICONS.get(record.levelno, "")
        msg = str(record.getMessage())

        # Add icon only if not already present
        if not msg.startswith(self.ICON_PREFIXES):
            record.msg = f"{icon} {record.msg}"

        return super().format(record)


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or return a module-specific logger configured to write to a rotating file.
    """

    # 1. Remove root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 2. Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # 3. Initialize or reuse logger
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # 4. Build a timestamped file path (e.g., feature_selection_2025-11-07_16-45-10.log)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # 5. Configure file handler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    file_formatter = IconFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger._configured = True
    logger.info("Logger initialized → %s", log_file_path)
    return logger


def cleanup_old_logs(
    logs_dir: str,
    retention_months: int = 12,
    logger: Optional[logging.Logger] = None,
):
    """
    Delete log files older than `retention_months` calendar months.

    Supported filename format:
        <name>_YYYY-MM-DD_HH-MM-SS.log
    """
    LOG_TIMESTAMP_PATTERN = re.compile(
        r".*_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.log$"
    )

    logs_path = Path(logs_dir)
    if not logs_path.exists() or not logs_path.is_dir():
        if logger:
            logger.info("Logs directory does not exist: %s", logs_dir)
        return

    cutoff_dt = datetime.now() - relativedelta(months=retention_months)
    deleted_files = 0

    for file_path in logs_path.iterdir():
        if not file_path.is_file() or file_path.suffix != ".log":
            continue

        try:
            file_dt = None
            match = LOG_TIMESTAMP_PATTERN.match(file_path.name)

            if match:
                date_part, time_part = match.groups()
                file_dt = datetime.strptime(
                    f"{date_part} {time_part.replace('-', ':')}",
                    "%Y-%m-%d %H:%M:%S",
                )
            else:
                file_dt = datetime.fromtimestamp(file_path.stat().st_mtime)

            if file_dt < cutoff_dt:
                file_path.unlink()
                deleted_files += 1
                if logger:
                    logger.info("Deleted old log file: %s", file_path.name)

        except Exception as e:
            if logger:
                logger.warning(
                    "Failed to delete log file %s: %s", file_path.name, e
                )

    if logger:
        logger.info(
            "Log cleanup completed. Deleted %d file(s) older than %d month(s).",
            deleted_files,
            retention_months,
        )


def delete_training_artifact_folders(logger: Optional[logging.Logger] = None):
    """
    Completely delete training artifact folders (with all contents)
    that exist in the same directory as this script.
    """

    folders_to_delete = [
        "catboost_info",
        "lightning_logs",
    ]

    for folder in folders_to_delete:

        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                if logger:
                    logger.info("Deleted training logs folder completely: %s", folder)
            except Exception as e:
                if logger:
                    logger.error(
                        "Failed to delete training logs folder %s: %s", folder, e
                    )
        else:
            if logger:
                logger.info(
                    "Training logs folder not found, skipping: %s", folder
                )

# =====================================================================
# Utilities for sanitizing column names, values, strings, file paths
# =====================================================================

def sanitize_column_names(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitize column names for modelling.

    Operations:
      - Replace spaces / special chars with underscores.
      - Collapse multiple underscores.
      - Strip leading/trailing underscores.
    """
    sanitized_col_names_dict: Dict[str, str] = {}
    for col in df.columns:
        sanitized_name = (
            col.replace(" ", "_")
            .replace(",", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
        )
        sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")
        sanitized_col_names_dict[col] = sanitized_name

    return df.rename(columns=sanitized_col_names_dict), sanitized_col_names_dict


def sanitize_column_values(
    df: pd.DataFrame,
    column: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitize all string values in a specified DataFrame column using rules
    similar to `sanitize_column_names`.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    df_out = df.copy()
    sanitized_col_values_dict: Dict[str, str] = {}

    original_values = df_out[column].tolist()
    new_values = []

    for val in original_values:
        if not isinstance(val, str):
            new_values.append(val)
            continue

        if val in sanitized_col_values_dict:
            new_values.append(sanitized_col_values_dict[val])
            continue

        sanitized = (
            val.replace(" ", "_")
            .replace(",", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
        )
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")

        sanitized_col_values_dict[val] = sanitized
        new_values.append(sanitized)

    df_out[column] = new_values
    return df_out, sanitized_col_values_dict


def sanitize_strings(
    values: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Sanitize a list of string values using the same rules as `sanitize_column_names`.
    """
    sanitized_dict: Dict[str, str] = {}
    sanitized_list: List[str] = []

    for v in values:
        original = "" if v is None else str(v)

        if original in sanitized_dict:
            sanitized = sanitized_dict[original]
        else:
            sanitized = (
                original.replace(" ", "_")
                .replace(",", "_")
                .replace("(", "_")
                .replace(")", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace("<", "_")
                .replace(">", "_")
            )
            sanitized = re.sub(r"_+", "_", sanitized).strip("_")
            sanitized_dict[original] = sanitized

        sanitized_list.append(sanitized)

    return sanitized_list, sanitized_dict


def sanitize_name_for_path(name: str) -> str:
    """
    Convert an arbitrary series / project name into a filesystem-safe slug.

    Steps:
      - Lowercase the name.
      - Replace spaces with underscores.
      - Replace any non-alphanumeric/underscore characters with "_".
      - Collapse repeated underscores and strip leading/trailing ones.
    """
    s = name.lower().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# =====================================================================
# Config helper
# =====================================================================

def fetch_series_config(
    input_folder_name: str,
    config_filename: str = "Config.csv",
) -> Dict[str, Any]:
    """
    Read Config.csv and extract the configuration for model selection.

    Expected columns (generic):
        - Input_File_Name
        - Target_Column_Name
        - Date_Column_Name
        - Frequency               (M/W/Y)
        - Minimum_Observations    (min observations required)
        - Model_Selection_Period  (holdout length, in periods)
        - Forecasting_Period      (forecasting length, in periods)
        - (Optional) Series_Name
    """
    path_config = os.path.join(input_folder_name, config_filename)
    if not os.path.exists(path_config):
        raise FileNotFoundError(f"Config file not found at: {path_config}")

    # Using utf-8-sig safely strips it from the first header.
    df_config = pd.read_csv(path_config, encoding="utf-8-sig")
    df_config.columns = df_config.columns.str.strip()

    if df_config.empty:
        raise ValueError(
            "No row found in Config.csv"
        )

    if len(df_config) > 1:
        raise ValueError(
            "This pipeline currently supports a single series per run. "
        )
    
    required_cols = [
        "Input_File_Name",
        "Target_Column_Name",
        "Date_Column_Name",
        "Frequency",
        "Minimum_Observations",
        "Model_Selection_Period",
        "Forecasting_Period"
    ]
    missing = [c for c in required_cols if c not in df_config.columns]
    if missing:
        raise ValueError(f"Config.csv is missing required columns: {missing}")

    row = df_config.iloc[0]

    freq = str(row["Frequency"]).strip().upper()
    if freq not in {"M", "W", "Y"}:
        raise ValueError(
            f"Invalid Frequency '{row['Frequency']}' in Config.csv. "
            "Allowed values: 'M' (monthly), 'W' (weekly), 'Y' (yearly)."
        )
    
    # Minimum observations (in *periods* of the configured frequency).
    try:
        min_obs = int(row["Minimum_Observations"])
    except Exception as e:
        raise ValueError(
            "Invalid 'Minimum_Observations' in Config.csv. "
        ) from e

    if min_obs <= 0:
        raise ValueError(
            "Invalid 'Minimum_Observations' in Config.csv. "
            "It must be a positive integer."
        )
    
    # Model selection period
    try:
        model_sel_period = int(row["Model_Selection_Period"])
    except Exception as e:
        raise ValueError(
            "Invalid 'Model_Selection_Period' in Config.csv. "
        ) from e

    if model_sel_period <= 0:
        raise ValueError(
            "Invalid 'Model_Selection_Period' in Config.csv. "
            "It must be a positive integer."
        )
    
    # Forecasting period
    try:
        forecast_period = int(row["Forecasting_Period"])
    except Exception as e:
        raise ValueError(
            "Invalid 'Forecasting_Period' in Config.csv. "
        ) from e

    if forecast_period <= 0:
        raise ValueError(
            "Invalid 'Forecasting_Period' in Config.csv. "
            "It must be a positive integer."
        )

    series_config: Dict[str, Any] = {
        "Series_Name": row.get("Series_Name", "Sample Series"),
        "Input_File_Name": row["Input_File_Name"],
        "Target_Column_Name": row["Target_Column_Name"],
        "Date_Column_Name": row["Date_Column_Name"],
        "Frequency": freq,
        "Minimum_Observations": min_obs,
        "Model_Selection_Period": model_sel_period,
        "Forecasting_Period": forecast_period
    }
    
    return series_config


# =====================================================================
# I/O + Data Utilities
# =====================================================================

def load_excel_file(
    path: str,
    sheet: str,
) -> pd.DataFrame:
    """
    Load a specific sheet from an Excel file and clean column names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at path: {path}")

    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except ValueError as e:
        raise ValueError(f"Sheet '{sheet}' not found in file: {path}") from e

    df.columns = df.columns.str.strip()
    return df


def load_time_series_data(
    input_folder_name: str,
    data_sheet_name: str,
    series_config: Dict[str, str],
) -> pd.DataFrame:
    """
    Load a single time-series dataset (target + independent variables)
    from a master Excel file specified in Config.csv.

    Expectations:
      - Excel file is in `input_folder_name` under `Input_File_Name`.
      - Sheet name is "Data".
      - Contains at least:
          • Date_Column_Name   → time column
          • Target_Column_Name → dependent variable

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by the date column (DatetimeIndex), sorted
        chronologically, with the target and all independent variables
        as columns.
    """
    input_file_name = series_config["Input_File_Name"]
    date_col = series_config["Date_Column_Name"]
    target_col = series_config["Target_Column_Name"]

    path = os.path.join(input_folder_name, input_file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required input file '{input_file_name}' does not exist in: {input_folder_name}"
        )

    df = load_excel_file(path, sheet=data_sheet_name)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in input file.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input file.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # If there are duplicate timestamps, keep the first occurrence
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    return df

# =====================================================================
# Time-series trimming and validation
# =====================================================================

def trim_or_take_last_n_observations(
    df: pd.DataFrame,
    target_col: str,
    freq: str,
    threshold: float = 0.75,
    min_obs: int = 60,
) -> Tuple[pd.DataFrame, str]:
    """
    Trim a time series based on coverage, or take the last N observations.

    freq:
      - "M" → monthly (periods = calendar months)
      - "W" → weekly  (periods = calendar weeks)
      - "Y" → yearly  (periods = calendar years)

    Logic:
      1) For each calendar YEAR, compute the fraction of independent variables
         (excluding the target column) that have at least one non-null value
         in that year.
      2) Find the earliest year where this fraction >= `threshold`.
      3) If trimming from Jan 1 (of that year) leaves at least `min_obs`
         distinct periods (as per `freq`), return trimmed data with
         strategy="trimmed".
      4) Otherwise, fallback to the last `min_obs` distinct periods
         with strategy="last_n".
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    freq = freq.upper()
    if freq not in {"M", "W", "Y"}:
        raise ValueError(f"Unsupported freq '{freq}'. Use 'M', 'W', or 'Y'.")

    df = df.sort_index()

    independent_cols = [col for col in df.columns if col != target_col]
    if not independent_cols:
        raise ValueError("No independent variables left after excluding target column.")

    # Coverage by calendar year (same logic for any freq)
    cutoff_year = None
    for year, g in df.groupby(df.index.year):
        has_any = g[independent_cols].notna().any(axis=0)
        if has_any.mean() >= threshold:
            cutoff_year = int(year)
            break

    if cutoff_year is not None:
        start_ts = pd.Timestamp(year=cutoff_year, month=1, day=1)
        trimmed = df.loc[df.index >= start_ts]
    else:
        trimmed = df.copy()

    # Count periods according to freq
    period_index_trimmed = trimmed.index.to_period(freq)
    periods_remaining = period_index_trimmed.nunique()

    if periods_remaining >= min_obs:
        final_df, strategy = trimmed, "trimmed"
    else:
        # fallback: last N periods according to freq
        period_index_full = df.index.to_period(freq)
        unique_periods = period_index_full.unique()
        if len(unique_periods) < min_obs:
            raise ValueError(
                f"Not enough data: only {len(unique_periods)} {freq}-periods available, "
                f"but at least {min_obs} required."
            )
        last_periods = unique_periods[-min_obs:]
        mask = period_index_full.isin(last_periods)
        final_df = df.loc[mask]
        strategy = "last_n"

    # Final safety check
    final_period_index = final_df.index.to_period(freq)
    final_periods = final_period_index.nunique()
    if final_periods < min_obs:
        raise ValueError(
            f"Not enough data: only {final_periods} {freq}-periods available "
            f"after trimming; at least {min_obs} required."
        )

    return final_df, strategy


def validate_time_series_strict(
    df: pd.DataFrame,
    target_col: str,
    freq: str,
) -> None:
    """
    Strictly validate a time series before model training.

    Frequency is explicitly config-driven:
      - freq = "M" → monthly (one obs per calendar month)
      - freq = "W" → weekly  (one obs per calendar week)
      - freq = "Y" → yearly  (one obs per calendar year)

    Checks:
      1) Index must be DatetimeIndex.
      2) When converted to PeriodIndex with given `freq`, there must be:
         - no duplicate periods
         - no missing periods between min and max
      3) Target column must have no missing values.

    No frequency inference / guessing is performed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    freq = freq.upper()
    if freq not in {"M", "W", "Y"}:
        raise ValueError(f"Unsupported freq '{freq}'. Use 'M', 'W', or 'Y'.")

    df = df.sort_index()

    # Convert to period index at the desired granularity
    periods = df.index.to_period(freq)

    # 1) Check duplicates at the period level
    if periods.duplicated().any():
        dup_periods = periods[periods.duplicated()].unique()
        raise ValueError(
            f"Duplicate {freq}-periods found in index: "
            f"{[str(p) for p in dup_periods]}"
        )

    # 2) Check continuity between min and max period
    expected_periods = pd.period_range(
        start=periods.min(),
        end=periods.max(),
        freq=freq,
    )
    missing_periods = expected_periods.difference(periods.unique())

    if len(missing_periods) > 0:
        raise ValueError(
            f"Missing {len(missing_periods)} {freq}-period(s): "
            f"{[str(p) for p in missing_periods]}"
        )

    # 3) Target must be complete
    missing_target = df.index[df[target_col].isna()].strftime("%Y-%m-%d").tolist()
    if len(missing_target) > 0:
        raise ValueError(
            f"Target '{target_col}' missing on {len(missing_target)} date(s): "
            f"{missing_target}"
        )

    return

