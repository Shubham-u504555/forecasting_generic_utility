# =============================================================================
# settings.py
#
# Purpose
# -------
# Central place for hyperparameters, thresholds, and constants used across the
# generic time-series forecasting pipeline (variable selection, model selection,
# forecasting, etc.).
#
# How It Relates to Master Data & Config.csv
# ------------------------------------------
# - This file does NOT read data itself. Instead, it defines generic settings
#   which are applied to whatever master dataset and configuration are supplied
#   at runtime.
# - The actual dataset and column names are specified in:
#       Config.csv
#       ├─ Input_File_Name      → e.g. "Master_Data_Input_File.xlsx"
#       ├─ Date_Column_Name     → e.g. "Date"
#       └─ Target_Column_Name   → e.g. "Target"
# - The master Excel file is expected to live under the input folder specified
#   when running step_1_feature_selection.py, and to have at least:
#       • Sheet "Data"
#       • Date_Column_Name      (time index)
#       • Target_Column_Name    (dependent variable)
#       • All remaining columns are treated as candidate independent variables.
#
# Scope & Design
# --------------
# - All settings here are domain-neutral.
# - This makes the same script usable for any time series:
#       • commodity prices
#       • sales volumes
#       • demand forecasts
#   as long as the input master file and Config.csv follow the expected format.
# =============================================================================

# =====================================================================
# GLOBAL RANDOMNESS / PARALLELISM
# =====================================================================

# Random seed for reproducibility across all models and selections.
RANDOM_STATE = 42

# Number of parallel jobs to use where supported (e.g. ElasticNetCV, XGBoost).
# -1 means "use all available cores".
N_JOBS = -1

# Output Formatting
CSV_FLOAT_FMT = "%.3f"

# =====================================================================
# LOG RETENTION MONTHS FOR ALL STEPS
# =====================================================================
LOG_RETENTION_MONTHS = 12

# =====================================================================
# MODEL ENABLE FLAGS
# =====================================================================

ENABLE_TWFE = True
ENABLE_SYSGMM = True
ENABLE_MG = True
ENABLE_AMG = True
ENABLE_CCEMG = True
ENABLE_DCCEMG = True
ENABLE_LASSO = True
ENABLE_RF = True
ENABLE_ANN = True
ENABLE_XGBOOST = True
ENABLE_SVR = True

# Canonical model names used across logs, selections, and outputs.
MODEL_NAME_MAP = {
    "twfe": "Two-way Fixed Effect (TWFE)",
    "sysgmm": "System Generalized Moments Estimator (sysGMM)",
    "mg": "Mean Group (MG)",
    "amg": "Augmented Mean Group (AMG)",
    "ccemg": "Common Correlated Effects Mean Group (CCEMG)",
    "dccemg": "Dynamic CCEMG (DCCEMG)",
    "lasso": "Least Absolute Shrinkage and Selection Operator (LASSO)",
    "rf": "Random Forest (RF)",
    "ann": "Artificial Neural Network (ANN)",
    "xgboost": "eXtreme Gradient Boosting (XGBoost)",
    "svr": "Support Vector Regression (SVR)",
}

# =====================================================================
# THRESHOLDS / CONTROLS FOR VARIABLE SELECTION PROCESS
# =====================================================================

# Minimum fraction of independent variables that must have at least one
# non-null value within a given calendar year. Used to decide the earliest
# year from which the dataset has acceptable coverage.
ROW_COMPLETENESS_THRESHOLD = 0.75    # 75%

# Minimum number of observations required for a series to be usable.
# Interpreted in the configured frequency (e.g., monthly/weekly/yearly periods).
MIN_OBSERVATIONS = 60

# Maximum allowable fraction of missing values in a column before it is dropped.
COLUMN_MISSING_THRESHOLD = 0.25      # 25%

# Minimum absolute correlation with the target to keep a variable in the
# domain-aware correlation filter.
CORR_MIN = 0.2

# Maximum allowed Variance Inflation Factor (VIF) after multicollinearity pruning.
VIF_MAX = 10.0

# =====================================================================
# COMPOSITE FEATURE RANKING CONFIGURATION
# =====================================================================
# These settings control how independent variables are ranked and filtered
# during the feature-selection phase of the pipeline.
#
# Each variable receives importance scores from four ranking algorithms:
#   • Correlation      → absolute correlation with target
#   • Mutual Information (MI)
#   • ElasticNet       → absolute regularized coefficients
#   • SHAP             → mean absolute SHAP values from XGBoost
#
# Before combining:
#   - All importance vectors are aligned to the same feature set.
#   - All scores are normalized to the range [0, 1].
#
# The final composite score is computed as:
#
#   CompositeScore =
#         (W_CORR * Corr_norm)
#       + (W_MI   * MI_norm)
#       + (W_ENET * ENET_norm)
#       + (W_SHAP * SHAP_norm)
#
# The weights below determine the contribution of each ranking method.
# Although weights typically sum to 1.0, the system is robust if they do not.
W_CORR = 0.20     # Weight for absolute correlation with target
W_MI = 0.20       # Weight for Mutual Information
W_ENET = 0.30     # Weight for ElasticNet coefficients
W_SHAP = 0.30     # Weight for SHAP values from XGBoost

# Minimum composite score threshold used to select meaningful variables.
# A variable is selected if:
#       CompositeScore ≥ COMPOSITE_MIN_THRESHOLD
COMPOSITE_MIN_THRESHOLD = 0.25

# Fallback minimum number of selected variables; if the threshold would
# select fewer, we take the top-K variables instead.
MIN_SELECTED_VARIABLES = 5        # ensure at least this many if possible

# =====================================================================
# LAG CONFIGURATION
# =====================================================================

# Maximum number of autoregressive (AR) lags of the target variable.
# This helps tree models learn temporal structure in the target series.
N_LAGS_TARGET = 12

# =====================================================================
# TARGET MODEL ENABLE FLAGS
# =====================================================================
# Toggle each *target (dependent variable)* model family on/off globally.
#
# If a flag is False, that model will be:
#   • skipped during target model selection (Step-2), and
#   • can also be skipped in future-forecasting logic (Step-3) where used.
#
# This lets you quickly run ablation studies or create "light" vs "full"
# experiment profiles without touching the core modelling code.

# Multivariate models (use independent variables / drivers)
ENABLE_TARGET_LGBM       = True
ENABLE_TARGET_ELASTICNET = True
ENABLE_TARGET_CATBOOST   = True
ENABLE_TARGET_XGBOOST    = True
ENABLE_TARGET_SARIMAX    = True

# Univariate classical time-series models (target-only)
ENABLE_TARGET_HOLTWINTERS = True
ENABLE_TARGET_THETA       = True

# Univariate neural models (target-only, via NeuralForecast)
ENABLE_TARGET_NHITS   = True
ENABLE_TARGET_NBEATS  = True
ENABLE_TARGET_TFT     = True
ENABLE_TARGET_LSTM    = True   # Long Short-Term Memory (via NeuralForecast)

# Univariate statistical — Facebook Prophet
ENABLE_TARGET_PROPHET = True   # Prophet (additive/multiplicative decomposition)

# Econometric panel models adapted for single time series
ENABLE_TARGET_TWFE    = True   # Two-way Fixed Effect (OLS with year/month dummies)
ENABLE_TARGET_SYSGMM  = True   # System GMM via 2SLS with lagged instruments
ENABLE_TARGET_MG      = True   # Mean Group (rolling-window OLS average)
ENABLE_TARGET_AMG     = True   # Augmented Mean Group (MG + common PCA factor)
ENABLE_TARGET_CCEMG   = True   # Common Correlated Effects Mean Group
ENABLE_TARGET_DCCEMG  = True   # Dynamic CCEMG (CCEMG + lagged CCE term)

# Additional ML models
ENABLE_TARGET_LASSO   = True   # LASSO regression (L1 penalty via LassoCV)
ENABLE_TARGET_RF      = True   # Random Forest regressor
ENABLE_TARGET_ANN     = True   # Artificial Neural Network (MLPRegressor)
ENABLE_TARGET_SVR     = True   # Support Vector Regression (RBF kernel)

# =====================================================================
# TARGET MODELS ENSEMBLE CONFIG
# =====================================================================

# FQS (Forecast Quality Score) WEIGHTS
# These weights determine how DA, MAPEScore, and WBA contribute to the
# final Forecast Quality Score used to rank target models.
#
# Components:
#   • DA         → Directional Accuracy (0–1)
#   • MAPEScore  → 1/(1+MAPE), gives higher score for lower MAPE
#   • WBA        → Within-Band Accuracy (percentage of points within
#                  a specified relative-error threshold)
# All components are in [0, 1].
WEIGHT_DA = 0.25
WEIGHT_MAPESCORE = 0.25
WEIGHT_WBA = 0.50

# Final ensemble = weighted average of top-K selected models.
TOP_K_ENSEMBLE = 2

# =====================================================================
# FILE NAMING CONVENTIONS
# =====================================================================
# Centralizing these avoids typos in multiple scripts.
# All modules import settings.py for consistent naming of:
#   • holdout results
#   • future forecasts
#   • anomaly reports
#   • merged results
#
# Suffix-style naming helps maintain a predictable directory structure.
HOLDOUT_INDEP_SUFFIX = "Indep_Holdout_Forecast.csv"                   # holdout forecasts for independent variable models
HOLDOUT_RESULTS_SUFFIX = "Dep_Holdout_Forecast.csv"                   # per-model target predictions on holdout set
PER_FOLD_HOLDOUT_METRICS_SUFFIX = "Dep_Per_Fold_Holdout_Metrics.csv"  # fold-wise metrics for each target model
HOLDOUT_METRICS_SUFFIX = "Dep_Holdout_Metrics.txt"                    # aggregated validation metrics
HOLDOUT_PLOT_SUFFIX = "Dep_Holdout_Plot.png"
SELECTED_MODELS_SUFFIX = "Dep_Selected_Models.csv"                    # final selected target models + ensemble weights
INDEP_ENSEMBLE_WEIGHTS_SUFFIX = "Indep_Ensemble_Weights.csv"          # ensemble weights for independent variable model ensembles

HIST_INDEP_NUM_MONTHS = 24                                            # Number of historical months to keep in independent variables output file for analysis
HIST_FUTURE_INDEP_SUFFIX = "Indep_Historical_Future_Both.csv"         # complete historical + future predictions for independent variables
FUTURE_INDEP_SUFFIX = "Indep_Future_Forecast.csv"                     # complete future predictions for independent variables
FUTURE_FORECAST_SUFFIX = "Dep_Future_Forecast.xlsx"                   # final future forecast for the target series
FUTURE_METRICS_SUFFIX = "Indep_Future_Metrics.txt"                    # optional diagnostics for future forecasting
INDEP_ANOMALY_SUFFIX = "Indep_Future_Anomaly_Report.csv"              # anomaly detection output for future independent variables

MERGED_HIST_FUTURE_INDEP_SUFFIX = "Merged_Indep_Historical_Future_Both.xlsx" # Merged historical + future predictions for independent variables
MERGED_FUTURE_INDEP_SUFFIX = "Merged_Indep_Future_Forecast.xlsx"             # Merged future predictions for independent variables
MERGED_FUTURE_FORECAST_SUFFIX = "Merged_Dep_Future_Forecast.xlsx"  # merged final future forecast for the target series