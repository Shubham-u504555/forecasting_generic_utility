#!/usr/bin/env bash
set -e  # stop if any script fails

# Activate your virtual environment (adjust path if needed)
source "$(dirname "$0")/venv_forecasting/bin/activate"

python step_1_feature_selection.py
python step_2_model_selection.py
python step_3_future_forecast.py
