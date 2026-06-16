#!/usr/bin/env bash
# Pull the latest match results and rebuild the dashboard.
# The model is fixed (trained pre-tournament), so this skips retraining.
set -e
cd "$(dirname "$0")"

# avoid BLAS/OpenMP thread oversubscription (can hang predict.py)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Downloading latest results..."
curl -sL -o data/results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
python3 src/etl.py
python3 src/predict.py
python3 src/sentiment.py
python3 src/build_dashboard.py
echo
echo "Refreshed -> open outputs/command_center.html"
