#!/usr/bin/env bash
# Reproduce the whole Command Center end-to-end.
set -e
cd "$(dirname "$0")"

# avoid BLAS/OpenMP thread oversubscription (can hang predict.py)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 0. get the data if missing (Kaggle dataset, mirrored on GitHub)
if [ ! -f data/results.csv ]; then
  mkdir -p data
  curl -sL -o data/results.csv \
    https://raw.githubusercontent.com/martj42/international_results/master/results.csv
fi

python src/etl.py            # collection -> cleaning -> SQLite -> KPI marts
python src/train.py          # train + evaluate the ML model (writes model.joblib)
python src/predict.py        # match predictions + upset risk
python src/sentiment.py      # NLP sentiment by team
python src/build_dashboard.py # assemble the interactive dashboard
echo
echo "Done -> open outputs/command_center.html"
