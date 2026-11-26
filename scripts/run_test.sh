#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python main.py --mode clean_em --dirty_csv data/hospital_error-01.csv --clean_csv data/hospital_clean.csv --output_dir results/clean_em_hospital
python main.py --mode clean_em --dirty_csv data/flights_error-01.csv --clean_csv data/flights_clean.csv --output_dir results/clean_em_flights
python main.py --mode clean_em --dirty_csv data/beers_error-01.csv --clean_csv data/beers_clean.csv --output_dir results/clean_em_beers
