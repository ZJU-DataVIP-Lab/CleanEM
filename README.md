# CleanEM

**CleanEM** is a calibrated evidence modeling framework for auditable tabular error detection, built on the principle that *the LLM proposes, but never decides*. This is the repository for the paper "CleanEM: Auditable Tabular Error Detection via Calibrated Evidence Modeling".

### Initialization

To set up the environment and get started with CleanEM, follow these steps:

```bash
# Clone the CleanEM repository
git clone https://github.com/missanchor/CleanEM.git
cd CleanEM

# Create and activate the environment (Python 3.10)
conda env create -f environment.yml
conda activate cleanem
```

### Dataset

All six benchmark datasets used in this work (`hospital`, `flights`, `beers`, `billionaire`, `movies`, and `rayyan`) can be found in the `data/` folder.

### Configuration

- **LLM API**: Configure via `--base_url` (default: `http://127.0.0.1:8000/v1`), `--model`, and the `OPENAI_API_KEY` environment variable.
- **Default parameters**: `--active_label_budget 10`, `--adaptive_mechanism_query_fraction 0.40`, `--score_threshold 0.5`.

### Usage

```bash
python main.py \
  --dirty_csv data/beers_error-01.csv \
  --clean_csv data/beers_clean.csv \
  --output_dir results/clean_em_beers \
  --base_url http://127.0.0.1:8000/v1 \
  --model <model_name>
```

Unit tests are located in `test/` and can be run with `pytest`.
