# CleanEM

**CleanEM** is an agentic error detection framework for tabular data. It combines table profiling, LLM-generated validation rules and canonicalization plans, adaptive error-mechanism discovery, and active-learning calibration to produce cell-level error probabilities, explanation traces, and repair candidates.

## Setup

```bash
# Clone the repository
git clone https://github.com/missanchor/CleanEM.git
cd CleanEM

# Create and activate the conda environment (Python 3.10)
conda env create -f environment.yml
conda activate cleaningllm
```

## Datasets

All benchmark datasets are located in the `data/` directory, each containing a dirty table (`*_error-01.csv`) and its corresponding ground-truth clean table (`*_clean.csv`):

- `beers`
- `billionaire`
- `flights`
- `hospital`
- `movies`
- `rayyan`

## Configuration

CleanEM connects to any OpenAI-compatible LLM endpoint (such as local vLLM, Ollama, or remote APIs):

- `--base_url`: OpenAI-compatible API base URL (default: `http://127.0.0.1:8000/v1`)
- `--model`: Model name (auto-detected from `/v1/models` if omitted)
- `OPENAI_API_KEY`: Environment variable for API authentication (default: `EMPTY` for local endpoints)

Key default parameters:

- `--mode`: `clean_em`
- `--mechanism_mode`: `adaptive_agent`
- `--active_label_budget`: `10`
- `--adaptive_mechanism_query_fraction`: `0.40`
- `--active_query_clean_proxy_fraction`: `0.30`
- `--score_threshold`: `0.5`

## Usage

Run CleanEM on a single dataset:

```bash
python main.py \
  --dirty_csv data/beers_error-01.csv \
  --clean_csv data/beers_clean.csv \
  --output_dir results/clean_em_beers \
  --base_url http://127.0.0.1:8000/v1
```

Or run the sample batch script:

```bash
bash scripts/run_test.sh
```

Outputs (cell error scores, explanation traces, repair candidates, and LLM token usage logs) are saved under `--output_dir`. For mechanism and ablation options, see [MECHANISM_MODES.md](MECHANISM_MODES.md) or run `python main.py --help`.

## Project Structure

- `main.py`: CLI entry point for running CleanEM
- `profiler.py`, `agent.py`, `cleanem_models.py`, `cleanem_inference.py`, `adaptive_mechanism_discovery.py`, `canonicalization.py`, `judge.py`: Core detection, discovery, and inference modules
- `core/`: Shared utilities (pattern exploration, safe evaluation, LLM usage logging)
- `data/`: Benchmark datasets
- `scripts/`: Batch runner and offline evaluation scripts
- `test/`: Unit test suite

## Testing

```bash
pytest
```
