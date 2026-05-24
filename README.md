# temp-max-predictor

LightGBM model to predict daily maximum temperature for Guarulhos (SBGR) using analog days methodology. Built for trading weather prediction markets on Polymarket.

## Approach

1. **LightGBM** — Gradient boosting on 23 features (morning observations, lags, cyclical date)
2. **Analog Days** — NearestNeighbors search for historical days with similar morning conditions
3. **Ensemble** — Adaptive weight (learned from validation set, typically 100% LGB)

## Project Structure

```
temp-max-predictor/
├── src/
│   ├── scrape_rp5.py    # Live data scraper from rp5.lv
│   ├── process_rp5.py   # Raw rp5.lv CSV → daily format
│   ├── data_loader.py   # Load, prepare features, lag features
│   ├── analog_days.py   # Analog days finder (NearestNeighbors)
│   ├── model.py         # LightGBM model wrapper
│   ├── train.py         # Training + hyperparameter search
│   ├── predict.py       # Prediction with adaptive ensemble weight
│   └── evaluate.py      # Evaluation
├── data/
│   ├── daily_at11.csv   # Processed daily data (≤11h observations)
│   └── sbgr_raw.csv     # Local copy of full rp5.lv export
├── models/              # Trained models (gitignored)
└── README.md
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On macOS, LightGBM requires OpenMP:
```bash
brew install libomp
```

## Usage (one-command daily prediction)

```bash
# Predict with all available data (auto-detects data/sbgr_raw.csv)
uv run python -m src.scrape_rp5

# Early-morning prediction (só dados até 09h)
uv run python -m src.scrape_rp5 --cutoff-hour 9

# Predict usando dados até 11h
uv run python -m src.scrape_rp5 --cutoff-hour 11
```

This single command:
1. Fetches the last ~27 hourly observations from rp5.lv
2. Merges them into `data/sbgr_raw.csv` (deduplicating by timestamp, file updated in-place)
3. Processes raw data to daily format using only observations ≤ `cutoff-hour` for features
4. Trains LightGBM on the full history (cutoff: 2024-01-01)
5. Computes optimal ensemble weight from validation
6. Predicts today's max temperature

### Train with hyperparameter search

```bash
uv run python -m src.train --data data/daily_at11.csv --tune
```

### Data format

Input is the raw semicolon-separated CSV exported from [rp5.lv](https://rp5.lv) for SBGR (METAR). The pipeline handles:
- Parse raw export → clean hourly observations
- Aggregate to daily: morning stats (3h–12h), 06h values, lag features
- Target: `temp_max_real` (max temperature from all hours)

### Features (23 total)

| Type | Features |
|------|----------|
| Morning (06h) | temp_06h, humidity_06h, pressure_06h |
| Morning stats | temp_morning_mean/std/min, humidity/pressure mean |
| Wind | wind_speed, wind_gust_max, wind_dir |
| Sky | cloud_cover, visibility |
| Lags | temp_max_yesterday, temp_range_yesterday, temp_mean_yesterday |
| Rolling | temp_max_3day_avg, temp_range_3day_avg |
| Cyclical | mes_sin/cos, dia_do_ano_sin/cos |

## Model Performance

| Metric | Value |
|--------|-------|
| Validation MAE | ~1.29°C |
| Validation RMSE | ~1.66°C |
| Ensemble weight | Adaptive (typically 100% LGB) |

## Prediction Output

```
=======================================================
  PREVISÃO PARA 2026-05-23
=======================================================
  Peso adaptativo: LGB=100% / Analog=0%

  Condições matinais:
    temp_06h: 16.0
    temp_09h: 16.0
    humidity_06h: 94.0
    pressure_06h: 768.1
    temp_morning_mean: 16.3
    wind_speed: 2.7
    cloud_cover: 7.6

  LightGBM:     17.7°C
  Analog Days:  20.2°C
  ───────────────────────
  ENSEMBLE:     17.7°C
  Real max:     18.0°C (erro: -0.3°C)
```
