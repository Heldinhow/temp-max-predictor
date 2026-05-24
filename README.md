# temp-max-predictor

LightGBM model to predict daily maximum temperature using analog days methodology.

Built for trading weather prediction markets on Polymarket (São Paulo / GRU airport).

## Approach

1. **Analog Days** - Find historical days with similar morning conditions (temp, humidity, pressure) and use their outcomes
2. **LightGBM** - Gradient boosting on tabular features
3. **Ensemble** - Weighted combination of both approaches

## Project Structure

```
temp-max-predictor/
├── src/
│   ├── data_loader.py   # Load and prepare CSV data
│   ├── features.py      # Feature engineering
│   ├── analog_days.py   # Analog days finder
│   ├── model.py         # LightGBM model class
│   ├── train.py         # Training script
│   ├── predict.py       # Prediction script
│   └── evaluate.py      # Evaluation script
├── data/
│   └── .gitkeep
├── notebooks/
│   └── eda.ipynb        # Exploratory data analysis
├── models/              # Trained models (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python -m src.train --data data/weather.csv --cutoff 2024-01-01
```

### 2. Evaluate

```bash
python -m src.evaluate --data data/weather.csv --cutoff 2024-01-01
```

### 3. Predict today

```bash
python -m src.predict --data data/today.csv
```

## Data Format

CSV should contain at minimum:

| Column | Description |
|--------|-------------|
| date | Date (YYYY-MM-DD) |
| temp_06h | Temperature at 06:00 (or earliest morning observation) |
| humidity_06h | Humidity at 06:00 |
| pressure_06h | Pressure at 06:00 |
| temp_max_real | Actual maximum temperature (target) |

Optional:
- wind_speed, wind_dir
- cloud_cover, visibility

## Output

Predictions include:
- Point estimate (°C)
- Probability for each threshold relevant to Polymarket (≥20°C, ≥21°C, etc.)
- Confidence based on analog days spread