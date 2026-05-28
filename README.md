# XGBoost Dengue Prediction Model

This repository contains Python scripts for forecasting dengue cases in Rio de Janeiro using temporal, socioeconomic, environmental, and spatial-lag features.

The main workflow compares multiple regression models across forecast horizons from 1 to 15 weeks and evaluates performance using RMSE, MAE, and R².

## Project Overview

The model uses weekly municipality-level data from Rio de Janeiro, Brazil. For the target municipality, Rio de Janeiro, the scripts combine dengue case counts with temperature, humidity, rainfall, population, IDHM, temporal lag features, rolling averages, immunity-style decay features, and spatial lag features based on neighboring municipalities.

The main script compares:

- XGBoost
- Random Forest
- Gradient Boosting
- Ridge Regression
- Linear Regression

## Project Structure

```text
XGBoost-Prediction-Model/
├── data/
│   ├── RJ.json
│   ├── cases.csv
│   ├── humidity.csv
│   ├── idhm.csv
│   ├── population.csv
│   ├── rainfall.csv
│   └── temperature.csv
├── publication_ready_multi_model.py
├── Optuna_model_optimization.py
├── training_ratio.py
├── multi_model_horizon_metrics.csv
├── multi_model_horizon_rmse_plot.png
├── xgboost_training_threshold_metrics.csv
├── xgboost_rmse_mae_vs_training_threshold.png
└── README.md
```

## Data Files

The project expects a `data` folder in the repository root.

Required files:

```text
data/cases.csv
data/humidity.csv
data/idhm.csv
data/population.csv
data/rainfall.csv
data/temperature.csv
data/RJ.json
```

Expected CSV columns:

```text
cases.csv: municipio,year,week,cases
humidity.csv: municipio,year,week,humidity
idhm.csv: municipio,year,week,idhm
population.csv: municipio,year,week,population
rainfall.csv: municipio,year,week,rainfall
temperature.csv: municipio,year,week,temperature
```

The GeoJSON file `RJ.json` must contain Rio de Janeiro municipality boundaries and a municipality name property called `NOME`.

## Feature Engineering

The model builds several feature groups:

- Temporal lags for dengue cases, rainfall, temperature, humidity, and spatial lag variables
- Three-week rolling averages using only past values
- Spatial lag features from neighboring municipalities
- Population and IDHM socioeconomic indicators
- An immunity-style feature using exponentially decayed past dengue cases
- A year index to represent long-term temporal progression

Spatial neighbors are built from municipality boundaries in `RJ.json`. Municipalities are treated as neighbors if their polygons touch or if they are within 5,000 meters in the projected coordinate system.

## Train, Validation, and Test Split

The main model script uses the following split:

```text
Training:   year <= 2018
Validation: 2019-2021
Testing:    2022-2025
```

Forecast horizons are evaluated from 1 to 15 weeks ahead.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib scikit-learn xgboost optuna geopandas shapely
```

If `geopandas` fails to install, install the geospatial dependencies first. On macOS with Homebrew:

```bash
brew install gdal geos proj spatialindex
python -m pip install geopandas
```

## Running the Main Model Comparison

Run:

```bash
python publication_ready_multi_model.py
```

This script:

1. Loads the weekly dengue, climate, socioeconomic, and GeoJSON data
2. Builds temporal and spatial features
3. Trains multiple models across forecast horizons from 1 to 15 weeks
4. Calibrates predictions using linear regression on the validation split
5. Saves model metrics and a publication-ready RMSE plot

Outputs:

```text
multi_model_horizon_metrics.csv
multi_model_horizon_rmse_plot.png
```

## Running Hyperparameter Optimization

Run:

```bash
python Optuna_model_optimization.py
```

This script uses Optuna and time-series cross-validation to tune model hyperparameters.

## Running Training Ratio Analysis

Run:

```bash
python training_ratio.py
```

This script evaluates XGBoost performance under different training-year thresholds.

Outputs:

```text
xgboost_training_threshold_metrics.csv
xgboost_rmse_mae_vs_training_threshold.png
```

## Main Outputs

`multi_model_horizon_metrics.csv` contains validation and test metrics for each model and forecast horizon.

`multi_model_horizon_rmse_plot.png` shows test RMSE by forecast horizon for all compared models.

`xgboost_training_threshold_metrics.csv` contains XGBoost performance across different training splits.

`xgboost_rmse_mae_vs_training_threshold.png` shows RMSE and MAE as the training threshold changes.

## Notes

- The current target municipality is Rio de Janeiro.
- The forecast target is dengue cases shifted ahead by the selected horizon.
- XGBoost uses a log-transformed target during training in the main comparison script.
- Model predictions are calibrated using a linear regression fit on the validation predictions.
- Missing spatial lag values are filled using prior target-municipality values.

## Requirements

Main Python dependencies:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
optuna
geopandas
shapely
```

## License and Data Sources

This project uses dengue, climate, socioeconomic, and geographic data for Rio de Janeiro municipalities. If publishing the repository publicly, include citations or links for the original data sources used to generate the CSV and GeoJSON files.

