import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# Parameters
# -----------------------------
municipio_info = {"name": "Rio de Janeiro"}
DATA_DIR = os.path.expanduser("~/Desktop")
CASES_FILE = os.path.join(DATA_DIR, "cases.csv")
TEMP_FILE = os.path.join(DATA_DIR, "temperature.csv")
HUMID_FILE = os.path.join(DATA_DIR, "humidity.csv")
RAIN_FILE = os.path.join(DATA_DIR, "rainfall.csv")
POP_FILE = os.path.join(DATA_DIR, "population.csv")
IDHM_FILE = os.path.join(DATA_DIR, "idhm.csv")

lags = [1, 2, 3, 4, 6, 8, 12]  # Lag features
training_years = list(range(2011, 2025))
horizons = list(range(1, 4))  # Horizons 1–4 weeks

# -----------------------------
# Load CSV helper
# -----------------------------
def load_csv_data(file, municipio, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["municipio"].str.upper() == municipio.upper()]
    df = df[["year", "week", value_col_name]].sort_values(["year", "week"]).reset_index(drop=True)
    return df

# -----------------------------
# Load datasets once
# -----------------------------
rio_cases_df = load_csv_data(CASES_FILE, municipio_info["name"], "cases")
rio_temp_df = load_csv_data(TEMP_FILE, municipio_info["name"], "temperature")
rio_hum_df = load_csv_data(HUMID_FILE, municipio_info["name"], "humidity")
rio_rain_df = load_csv_data(RAIN_FILE, municipio_info["name"], "rainfall")

data_df = rio_cases_df.merge(rio_temp_df, on=["year", "week"], how="inner") \
                      .merge(rio_hum_df, on=["year", "week"], how="inner") \
                      .merge(rio_rain_df, on=["year", "week"], how="inner")

data_df.rename(columns={
    "cases": "DENGUE_CASES",
    "temperature": "TEMP",
    "humidity": "HUMIDITY",
    "rainfall": "RAINFALL"
}, inplace=True)

# Add population and IDHM
pop_df = pd.read_csv(POP_FILE)
pop_df.columns = [c.strip().lower() for c in pop_df.columns]
idhm_df = pd.read_csv(IDHM_FILE)
idhm_df.columns = [c.strip().lower() for c in idhm_df.columns]

pop_val = pop_df.loc[pop_df["municipio"].str.upper() == municipio_info["name"].upper(), "population"].values[0]
idhm_val = idhm_df.loc[idhm_df["municipio"].str.upper() == municipio_info["name"].upper(), "idhm"].values[0]

data_df["POPULATION"] = pop_val
data_df["IDHM"] = idhm_val
data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(drop=True)

# -----------------------------
# Lagged & rolling features
# -----------------------------
for col in ['RAINFALL', 'TEMP', 'HUMIDITY']:
    for lag in lags:
        data_df[f'{col}_lag{lag}'] = data_df[col].shift(lag)
    data_df[f'{col}_roll3'] = data_df[col].shift(1).rolling(3, min_periods=1).mean()

# Immunity feature
decay = 0.8
K = 10
cases_series = data_df['DENGUE_CASES'].values
immunity = []
for i in range(len(cases_series)):
    past_cases = sum(cases_series[i - k] * np.exp(-decay * k) for k in range(1, K + 1) if i - k >= 0)
    immunity.append(past_cases)
data_df['IMMUNITY'] = immunity

# Smoothed target
data_df['DENGUE_CASES_ROLL3'] = data_df['DENGUE_CASES'].shift(1).rolling(3, min_periods=1).mean()
data_df.fillna(0, inplace=True)

feature_cols = ['RAINFALL', 'TEMP', 'HUMIDITY', 'POPULATION', 'IDHM', 'IMMUNITY'] + \
               [c for c in data_df.columns if '_lag' in c or '_roll' in c]

# -----------------------------
# Store metrics
# -----------------------------
metrics = {h: {"R2": [], "RMSE": [], "MAE": []} for h in horizons}

# -----------------------------
# Run models
# -----------------------------
for horizon in horizons:
    data_df[f'TARGET_{horizon}W_AHEAD'] = data_df['DENGUE_CASES_ROLL3'].shift(-horizon)
    data_df.fillna(0, inplace=True)

    for training_threshold in training_years:
        train_mask = data_df['year'] <= training_threshold
        test_mask = data_df['year'] > training_threshold

        X_train = data_df.loc[train_mask, feature_cols]
        y_train = data_df.loc[train_mask, f'TARGET_{horizon}W_AHEAD']
        X_test = data_df.loc[test_mask, feature_cols]
        y_test = data_df.loc[test_mask, f'TARGET_{horizon}W_AHEAD']

        y_train_log = np.log1p(y_train)
        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        calib = LinearRegression().fit(y_pred.reshape(-1, 1), y_test)
        y_pred_calib = np.clip(calib.predict(y_pred.reshape(-1, 1)), 0, None)

        # Store metrics, scale R2 by 100
        metrics[horizon]["R2"].append(r2_score(y_test, y_pred_calib) * 100)
        metrics[horizon]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred_calib)))
        metrics[horizon]["MAE"].append(mean_absolute_error(y_test, y_pred_calib))

# -----------------------------
# Plot all metrics for horizons 1–4 on same plot
# -----------------------------
plt.figure(figsize=(14, 7))

colors = ['C0', 'C1', 'C2', 'C3']  # One color per horizon
markers = {'R2':'o', 'RMSE':'s', 'MAE':'^'}

for i, h in enumerate(horizons):
    for metric_name in ["R2", "RMSE", "MAE"]:
        plt.plot(training_years, metrics[h][metric_name],
                 marker=markers[metric_name],
                 color=colors[i],
                 linestyle='-' if metric_name=='R2' else '--' if metric_name=='RMSE' else ':',
                 label=f'H{h} {metric_name}')

plt.xlabel('Training End Year')
plt.ylabel('Metric Value (R² scaled by 100)')
plt.title('R², RMSE, and MAE vs Training Year (Horizons 1–4 Weeks)')
plt.legend(ncol=2, fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
