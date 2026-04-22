import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# PARAMETERS
# -----------------------------
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")

municipio_info = {"name": "Rio de Janeiro"}
CASES_FILE = os.path.join(data_dir, "cases.csv")
TEMP_FILE = os.path.join(data_dir, "temperature.csv")
HUMID_FILE = os.path.join(data_dir, "humidity.csv")
RAIN_FILE = os.path.join(data_dir, "rainfall.csv")
IDHM_FILE = os.path.join(data_dir, "idhm.csv")
POP_FILE = os.path.join(data_dir, "population.csv")
GEOJSON_FILE = os.path.join(data_dir, "RJ.json")  # GeoJSON for adjacency

HORIZON = 2 # Predict HORIZON weeks ahead
lags = [1, 2, 3, 4, 6, 8, 12]  # Lag features

# -----------------------------
# 1) Load CSV helper
# -----------------------------
def load_csv_data(file, municipio, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["municipio"].str.upper() == municipio.upper()]
    df = df[["year", "week", value_col_name]].sort_values(["year", "week"]).reset_index(drop=True)
    return df

# -----------------------------
# 2) Load datasets
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

# -----------------------------
# 3) Add population and IDHM
# -----------------------------
pop_df = pd.read_csv(POP_FILE)
pop_df.columns = [c.strip().lower() for c in pop_df.columns]
idhm_df = pd.read_csv(IDHM_FILE)
idhm_df.columns = [c.strip().lower() for c in idhm_df.columns]

pop_val = pop_df.loc[pop_df["municipio"].str.upper() == municipio_info["name"].upper(), "population"].values[0]
idhm_val = idhm_df.loc[idhm_df["municipio"].str.upper() == municipio_info["name"].upper(), "idhm"].values[0]

data_df["POPULATION"] = pop_val
data_df["IDHM"] = idhm_val

# -----------------------------
# 4) Filter for 2010–2025
# -----------------------------
data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(drop=True)

# -----------------------------
# 5) Lagged and rolling features (past-only)
# -----------------------------
for col in ['RAINFALL', 'TEMP', 'HUMIDITY']:
    for lag in lags:
        data_df[f'{col}_lag{lag}'] = data_df[col].shift(lag)
    data_df[f'{col}_roll3'] = data_df[col].shift(1).rolling(3, min_periods=1).mean()

# -----------------------------
# 5b) Immunity feature (past-only)
# -----------------------------
decay = 0.8
K = 10
cases_series = data_df['DENGUE_CASES'].values
immunity = []
for i in range(len(cases_series)):
    past_cases = sum(cases_series[i - k] * np.exp(-decay * k) for k in range(1, K + 1) if i - k >= 0)
    immunity.append(past_cases)
data_df['IMMUNITY'] = immunity

# -----------------------------
# 5c) Smoothed target (past-only)
# -----------------------------
data_df['DENGUE_CASES_ROLL3'] = data_df['DENGUE_CASES'].shift(1).rolling(3, min_periods=1).mean()

# -----------------------------
# 5d) Shift target for multi-week prediction
# -----------------------------
data_df[f'TARGET_{HORIZON}W_AHEAD'] = data_df['DENGUE_CASES_ROLL3'].shift(-HORIZON)
data_df.fillna(0, inplace=True)

# -----------------------------
# 5e) Feature columns
# -----------------------------
feature_cols = ['RAINFALL', 'TEMP', 'HUMIDITY', 'POPULATION', 'IDHM', 'IMMUNITY'] + \
               [c for c in data_df.columns if '_lag' in c or '_roll' in c]

# -----------------------------
# 6) Train/Test Split (train ≤2024, test >2024)
# -----------------------------
training_threshold = 2012
train_mask = data_df['year'] <= training_threshold
test_mask = data_df['year'] > training_threshold

X_train = data_df.loc[train_mask, feature_cols]
y_train = data_df.loc[train_mask, f'TARGET_{HORIZON}W_AHEAD']
X_test = data_df.loc[test_mask, feature_cols]
y_test = data_df.loc[test_mask, f'TARGET_{HORIZON}W_AHEAD']

weeks_since_test_start = (data_df.loc[test_mask, 'year'] - data_df.loc[test_mask, 'year'].min()) * 52 + data_df.loc[test_mask, 'week']

# -----------------------------
# 7) Train XGBoost with log-transform
# -----------------------------
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

# -----------------------------
# 7b) Linear calibration
# -----------------------------
calib = LinearRegression().fit(y_pred.reshape(-1, 1), y_test)
y_pred_calib = calib.predict(y_pred.reshape(-1, 1))
y_pred_calib = np.clip(y_pred_calib, 0, None)

# -----------------------------
# 8) Weekly output
# -----------------------------
print(f"=== Weekly Data for Rio de Janeiro ({HORIZON}-week ahead prediction) ===\n")
for i, row in X_test.iterrows():
    week_idx = weeks_since_test_start.iloc[i - X_test.index[0]]
    print(f"Week {int(week_idx)}: Actual={int(y_test.iloc[i - X_test.index[0]])}, "
          f"Predicted={y_pred_calib[i - X_test.index[0]]:.1f}, "
          f"Population={int(row['POPULATION'])}, IDHM={row['IDHM']:.3f}, Immunity={row['IMMUNITY']:.1f}")

# -----------------------------
# 9) Accuracy metrics
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred_calib))
mae = mean_absolute_error(y_test, y_pred_calib)
r2 = r2_score(y_test, y_pred_calib)

print(f"\nModel Accuracy on Test Set ({HORIZON}-week ahead):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R²: {r2:.3f}\n")

# -----------------------------
# 10) Plot
# -----------------------------
test_df = data_df.loc[test_mask].copy()
test_df['week_since_test_start'] = weeks_since_test_start.values

plt.figure(figsize=(18, 7))
plt.plot(test_df['week_since_test_start'], y_test, 'ro-', label=f'Actual Dengue Cases ({HORIZON}-week ahead)')
plt.plot(test_df['week_since_test_start'], y_pred_calib, 'bx--', label='Predicted Dengue Cases (Calibrated)')
plt.xlabel('Weeks Since Test Period Start')
plt.ylabel('Dengue Cases')
plt.title(f'Dengue Cases: Rio de Janeiro ({training_threshold}–2025, {HORIZON}-week ahead)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
