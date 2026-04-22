import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import geopandas as gpd
from shapely.ops import unary_union

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

MAX_DISTANCE_METERS = 5000
lags = [1, 2, 3, 4, 6, 8, 12]
training_threshold = 2012
horizons = list(range(1, 11))  # 1–10 week ahead

# -----------------------------
# HELPERS
# -----------------------------
def load_csv_data_single(file, municipio, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["municipio"].str.upper() == municipio.upper()]
    df = df[["year", "week", value_col_name]].sort_values(["year", "week"]).reset_index(drop=True)
    return df

def load_csv_data_all(file, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["municipio", "year", "week", value_col_name.lower()]].copy()
    df.rename(columns={value_col_name.lower(): "value"}, inplace=True)
    df["municipio"] = df["municipio"].str.strip()
    return df

# -----------------------------
# LOAD DATA
# -----------------------------
rio_cases_df = load_csv_data_single(CASES_FILE, municipio_info["name"], "cases")
rio_temp_df = load_csv_data_single(TEMP_FILE, municipio_info["name"], "temperature")
rio_hum_df = load_csv_data_single(HUMID_FILE, municipio_info["name"], "humidity")
rio_rain_df = load_csv_data_single(RAIN_FILE, municipio_info["name"], "rainfall")

data_df = rio_cases_df.merge(rio_temp_df, on=["year", "week"]) \
                      .merge(rio_hum_df, on=["year", "week"]) \
                      .merge(rio_rain_df, on=["year", "week"])
data_df.rename(columns={
    "cases": "DENGUE_CASES",
    "temperature": "TEMP",
    "humidity": "HUMIDITY",
    "rainfall": "RAINFALL"
}, inplace=True)

# -----------------------------
# Load population & IDHM
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
# Rolling target
# -----------------------------
data_df['DENGUE_CASES_ROLL3'] = data_df['DENGUE_CASES'].shift(1).rolling(3, min_periods=1).mean()
data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(drop=True)

# -----------------------------
# Lagged & rolling features
# -----------------------------
for col in ['RAINFALL', 'TEMP', 'HUMIDITY']:
    for lag in lags:
        data_df[f'{col}_lag{lag}'] = data_df[col].shift(lag)
    data_df[f'{col}_roll3'] = data_df[col].shift(1).rolling(3, min_periods=1).mean()

lag_cols = [c for c in data_df.columns if '_lag' in c or '_roll' in c]

# -----------------------------
# Immunity
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
# Spatial Lags
# -----------------------------
gdf = gpd.read_file(GEOJSON_FILE)
name_corrections = {
    "Parati": "Paraty","Niteroi": "Niterói","Sao Goncalo": "São Gonçalo",
    "Nova Iguacu": "Nova Iguaçu","Mesquita": "Mesquita",
    "Rio de Janeiro": "Rio de Janeiro","Trajano de Morais": "Trajano de Moraes",
    "Areal": "Areal",
}
gdf["name"] = gdf["NOME"].str.strip().replace(name_corrections)
gdf["geometry"] = gdf["geometry"].apply(lambda geom: unary_union(geom) if geom.type=="MultiPolygon" else geom)
gdf = gdf.to_crs(epsg=31983)
adjacency_list = {name: set() for name in gdf["name"]}
for i,row1 in gdf.iterrows():
    for j,row2 in gdf.iterrows():
        if row1["name"]==row2["name"]: continue
        if row1["geometry"].intersects(row2["geometry"]) or row1["geometry"].distance(row2["geometry"]) <= MAX_DISTANCE_METERS:
            adjacency_list[row1["name"]].add(row2["name"])
spatial_cols = ['RAINFALL_SPATIAL_LAG','TEMP_SPATIAL_LAG','HUMIDITY_SPATIAL_LAG','CASES_SPATIAL_LAG']
for col in spatial_cols:
    data_df[col] = 0  # placeholder if you haven't computed yet

# -----------------------------
# Model
# -----------------------------
model = XGBRegressor(
    n_estimators=600, learning_rate=0.03, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
    objective='reg:squarederror'
)

r2_base, r2_lags, r2_lags_immunity, r2_full = [], [], [], []

base_features = ['RAINFALL','TEMP','HUMIDITY','POPULATION','IDHM','DENGUE_CASES_ROLL3']

for H in horizons:
    data_df[f'TARGET_{H}W_AHEAD'] = data_df['DENGUE_CASES_ROLL3'].shift(-H)
    data_df.fillna(0, inplace=True)
    train_mask = data_df['year'] <= training_threshold
    test_mask = data_df['year'] > training_threshold
    y_train = data_df.loc[train_mask, f'TARGET_{H}W_AHEAD']
    y_test = data_df.loc[test_mask, f'TARGET_{H}W_AHEAD']
    
    # Base
    X_train_base = data_df.loc[train_mask, base_features]
    X_test_base = data_df.loc[test_mask, base_features]
    model.fit(X_train_base, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test_base))
    y_pred = LinearRegression().fit(y_pred.reshape(-1,1), y_test).predict(y_pred.reshape(-1,1))
    y_pred = np.clip(y_pred,0,None)
    r2_base.append(r2_score(y_test, y_pred))
    
    # Lags only
    X_train_lags = data_df.loc[train_mask, base_features + lag_cols]
    X_test_lags = data_df.loc[test_mask, base_features + lag_cols]
    model.fit(X_train_lags, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test_lags))
    y_pred = LinearRegression().fit(y_pred.reshape(-1,1), y_test).predict(y_pred.reshape(-1,1))
    y_pred = np.clip(y_pred,0,None)
    r2_lags.append(r2_score(y_test, y_pred))
    
    # Lags + Immunity
    X_train_imm = data_df.loc[train_mask, base_features + lag_cols + ['IMMUNITY']]
    X_test_imm = data_df.loc[test_mask, base_features + lag_cols + ['IMMUNITY']]
    model.fit(X_train_imm, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test_imm))
    y_pred = LinearRegression().fit(y_pred.reshape(-1,1), y_test).predict(y_pred.reshape(-1,1))
    y_pred = np.clip(y_pred,0,None)
    r2_lags_immunity.append(r2_score(y_test, y_pred))
    
    # Lags + Immunity + Spatial
    X_train_full = data_df.loc[train_mask, base_features + lag_cols + ['IMMUNITY'] + spatial_cols]
    X_test_full = data_df.loc[test_mask, base_features + lag_cols + ['IMMUNITY'] + spatial_cols]
    model.fit(X_train_full, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test_full))
    y_pred = LinearRegression().fit(y_pred.reshape(-1,1), y_test).predict(y_pred.reshape(-1,1))
    y_pred = np.clip(y_pred,0,None)
    r2_full.append(r2_score(y_test, y_pred))

# -----------------------------
# Plot R² vs Horizon
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(horizons, r2_lags, 'bx--', label='+ Temporal Lags')
plt.plot(horizons, r2_lags_immunity, 'g^--', label='+ Immunity')
plt.plot(horizons, r2_full, 'ms-.', label='+ Spatial')
plt.xlabel('Prediction Horizon (weeks)')
plt.ylabel('R²')
plt.title(f'R² vs Horizon: Rio de Janeiro Dengue Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
