import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Models
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# SHAP
import shap

# Geospatial
import geopandas as gpd
from shapely.ops import unary_union

# -----------------------------
# PARAMETERS
# -----------------------------
municipio_info = {"name": "Rio de Janeiro"}
DATA_DIR = os.path.expanduser("~/Desktop")
CASES_FILE = os.path.join(DATA_DIR, "cases.csv")
TEMP_FILE = os.path.join(DATA_DIR, "temperature.csv")
HUMID_FILE = os.path.join(DATA_DIR, "humidity.csv")
RAIN_FILE = os.path.join(DATA_DIR, "rainfall.csv")
POP_FILE = os.path.join(DATA_DIR, "population.csv")
IDHM_FILE = os.path.join(DATA_DIR, "idhm.csv")
GEOJSON_FILE = os.path.join(DATA_DIR, "RJ.json")  # GeoJSON for adjacency

HORIZON = 4
lags = [1, 2, 3, 4, 6, 8, 12]
MAX_DISTANCE_METERS = 5000

# -----------------------------
# CSV loaders
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
    if value_col_name.lower() not in df.columns:
        raise ValueError(f"{value_col_name} not found in {file} columns")
    df = df[["municipio", "year", "week", value_col_name.lower()]].copy()
    df.rename(columns={value_col_name.lower(): "value"}, inplace=True)
    df["municipio"] = df["municipio"].str.strip()
    return df

# -----------------------------
# Build adjacency
# -----------------------------
gdf = gpd.read_file(GEOJSON_FILE)
name_corrections = {
    "Parati": "Paraty",
    "Niteroi": "Niterói",
    "Sao Goncalo": "São Gonçalo",
    "Nova Iguacu": "Nova Iguaçu",
    "Mesquita": "Mesquita",
    "Rio de Janeiro": "Rio de Janeiro",
    "Trajano de Morais": "Trajano de Moraes",
    "Areal": "Areal",
}
gdf["name"] = gdf["NOME"].str.strip().replace(name_corrections)
gdf["geometry"] = gdf["geometry"].apply(lambda geom: unary_union(geom) if geom.type == "MultiPolygon" else geom)
gdf = gdf.to_crs(epsg=31983)

adjacency_list = {name: set() for name in gdf["name"]}
for i, row1 in gdf.iterrows():
    name1 = row1["name"]
    poly1 = row1["geometry"]
    for j, row2 in gdf.iterrows():
        name2 = row2["name"]
        if name1 == name2:
            continue
        poly2 = row2["geometry"]
        if poly1.intersects(poly2) or poly1.distance(poly2) <= MAX_DISTANCE_METERS:
            adjacency_list[name1].add(name2)
            adjacency_list[name2].add(name1)
adjacency_list = {k: sorted(list(v)) for k, v in adjacency_list.items()}

neighbors = adjacency_list[municipio_info["name"]]

# -----------------------------
# Load target municipio
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
# Load all municipios for spatial lag
# -----------------------------
cases_all = load_csv_data_all(CASES_FILE, "cases")
temp_all = load_csv_data_all(TEMP_FILE, "temperature")
hum_all = load_csv_data_all(HUMID_FILE, "humidity")
rain_all = load_csv_data_all(RAIN_FILE, "rainfall")

cases_idx = cases_all.groupby(["year","week","municipio"])["value"].first().reset_index().set_index(["year","week","municipio"])
temp_idx = temp_all.groupby(["year","week","municipio"])["value"].first().reset_index().set_index(["year","week","municipio"])
hum_idx = hum_all.groupby(["year","week","municipio"])["value"].first().reset_index().set_index(["year","week","municipio"])
rain_idx = rain_all.groupby(["year","week","municipio"])["value"].first().reset_index().set_index(["year","week","municipio"])

cases_spatial, rain_spatial, temp_spatial, hum_spatial = [], [], [], []
for _, row in data_df.iterrows():
    y, w = int(row["year"]), int(row["week"])
    neigh_vals_cases, neigh_vals_rain, neigh_vals_temp, neigh_vals_hum = [], [], [], []
    for n in neighbors:
        try: neigh_vals_cases.append(cases_idx.loc[(y,w,n),"value"])
        except KeyError: pass
        try: neigh_vals_rain.append(rain_idx.loc[(y,w,n),"value"])
        except KeyError: pass
        try: neigh_vals_temp.append(temp_idx.loc[(y,w,n),"value"])
        except KeyError: pass
        try: neigh_vals_hum.append(hum_idx.loc[(y,w,n),"value"])
        except KeyError: pass
    cases_spatial.append(np.mean(neigh_vals_cases) if neigh_vals_cases else np.nan)
    rain_spatial.append(np.mean(neigh_vals_rain) if neigh_vals_rain else np.nan)
    temp_spatial.append(np.mean(neigh_vals_temp) if neigh_vals_temp else np.nan)
    hum_spatial.append(np.mean(neigh_vals_hum) if neigh_vals_hum else np.nan)

data_df["CASES_SPATIAL_LAG"] = cases_spatial
data_df["RAINFALL_SPATIAL_LAG"] = rain_spatial
data_df["TEMP_SPATIAL_LAG"] = temp_spatial
data_df["HUMIDITY_SPATIAL_LAG"] = hum_spatial

# Fill missing spatial lags
for col in ["CASES_SPATIAL_LAG","RAINFALL_SPATIAL_LAG","TEMP_SPATIAL_LAG","HUMIDITY_SPATIAL_LAG"]:
    data_df[col].fillna(data_df[col].shift(1).fillna(0), inplace=True)

# -----------------------------
# Population & IDHM
# -----------------------------
pop_df = pd.read_csv(POP_FILE)
idhm_df = pd.read_csv(IDHM_FILE)
data_df["POPULATION"] = pop_df.loc[pop_df["municipio"].str.upper() == municipio_info["name"].upper(),"population"].values[0]
data_df["IDHM"] = idhm_df.loc[idhm_df["municipio"].str.upper() == municipio_info["name"].upper(),"idhm"].values[0]

# -----------------------------
# Filter years
# -----------------------------
data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(drop=True)

# -----------------------------
# Lag & rolling features
# -----------------------------
for col in ['RAINFALL','TEMP','HUMIDITY','RAINFALL_SPATIAL_LAG','TEMP_SPATIAL_LAG','HUMIDITY_SPATIAL_LAG','CASES_SPATIAL_LAG']:
    for lag in lags:
        data_df[f'{col}_lag{lag}'] = data_df[col].shift(lag)
    data_df[f'{col}_roll3'] = data_df[col].shift(1).rolling(3, min_periods=1).mean()

# -----------------------------
# Immunity
# -----------------------------
decay, K = 0.8, 10
cases_series = data_df['DENGUE_CASES'].values
immunity = []
for i in range(len(cases_series)):
    past_cases = sum(cases_series[i - k]*np.exp(-decay*k) for k in range(1,K+1) if i-k >= 0)
    immunity.append(past_cases)
data_df['IMMUNITY'] = immunity

# -----------------------------
# Smoothed target
# -----------------------------
data_df['DENGUE_CASES_ROLL3'] = data_df['DENGUE_CASES'].shift(1).rolling(3, min_periods=1).mean()
data_df[f'TARGET_{HORIZON}W_AHEAD'] = data_df['DENGUE_CASES_ROLL3'].shift(-HORIZON)
data_df.fillna(0, inplace=True)

# -----------------------------
# Features & split
# -----------------------------
feature_cols = ['RAINFALL','TEMP','HUMIDITY','RAINFALL_SPATIAL_LAG','TEMP_SPATIAL_LAG','HUMIDITY_SPATIAL_LAG','CASES_SPATIAL_LAG','POPULATION','IDHM','IMMUNITY'] + [c for c in data_df.columns if '_lag' in c or '_roll' in c]
training_threshold = 2012
train_mask = data_df['year'] <= training_threshold
test_mask = data_df['year'] > training_threshold
X_train = data_df.loc[train_mask, feature_cols]
y_train = data_df.loc[train_mask, f'TARGET_{HORIZON}W_AHEAD']
X_test = data_df.loc[test_mask, feature_cols]
y_test = data_df.loc[test_mask, f'TARGET_{HORIZON}W_AHEAD']

# -----------------------------
# Scale for linear models
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Define models
# -----------------------------
#use same parameters for each model, and explain why we use those parameters

models = {
    "Random Forest": RandomForestRegressor(n_estimators=645, max_depth=9, min_samples_split = 2, min_samples_leaf = 1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=639, learning_rate=0.06053197206271428, max_depth=3, subsample = 0.8013654487520273),
    "XGBoost": XGBRegressor(n_estimators=623, learning_rate=0.02081058596075694, max_depth=3, subsample=0.8242253369150914, colsample_bytree=0.9918052007526491),
    "Ridge Regression": Ridge(alpha=2.4304352434763596),
    "Linear Regression": LinearRegression()
}

# -----------------------------
# Train & evaluate all models
# -----------------------------
results = {}
for name, model in models.items():
    if "Ridge" in name or "Linear" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Tree-based: log-transform
        y_train_log = np.log1p(y_train)
        model.fit(X_train, y_train_log)
        y_pred = np.expm1(model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (rmse, mae, r2)

# -----------------------------
# Print metrics together
# -----------------------------
print("\n=== Model Comparison ===\n")
for name, (rmse, mae, r2) in results.items():
    print(f"{name:<20} RMSE: {rmse:8.2f}  MAE: {mae:8.2f}  R²: {r2:6.3f}")
