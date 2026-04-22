import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

# --- geopandas for adjacency ---
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

HORIZON = 4  # weeks ahead
lags = [1, 2, 3, 4, 6, 8, 12]
MAX_DISTANCE_METERS = 5000

# -----------------------------
# HELPERS: CSV loaders
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
# Build adjacency list from GeoJSON
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

if municipio_info["name"] not in adjacency_list:
    raise KeyError(f"{municipio_info['name']} not found in adjacency list")
print("Neighbors for", municipio_info["name"], ":", adjacency_list[municipio_info["name"]])

# -----------------------------
# Load data for target municipio
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

neighbors = adjacency_list[municipio_info["name"]]

cases_spatial, rain_spatial, temp_spatial, hum_spatial, num_neighbors_available = [], [], [], [], []

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
    num_neighbors_available.append(len(neigh_vals_cases))

data_df["CASES_SPATIAL_LAG"] = cases_spatial
data_df["RAINFALL_SPATIAL_LAG"] = rain_spatial
data_df["TEMP_SPATIAL_LAG"] = temp_spatial
data_df["HUMIDITY_SPATIAL_LAG"] = hum_spatial
data_df["NUM_NEIGHBORS_AVAILABLE"] = num_neighbors_available

data_df["CASES_SPATIAL_LAG"].fillna(data_df["DENGUE_CASES"].shift(1).fillna(0), inplace=True)
data_df["RAINFALL_SPATIAL_LAG"].fillna(data_df["RAINFALL"].shift(1).fillna(0), inplace=True)
data_df["TEMP_SPATIAL_LAG"].fillna(data_df["TEMP"].shift(1).fillna(0), inplace=True)
data_df["HUMIDITY_SPATIAL_LAG"].fillna(data_df["HUMIDITY"].shift(1).fillna(0), inplace=True)

# -----------------------------
# Population & IDHM
# -----------------------------
pop_df = pd.read_csv(POP_FILE)
pop_df.columns = [c.strip().lower() for c in pop_df.columns]
idhm_df = pd.read_csv(IDHM_FILE)
idhm_df.columns = [c.strip().lower() for c in idhm_df.columns]

data_df["POPULATION"] = pop_df.loc[pop_df["municipio"].str.upper() == municipio_info["name"].upper(),"population"].values[0]
data_df["IDHM"] = idhm_df.loc[idhm_df["municipio"].str.upper() == municipio_info["name"].upper(),"idhm"].values[0]

# -----------------------------
# Filter years 2010–2025
# -----------------------------
data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(drop=True)

# -----------------------------
# Lagged & rolling features (past-only)
# -----------------------------
for col in ['RAINFALL','TEMP','HUMIDITY','RAINFALL_SPATIAL_LAG','TEMP_SPATIAL_LAG','HUMIDITY_SPATIAL_LAG','CASES_SPATIAL_LAG']:
    for lag in lags:
        data_df[f'{col}_lag{lag}'] = data_df[col].shift(lag)
    data_df[f'{col}_roll3'] = data_df[col].shift(1).rolling(3, min_periods=1).mean()

# -----------------------------
# Immunity (past-only)
# -----------------------------
decay, K = 0.8, 10
cases_series = data_df['DENGUE_CASES'].values
immunity = []
for i in range(len(cases_series)):
    past_cases = sum(cases_series[i - k]*np.exp(-decay*k) for k in range(1,K+1) if i-k >= 0)
    immunity.append(past_cases)
data_df['IMMUNITY'] = immunity

# -----------------------------
# Smoothed target (past-only)
# -----------------------------
data_df['DENGUE_CASES_ROLL3'] = data_df['DENGUE_CASES'].shift(1).rolling(3, min_periods=1).mean()

# -----------------------------
# Shift target for HORIZON
# -----------------------------
data_df[f'TARGET_{HORIZON}W_AHEAD'] = data_df['DENGUE_CASES_ROLL3'].shift(-HORIZON)
data_df.fillna(0, inplace=True)

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = ['RAINFALL','TEMP','HUMIDITY','RAINFALL_SPATIAL_LAG','TEMP_SPATIAL_LAG','HUMIDITY_SPATIAL_LAG','CASES_SPATIAL_LAG','POPULATION','IDHM','IMMUNITY'] + [c for c in data_df.columns if '_lag' in c or '_roll' in c]

# -----------------------------
# Train/Test Split
# -----------------------------
training_threshold = 2012
train_mask = data_df['year'] <= training_threshold
test_mask = data_df['year'] > training_threshold

X_train = data_df.loc[train_mask, feature_cols]
y_train = data_df.loc[train_mask, f'TARGET_{HORIZON}W_AHEAD']
X_test = data_df.loc[test_mask, feature_cols]
y_test = data_df.loc[test_mask, f'TARGET_{HORIZON}W_AHEAD']

weeks_since_test_start = (data_df.loc[test_mask,'year']-data_df.loc[test_mask,'year'].min())*52 + data_df.loc[test_mask,'week']

# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestRegressor(n_estimators=600, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Linear calibration
# -----------------------------
calib = LinearRegression().fit(y_pred.reshape(-1,1), y_test)
y_pred_calib = calib.predict(y_pred.reshape(-1,1))
y_pred_calib = np.clip(y_pred_calib, 0, None)

# -----------------------------
# Weekly output
# -----------------------------
print(f"=== Weekly Data for Rio de Janeiro ({HORIZON}-week ahead prediction) ===\n")
for i, row in X_test.iterrows():
    week_idx = weeks_since_test_start.iloc[i - X_test.index[0]]
    print(f"Week {int(week_idx)}: Actual={int(y_test.iloc[i - X_test.index[0]])}, "
          f"Predicted={y_pred_calib[i - X_test.index[0]]:.1f}, "
          f"Population={int(row['POPULATION'])}, IDHM={row['IDHM']:.3f}, "
          f"Immunity={row['IMMUNITY']:.1f}, Cases_spatial={row['CASES_SPATIAL_LAG']:.1f}")

# -----------------------------
# Accuracy metrics
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred_calib))
mae = mean_absolute_error(y_test, y_pred_calib)
r2 = r2_score(y_test, y_pred_calib)

print(f"\nModel Accuracy on Test Set ({HORIZON}-week ahead):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R²: {r2:.3f}\n")

# -----------------------------
# Plot
# -----------------------------
test_df = data_df.loc[test_mask].copy()
test_df['week_since_test_start'] = weeks_since_test_start.values

plt.figure(figsize=(18,7))
plt.plot(test_df['week_since_test_start'], y_test, 'ro-', label=f'Actual Dengue Cases ({HORIZON}-week ahead)')
plt.plot(test_df['week_since_test_start'], y_pred_calib, 'bx--', label='Predicted Dengue Cases (Calibrated)')
plt.xlabel('Weeks Since Test Period Start')
plt.ylabel('Dengue Cases')
plt.title(f'Dengue Cases: Rio de Janeiro ({training_threshold}–2025, {HORIZON}-week ahead)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# SHAP explanations
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type="bar", show=True)

top_feature = feature_cols[np.argmax(np.abs(shap_values).mean(axis=0))]
shap.dependence_plot(top_feature, shap_values, X_test, show=True)
