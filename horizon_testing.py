import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


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
GEOJSON_FILE = os.path.join(data_dir, "RJ.json")

LAGS = [1, 2, 3, 4, 6, 8, 12]
HORIZONS = range(1, 16)
MAX_DISTANCE_METERS = 5000

TRAIN_END_YEAR = 2010
VALID_END_YEAR = 2012
TEST_START_YEAR = 2013
TEST_END_YEAR = 2025

PLOT_FILE = os.path.join(base_dir, "xgboost_horizon_rmse_plot.png")
METRICS_FILE = os.path.join(base_dir, "xgboost_horizon_metrics.csv")


# -----------------------------
# Helpers
# -----------------------------
def load_csv_data_single(file, municipio, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df = df[df["municipio"].str.upper() == municipio.upper()]
    df = (
        df[["year", "week", value_col_name.lower()]]
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )
    return df


def load_csv_data_all(file, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    value_col_name = value_col_name.lower()
    if value_col_name not in df.columns:
        raise ValueError(f"{value_col_name} not found in {file} columns")

    df = df[["municipio", "year", "week", value_col_name]].copy()
    df.rename(columns={value_col_name: "value"}, inplace=True)
    df["municipio"] = df["municipio"].astype(str).str.strip()
    return df


def build_adjacency_list(geojson_file, max_distance_meters):
    gdf = gpd.read_file(geojson_file)
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

    gdf["name"] = gdf["NOME"].astype(str).str.strip().replace(name_corrections)
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: unary_union(geom) if geom.geom_type == "MultiPolygon" else geom
    )
    gdf = gdf.to_crs(epsg=31983)

    adjacency_list = {name: set() for name in gdf["name"]}
    for _, row1 in gdf.iterrows():
        name1 = row1["name"]
        poly1 = row1["geometry"]
        for _, row2 in gdf.iterrows():
            name2 = row2["name"]
            if name1 == name2:
                continue
            poly2 = row2["geometry"]
            if poly1.intersects(poly2) or poly1.distance(poly2) <= max_distance_meters:
                adjacency_list[name1].add(name2)
                adjacency_list[name2].add(name1)

    return {k: sorted(v) for k, v in adjacency_list.items()}


def add_population_and_idhm(data_df, municipio_name):
    pop_df = pd.read_csv(POP_FILE)
    pop_df.columns = [c.strip().lower() for c in pop_df.columns]
    pop_df["municipio"] = pop_df["municipio"].astype(str).str.strip()

    idhm_df = pd.read_csv(IDHM_FILE)
    idhm_df.columns = [c.strip().lower() for c in idhm_df.columns]
    idhm_df["municipio"] = idhm_df["municipio"].astype(str).str.strip()

    pop_val = pop_df.loc[
        pop_df["municipio"].str.upper() == municipio_name.upper(), "population"
    ].iloc[0]
    idhm_val = idhm_df.loc[
        idhm_df["municipio"].str.upper() == municipio_name.upper(), "idhm"
    ].iloc[0]

    data_df["POPULATION"] = pop_val
    data_df["IDHM"] = idhm_val
    return data_df


def build_base_dataframe():
    rio_cases_df = load_csv_data_single(CASES_FILE, municipio_info["name"], "cases")
    rio_temp_df = load_csv_data_single(TEMP_FILE, municipio_info["name"], "temperature")
    rio_hum_df = load_csv_data_single(HUMID_FILE, municipio_info["name"], "humidity")
    rio_rain_df = load_csv_data_single(RAIN_FILE, municipio_info["name"], "rainfall")

    data_df = (
        rio_cases_df.merge(rio_temp_df, on=["year", "week"])
        .merge(rio_hum_df, on=["year", "week"])
        .merge(rio_rain_df, on=["year", "week"])
    )
    data_df.rename(
        columns={
            "cases": "DENGUE_CASES",
            "temperature": "TEMP",
            "humidity": "HUMIDITY",
            "rainfall": "RAINFALL",
        },
        inplace=True,
    )

    adjacency_list = build_adjacency_list(GEOJSON_FILE, MAX_DISTANCE_METERS)
    if municipio_info["name"] not in adjacency_list:
        raise KeyError(f"{municipio_info['name']} not found in adjacency list")

    neighbors = adjacency_list[municipio_info["name"]]
    print("Neighbors for", municipio_info["name"], ":", neighbors)

    cases_all = load_csv_data_all(CASES_FILE, "cases")
    temp_all = load_csv_data_all(TEMP_FILE, "temperature")
    hum_all = load_csv_data_all(HUMID_FILE, "humidity")
    rain_all = load_csv_data_all(RAIN_FILE, "rainfall")

    cases_idx = (
        cases_all.groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    temp_idx = (
        temp_all.groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    hum_idx = (
        hum_all.groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    rain_idx = (
        rain_all.groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )

    cases_spatial, rain_spatial, temp_spatial, hum_spatial, n_neighbors = [], [], [], [], []

    for _, row in data_df.iterrows():
        y = int(row["year"])
        w = int(row["week"]) - 1
        if w <= 0:
            y -= 1
            w = 52

        neigh_vals_cases, neigh_vals_rain, neigh_vals_temp, neigh_vals_hum = [], [], [], []
        for n in neighbors:
            try:
                neigh_vals_cases.append(cases_idx.loc[(y, w, n), "value"])
            except KeyError:
                pass
            try:
                neigh_vals_rain.append(rain_idx.loc[(y, w, n), "value"])
            except KeyError:
                pass
            try:
                neigh_vals_temp.append(temp_idx.loc[(y, w, n), "value"])
            except KeyError:
                pass
            try:
                neigh_vals_hum.append(hum_idx.loc[(y, w, n), "value"])
            except KeyError:
                pass

        cases_spatial.append(np.mean(neigh_vals_cases) if neigh_vals_cases else np.nan)
        rain_spatial.append(np.mean(neigh_vals_rain) if neigh_vals_rain else np.nan)
        temp_spatial.append(np.mean(neigh_vals_temp) if neigh_vals_temp else np.nan)
        hum_spatial.append(np.mean(neigh_vals_hum) if neigh_vals_hum else np.nan)
        n_neighbors.append(len(neigh_vals_cases))

    data_df["CASES_SPATIAL_LAG1"] = cases_spatial
    data_df["RAINFALL_SPATIAL_LAG1"] = rain_spatial
    data_df["TEMP_SPATIAL_LAG1"] = temp_spatial
    data_df["HUMIDITY_SPATIAL_LAG1"] = hum_spatial
    data_df["NUM_NEIGHBORS_AVAILABLE"] = n_neighbors

    data_df["CASES_SPATIAL_LAG1"] = data_df["CASES_SPATIAL_LAG1"].fillna(
        data_df["DENGUE_CASES"].shift(1)
    )
    data_df["RAINFALL_SPATIAL_LAG1"] = data_df["RAINFALL_SPATIAL_LAG1"].fillna(
        data_df["RAINFALL"].shift(1)
    )
    data_df["TEMP_SPATIAL_LAG1"] = data_df["TEMP_SPATIAL_LAG1"].fillna(
        data_df["TEMP"].shift(1)
    )
    data_df["HUMIDITY_SPATIAL_LAG1"] = data_df["HUMIDITY_SPATIAL_LAG1"].fillna(
        data_df["HUMIDITY"].shift(1)
    )

    data_df = add_population_and_idhm(data_df, municipio_info["name"])
    data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].reset_index(
        drop=True
    )
    data_df = data_df.sort_values(["year", "week"]).reset_index(drop=True)

    return data_df


def add_features(df):
    df = df.copy()

    lag_feature_sources = [
        "DENGUE_CASES",
        "RAINFALL",
        "TEMP",
        "HUMIDITY",
        "RAINFALL_SPATIAL_LAG1",
        "TEMP_SPATIAL_LAG1",
        "HUMIDITY_SPATIAL_LAG1",
        "CASES_SPATIAL_LAG1",
    ]

    for col in lag_feature_sources:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_roll3"] = df[col].shift(1).rolling(3, min_periods=1).mean()

    decay = 0.8
    K = 10
    cases_series = df["DENGUE_CASES"].to_numpy()
    immunity = []
    for i in range(len(cases_series)):
        past_cases = sum(
            cases_series[i - k] * np.exp(-decay * k)
            for k in range(1, K + 1)
            if i - k >= 0
        )
        immunity.append(past_cases)
    df["IMMUNITY"] = immunity

    df["YEAR_INDEX"] = (
        (df["year"].astype(int) - df["year"].astype(int).min()) * 52 + df["week"].astype(int)
    )

    return df


def build_feature_columns(df):
    base_cols = [
        "POPULATION",
        "IDHM",
        "IMMUNITY",
        "NUM_NEIGHBORS_AVAILABLE",
        "YEAR_INDEX",
    ]
    lag_cols = [c for c in df.columns if "_lag" in c or "_roll" in c]
    return base_cols + sorted(lag_cols)


def evaluate_predictions(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    total_actual = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(total_actual, 1e-9))
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


def fit_xgboost_with_validation_calibration(X_train, y_train, X_valid, y_valid, X_test):
    model = XGBRegressor(
        n_estimators=623,
        learning_rate=0.02081058596075694,
        max_depth=3,
        subsample=0.8242253369150914,
        colsample_bytree=0.9918052007526491,
        random_state=42,
        objective="reg:squarederror",
    )

    y_train_log = np.log1p(y_train)
    model.fit(X_train, y_train_log)

    valid_pred_raw = np.expm1(model.predict(X_valid))
    test_pred_raw = np.expm1(model.predict(X_test))

    calib = LinearRegression()
    calib.fit(valid_pred_raw.reshape(-1, 1), y_valid)

    valid_pred = np.clip(calib.predict(valid_pred_raw.reshape(-1, 1)), 0, None)
    test_pred = np.clip(calib.predict(test_pred_raw.reshape(-1, 1)), 0, None)

    return valid_pred, test_pred


def main():
    base_df = build_base_dataframe()
    featured_df = add_features(base_df)

    horizon_scores = []
    all_metrics = []

    for horizon in HORIZONS:
        model_df = featured_df.copy()
        model_df[f"TARGET_{horizon}W_AHEAD"] = model_df["DENGUE_CASES"].shift(-horizon)
        model_df = model_df.dropna(subset=[f"TARGET_{horizon}W_AHEAD"]).copy()

        feature_cols = build_feature_columns(model_df)
        model_df = model_df.dropna(subset=feature_cols).copy()

        train_mask = model_df["year"] <= TRAIN_END_YEAR
        valid_mask = (model_df["year"] > TRAIN_END_YEAR) & (model_df["year"] <= VALID_END_YEAR)
        test_mask = (model_df["year"] >= TEST_START_YEAR) & (model_df["year"] <= TEST_END_YEAR)

        X_train = model_df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_train = model_df.loc[train_mask, f"TARGET_{horizon}W_AHEAD"].to_numpy(dtype=float)
        X_valid = model_df.loc[valid_mask, feature_cols].to_numpy(dtype=float)
        y_valid = model_df.loc[valid_mask, f"TARGET_{horizon}W_AHEAD"].to_numpy(dtype=float)
        X_test = model_df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_test = model_df.loc[test_mask, f"TARGET_{horizon}W_AHEAD"].to_numpy(dtype=float)

        if min(len(X_train), len(X_valid), len(X_test)) == 0:
            raise ValueError(f"Empty split for horizon {horizon}. Check year ranges.")

        valid_pred, test_pred = fit_xgboost_with_validation_calibration(
            X_train, y_train, X_valid, y_valid, X_test
        )

        valid_metrics = evaluate_predictions(y_valid, valid_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)

        horizon_scores.append(test_metrics["rmse"])
        all_metrics.append(
            {
                "horizon": horizon,
                "valid_r2": valid_metrics["r2"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_wape": valid_metrics["wape"],
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_wape": test_metrics["wape"],
            }
        )

        print(
            f"Horizon {horizon}: "
            f"valid_rmse={valid_metrics['rmse']:.2f}, "
            f"test_rmse={test_metrics['rmse']:.2f}, "
            f"test_mae={test_metrics['mae']:.2f}, "
            f"test_wape={test_metrics['wape']:.4f}, "
            f"test_r2={test_metrics['r2']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"Saved metrics to: {METRICS_FILE}")

    x_values = np.arange(1, len(horizon_scores) + 1)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(
        x_values,
        horizon_scores,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="XGBoost",
    )
    plt.xlabel("Horizon (weeks)", fontsize=10, fontweight="bold")
    plt.ylabel("Test RMSE", fontsize=10, fontweight="bold")
    plt.title("XGBoost: Test RMSE vs Forecast Horizon", fontsize=12, fontweight="bold")
    plt.xticks(x_values[::2])
    plt.tick_params(axis="both", which="both", length=6)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8, frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
