import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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
GEOJSON_FILE = os.path.join(data_dir, "RJ.json")

LAGS = [1, 2, 3, 4, 6, 8, 12]
HORIZONS = range(1, 16)
MAX_DISTANCE_METERS = 5000

TRAIN_END_YEAR = 2010
VALID_END_YEAR = 2013
TEST_START_YEAR = 2014
TEST_END_YEAR = 2025

CALIBRATE_PREDICTIONS = True

PLOT_FILE = os.path.join(base_dir, "multi_model_horizon_rmse_plot.png")
METRICS_FILE = os.path.join(base_dir, "multi_model_horizon_metrics.csv")


# -----------------------------
# Loading helpers
# -----------------------------
def load_csv_data_single(file, municipio, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    value_col_name = value_col_name.lower()
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")

    df = df[df["municipio"].str.upper() == municipio.upper()]
    df = df.dropna(subset=["year", "week", value_col_name]).copy()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)

    return (
        df[["year", "week", value_col_name]]
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )


def load_csv_data_all(file, value_col_name):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    value_col_name = value_col_name.lower()
    if value_col_name not in df.columns:
        raise ValueError(f"{value_col_name} not found in {file} columns")

    df = df[["municipio", "year", "week", value_col_name]].copy()
    df.rename(columns={value_col_name: "value"}, inplace=True)
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["municipio", "year", "week", "value"]).copy()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def previous_iso_week(year, week):
    ts = pd.Timestamp.fromisocalendar(int(year), int(week), 1) - pd.Timedelta(weeks=1)
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


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


# -----------------------------
# Feature engineering
# -----------------------------
def build_base_dataframe():
    rio_cases_df = load_csv_data_single(CASES_FILE, municipio_info["name"], "cases")
    rio_temp_df = load_csv_data_single(TEMP_FILE, municipio_info["name"], "temperature")
    rio_hum_df = load_csv_data_single(HUMID_FILE, municipio_info["name"], "humidity")
    rio_rain_df = load_csv_data_single(RAIN_FILE, municipio_info["name"], "rainfall")

    data_df = (
        rio_cases_df.merge(rio_temp_df, on=["year", "week"], how="inner")
        .merge(rio_hum_df, on=["year", "week"], how="inner")
        .merge(rio_rain_df, on=["year", "week"], how="inner")
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

    cases_idx = (
        load_csv_data_all(CASES_FILE, "cases")
        .groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    temp_idx = (
        load_csv_data_all(TEMP_FILE, "temperature")
        .groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    hum_idx = (
        load_csv_data_all(HUMID_FILE, "humidity")
        .groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )
    rain_idx = (
        load_csv_data_all(RAIN_FILE, "rainfall")
        .groupby(["year", "week", "municipio"])["value"]
        .first()
        .reset_index()
        .set_index(["year", "week", "municipio"])
    )

    cases_spatial, rain_spatial, temp_spatial, hum_spatial, n_neighbors = [], [], [], [], []

    for _, row in data_df.iterrows():
        y_prev, w_prev = previous_iso_week(row["year"], row["week"])

        neigh_cases, neigh_rain, neigh_temp, neigh_hum = [], [], [], []
        for neighbor in neighbors:
            try:
                neigh_cases.append(cases_idx.loc[(y_prev, w_prev, neighbor), "value"])
            except KeyError:
                pass
            try:
                neigh_rain.append(rain_idx.loc[(y_prev, w_prev, neighbor), "value"])
            except KeyError:
                pass
            try:
                neigh_temp.append(temp_idx.loc[(y_prev, w_prev, neighbor), "value"])
            except KeyError:
                pass
            try:
                neigh_hum.append(hum_idx.loc[(y_prev, w_prev, neighbor), "value"])
            except KeyError:
                pass

        cases_spatial.append(np.mean(neigh_cases) if neigh_cases else np.nan)
        rain_spatial.append(np.mean(neigh_rain) if neigh_rain else np.nan)
        temp_spatial.append(np.mean(neigh_temp) if neigh_temp else np.nan)
        hum_spatial.append(np.mean(neigh_hum) if neigh_hum else np.nan)
        n_neighbors.append(len(neigh_cases))

    data_df["CASES_SPATIAL_LAG1"] = cases_spatial
    data_df["RAINFALL_SPATIAL_LAG1"] = rain_spatial
    data_df["TEMP_SPATIAL_LAG1"] = temp_spatial
    data_df["HUMIDITY_SPATIAL_LAG1"] = hum_spatial
    data_df["NUM_NEIGHBORS_AVAILABLE"] = n_neighbors

    # Missingness is recorded before fallback imputation.
    data_df["CASES_SPATIAL_MISSING"] = data_df["CASES_SPATIAL_LAG1"].isna().astype(int)
    data_df["RAINFALL_SPATIAL_MISSING"] = data_df["RAINFALL_SPATIAL_LAG1"].isna().astype(int)
    data_df["TEMP_SPATIAL_MISSING"] = data_df["TEMP_SPATIAL_LAG1"].isna().astype(int)
    data_df["HUMIDITY_SPATIAL_MISSING"] = data_df["HUMIDITY_SPATIAL_LAG1"].isna().astype(int)

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

    data_df = data_df[(data_df["year"] >= 2010) & (data_df["year"] <= 2025)].copy()
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
    k_max = 10
    cases = df["DENGUE_CASES"].to_numpy()
    df["IMMUNITY"] = [
        sum(cases[i - k] * np.exp(-decay * k) for k in range(1, k_max + 1) if i - k >= 0)
        for i in range(len(cases))
    ]

    df["YEAR_INDEX"] = (
        (df["year"].astype(int) - df["year"].astype(int).min()) * 52
        + df["week"].astype(int)
    )

    week_float = df["week"].astype(float)
    df["WEEK_SIN"] = np.sin(2 * np.pi * week_float / 52.0)
    df["WEEK_COS"] = np.cos(2 * np.pi * week_float / 52.0)

    return df


def build_feature_columns(df):
    base_cols = [
        "IMMUNITY",
        "NUM_NEIGHBORS_AVAILABLE",
        "YEAR_INDEX",
        "WEEK_SIN",
        "WEEK_COS",
        "CASES_SPATIAL_MISSING",
        "RAINFALL_SPATIAL_MISSING",
        "TEMP_SPATIAL_MISSING",
        "HUMIDITY_SPATIAL_MISSING",
    ]
    lag_cols = [c for c in df.columns if "_lag" in c or "_roll" in c]
    return base_cols + sorted(lag_cols)


# -----------------------------
# Modeling
# -----------------------------
def evaluate_predictions(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    total_actual = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(total_actual, 1e-9))
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


def make_model(name):
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=623,
            learning_rate=0.02081058596075694,
            max_depth=3,
            subsample=0.8242253369150914,
            colsample_bytree=0.9918052007526491,
            random_state=42,
            objective="reg:squarederror",
        )
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=645,
            max_depth=9,
            random_state=42,
        )
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=639,
            learning_rate=0.06053197206271428,
            subsample=0.8013654487520273,
            random_state=42,
        )
    if name == "Ridge":
        return Ridge(alpha=2.4304352434763596)
    if name == "Linear":
        return LinearRegression()
    raise ValueError(f"Unknown model: {name}")


def fit_and_predict(model_name, X_train, y_train, X_valid, y_valid, X_test):
    model = make_model(model_name)

    if model_name in {"Linear", "Ridge"}:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_valid_fit = scaler.transform(X_valid)
        X_test_fit = scaler.transform(X_test)
    else:
        X_train_fit = X_train
        X_valid_fit = X_valid
        X_test_fit = X_test

    if model_name == "XGBoost":
        model.fit(X_train_fit, np.log1p(y_train))
        valid_pred = np.expm1(model.predict(X_valid_fit))
        test_pred = np.expm1(model.predict(X_test_fit))
    else:
        model.fit(X_train_fit, y_train)
        valid_pred = model.predict(X_valid_fit)
        test_pred = model.predict(X_test_fit)

    valid_pred = np.clip(valid_pred, 0, None)
    test_pred = np.clip(test_pred, 0, None)

    if CALIBRATE_PREDICTIONS:
        calibrator = LinearRegression()
        calibrator.fit(valid_pred.reshape(-1, 1), y_valid)
        valid_pred = np.clip(calibrator.predict(valid_pred.reshape(-1, 1)), 0, None)
        test_pred = np.clip(calibrator.predict(test_pred.reshape(-1, 1)), 0, None)

    return valid_pred, test_pred


def main():
    base_df = build_base_dataframe()
    featured_df = add_features(base_df)

    model_names = ["XGBoost", "Random Forest", "Gradient Boosting", "Ridge", "Linear"]
    model_scores = {name: [] for name in model_names}
    model_metrics = []

    for horizon in HORIZONS:
        target_col = f"TARGET_{horizon}W_AHEAD"
        model_df = featured_df.copy()
        model_df[target_col] = model_df["DENGUE_CASES"].shift(-horizon)
        model_df = model_df.dropna(subset=[target_col]).copy()

        feature_cols = build_feature_columns(model_df)
        model_df = model_df.dropna(subset=feature_cols).copy()

        train_mask = model_df["year"] <= TRAIN_END_YEAR
        valid_mask = (model_df["year"] > TRAIN_END_YEAR) & (model_df["year"] <= VALID_END_YEAR)
        test_mask = (model_df["year"] >= TEST_START_YEAR) & (model_df["year"] <= TEST_END_YEAR)

        X_train = model_df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_train = model_df.loc[train_mask, target_col].to_numpy(dtype=float)
        X_valid = model_df.loc[valid_mask, feature_cols].to_numpy(dtype=float)
        y_valid = model_df.loc[valid_mask, target_col].to_numpy(dtype=float)
        X_test = model_df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_test = model_df.loc[test_mask, target_col].to_numpy(dtype=float)

        if min(len(X_train), len(X_valid), len(X_test)) == 0:
            raise ValueError(f"Empty split for horizon {horizon}. Check year ranges.")

        for model_name in model_names:
            valid_pred, test_pred = fit_and_predict(
                model_name, X_train, y_train, X_valid, y_valid, X_test
            )
            valid_metrics = evaluate_predictions(y_valid, valid_pred)
            test_metrics = evaluate_predictions(y_test, test_pred)

            model_scores[model_name].append(test_metrics["rmse"])
            model_metrics.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "calibrated": CALIBRATE_PREDICTIONS,
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
                f"Horizon {horizon}, {model_name}: "
                f"test_rmse={test_metrics['rmse']:.2f}, "
                f"test_mae={test_metrics['mae']:.2f}, "
                f"test_r2={test_metrics['r2']:.4f}, "
                f"test_wape={test_metrics['wape']:.4f}"
            )

    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"Saved metrics to: {METRICS_FILE}")

    x_values = np.arange(1, len(model_scores["XGBoost"]) + 1)
    plt.figure(figsize=(6, 4), dpi=300)

    for model_name, scores in model_scores.items():
        plt.plot(
            x_values,
            scores,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=model_name,
        )

    plt.xlabel("Horizon (weeks)", fontsize=10, fontweight="bold")
    plt.ylabel("Test RMSE", fontsize=10, fontweight="bold")
    plt.title("Model Comparison: Test RMSE vs Forecast Horizon", fontsize=12, fontweight="bold")
    plt.xticks(x_values[::2])
    plt.tick_params(axis="both", which="both", length=6)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=7, frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
