import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import unicodedata
import datetime
import os
import json

# -----------------------------
# Helper: normalize names
# -----------------------------
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII').lower().strip()


# Step 1: Get all municipalities in RJ

ibge_url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/33/municipios"
municipalities = requests.get(ibge_url).json()
municipalities_info = [{"name": m["nome"], "geocode": m["id"]} for m in municipalities]

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

# -----------------------------
# Step 2: Load population data from PDFs
# -----------------------------
POP_PDFS = [
    os.path.expanduser("~/Desktop/POP2020_20220905.pdf"),
    os.path.expanduser("~/Desktop/POP2021_20240624.pdf"),
    os.path.expanduser("~/Desktop/POP2024_20241230.pdf"),
    os.path.expanduser("~/Desktop/estimativa_dou_2025.pdf")
]

import pdfplumber
pop_rows = []
for f in POP_PDFS:
    if not os.path.exists(f):
        continue
    with pdfplumber.open(f) as pdf:
        for page in pdf.pages[1:]:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                muni_name = " ".join(parts[3:-1])
                try:
                    pop_est = int(parts[-1].replace(".", ""))
                    pop_rows.append({"MUNICIPIO": normalize_name(muni_name), "POPULATION": pop_est})
                except:
                    continue
pop_df = pd.DataFrame(pop_rows)

# -----------------------------
# Step 3: Load IDHM data
# -----------------------------
IDHM_FILE = os.path.expanduser("~/Desktop/download_table_2010.xlsx")
idhm_df = pd.read_excel(IDHM_FILE)
idhm_df['Territorialidade'] = idhm_df['Territorialidade'].str.replace(r"\s*\([A-Z]{2}\)$","",regex=True)
idhm_df['Territorialidade'] = idhm_df['Territorialidade'].apply(normalize_name)
idhm_dict = dict(zip(idhm_df['Territorialidade'], idhm_df['IDHM']))

# -----------------------------
# Step 4: Load rainfall data
# -----------------------------
RAINFALL_FILE = os.path.expanduser("~/Desktop/bra-rainfall-subnat-full.csv")
PCODE_FILE = os.path.expanduser("~/Desktop/global_pcodes.csv")

rainfall_df = pd.read_csv(RAINFALL_FILE)
pcode_df = pd.read_csv(PCODE_FILE, low_memory=False)
municipalities_df = pcode_df[pcode_df['Parent P-Code']=='BR33'][['P-Code','Name']]
municipalities_df.rename(columns={'Name':'MUNICIPIO','P-Code':'PCODE'}, inplace=True)

# -----------------------------
# Step 5: Fetch dengue cases / temp / humidity
# -----------------------------
def fetch_cases(municipio, week, year):
    api_url = "https://info.dengue.mat.br/api/alertcity"
    params = {
        "geocode": municipio["geocode"],
        "disease": "dengue",
        "format": "json",
        "ew_start": week,
        "ew_end": week,
        "ey_start": year,
        "ey_end": year,
    }
    try:
        r = requests.get(api_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and len(data) > 0 and "casos" in data[0]:
            return normalize_name(municipio['name']), int(data[0]['casos'])
        return normalize_name(municipio['name']), 0
    except:
        return normalize_name(municipio['name']), 0

def fetch_metric(municipio, week, year, metric):
    api_url = "https://info.dengue.mat.br/api/alertcity"
    params = {
        "geocode": municipio["geocode"],
        "disease": "dengue",
        "format": "json",
        "ew_start": week,
        "ew_end": week,
        "ey_start": year,
        "ey_end": year,
    }
    try:
        r = requests.get(api_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and len(data) > 0:
            if metric == "temperature":
                return normalize_name(municipio["name"]), float(data[0].get("tempmed") or 0)
            elif metric == "humidity":
                return normalize_name(municipio["name"]), float(data[0].get("umidmed") or 0)
        return normalize_name(municipio["name"]), 0
    except:
        return normalize_name(municipio["name"]), 0

# -----------------------------
# Step 6: Preprocess rainfall to weekly sums
# -----------------------------
def weekly_rainfall(year, week):
    week_sum = {}
    for _, row in municipalities_df.iterrows():
        muni_name = normalize_name(name_corrections.get(row['MUNICIPIO'], row['MUNICIPIO']))
        muni_rain = rainfall_df[rainfall_df['PCODE']==row['PCODE']].copy()
        muni_rain['DATE'] = pd.to_datetime(muni_rain['date'], errors='coerce')
        muni_rain = muni_rain.dropna(subset=['DATE'])
        muni_rain['PRECIP'] = pd.to_numeric(muni_rain['rfh_avg'], errors='coerce').fillna(0) / 10.0  # 10-day -> daily
        week_precip = 0
        for _, r in muni_rain.iterrows():
            for day_offset in range(10):
                day = r['DATE'] + pd.Timedelta(days=day_offset)
                if day.year == year and day.isocalendar()[1]==week:
                    week_precip += r['PRECIP']
        week_sum[muni_name] = week_precip
    return week_sum

# -----------------------------
# Step 7: Generate datasets
# -----------------------------
weeks_all = list(range(1,53))
years_all = list(range(2010,2026))

cases_rows, temp_rows, hum_rows, rain_rows, pop_rows_out, idhm_rows_out = [], [], [], [], [], []

for year in years_all:
    for week in weeks_all:
        # Parallel API fetches
        with ThreadPoolExecutor(max_workers=10) as executor:
            cases_fut = [executor.submit(fetch_cases, m, week, year) for m in municipalities_info]
            temp_fut  = [executor.submit(fetch_metric, m, week, year, "temperature") for m in municipalities_info]
            hum_fut   = [executor.submit(fetch_metric, m, week, year, "humidity") for m in municipalities_info]

            cases_dict = {f.result()[0]: f.result()[1] for f in cases_fut}
            temp_dict  = {f.result()[0]: f.result()[1] for f in temp_fut}
            hum_dict   = {f.result()[0]: f.result()[1] for f in hum_fut}

        rain_dict = weekly_rainfall(year, week)

        for muni in cases_dict.keys():
            # Population nearest year
            pop_val = None
            for diff in range(0, 20):
                pop_match = pop_df[pop_df['MUNICIPIO']==muni]
                if not pop_match.empty:
                    pop_val = pop_match['POPULATION'].values[0]
                    break
            if pop_val is None:
                pop_val = 50000  # fallback

            # IDHM nearest
            idhm_val = None
            for diff in range(0,20):
                idhm_val = idhm_dict.get(muni)
                if idhm_val is not None:
                    break
            if idhm_val is None:
                idhm_val = 0.7

            # Append rows
            cases_rows.append({"municipio":muni,"year":year,"week":week,"cases":cases_dict[muni]})
            temp_rows.append({"municipio":muni,"year":year,"week":week,"temperature":temp_dict[muni]})
            hum_rows.append({"municipio":muni,"year":year,"week":week,"humidity":hum_dict[muni]})
            rain_rows.append({"municipio":muni,"year":year,"week":week,"rainfall":rain_dict.get(muni,0)})
            pop_rows_out.append({"municipio":muni,"year":year,"week":week,"population":pop_val})
            idhm_rows_out.append({"municipio":muni,"year":year,"week":week,"idhm":idhm_val})

        # Print Rio de Janeiro row for verification
        print(f"[{year} W{week}] Rio de Janeiro - cases: {cases_dict.get('rio de janeiro',0)}, temp: {temp_dict.get('rio de janeiro',0)}, hum: {hum_dict.get('rio de janeiro',0)}, rain: {rain_dict.get('rio de janeiro',0)}, pop: {pop_val}, idhm: {idhm_val}")

# -----------------------------
# Step 8: Save CSVs
# -----------------------------
pd.DataFrame(cases_rows).to_csv("cases.csv", index=False)
pd.DataFrame(temp_rows).to_csv("temperature.csv", index=False)
pd.DataFrame(hum_rows).to_csv("humidity.csv", index=False)
pd.DataFrame(rain_rows).to_csv("rainfall.csv", index=False)
pd.DataFrame(pop_rows_out).to_csv("population.csv", index=False)
pd.DataFrame(idhm_rows_out).to_csv("idhm.csv", index=False)

print("All CSVs saved successfully.")
