import requests
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Define municipality
# -----------------------------
municipio = {"name": "Rio de Janeiro", "geocode": "3304557"}
years = range(2020, 2024)
weeks = range(1, 53)

# -----------------------------
# Function to fetch dengue cases
# -----------------------------
def fetch_cases_for_week(week, year):
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
        if data and "casos" in data[0]:
            return week + (year-2020)*52, int(data[0]["casos"])
    except:
        pass
    return week + (year-2020)*52, 0

# -----------------------------
# Fetch all weeks in parallel
# -----------------------------
all_requests = [(w, y) for y in years for w in weeks]
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_cases_for_week, w, y) for w, y in all_requests]
    for future in futures:
        results.append(future.result())

# -----------------------------
# Prepare data
# -----------------------------
results.sort(key=lambda x: x[0])  # sort by week number
x_vals = np.array([r[0] for r in results])
y_cases = np.array([r[1] for r in results])

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(15,6))
plt.plot(x_vals, y_cases, 'ro-', label='Dengue Cases')
plt.xlabel('Week Number (since 2020)')
plt.ylabel('Number of Cases')
plt.title('Dengue Cases in Rio de Janeiro (2020–2023)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
