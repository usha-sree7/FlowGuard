"""Screen CWC hourly telemetry stations (National Water Data Portal) for a usable level + rain pair.

    python scripts/audit_cwc.py <level_csv> <rain_csv> [<rain_csv> ...]

Example (files from https://nwdp.nwic.gov.in, see docs/INDIAN_DATA_AUDIT.md):
    python scripts/audit_cwc.py ../data/raw/cwc_chindnar/rwl_tel_hr_cwc_006_2021_2025.csv \
                                ../data/raw/cwc_chindnar/rainfall_tel_hr_cwc_cg_2021_2025.csv

For every level station it reports:
  hours        number of hourly level readings
  stuck_frac   share of readings equal to the most common value (high = flat / stuck sensor or regulated pond)
  p01/med/p99  level percentiles [m RL]
  rain_km      distance to the nearest rain gauge in the given rain files (0 = same coordinates)
  rain_n       number of rain records at that gauge (the files store non-zero rain only)
  rain_gt100   rain records above 100 mm/h (physically implausible)
Writes nothing; prints a table sorted by distance to the nearest rain gauge.
"""
import sys

import numpy as np
import pandas as pd

LEVEL_COL = "River Water Level Telemetry Hourly (meter)"
RAIN_COL = "Telemetry Hourly Rainfall (mm)"


def read(path, value_col):
    df = pd.read_csv(path, usecols=["Station", "Latitude", "Longitude", value_col], low_memory=False)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["v"] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=["Latitude", "Longitude"])


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    level = read(sys.argv[1], LEVEL_COL)
    rain = pd.concat([read(p, RAIN_COL) for p in sys.argv[2:]])

    gauges = rain.groupby("Station").agg(lat=("Latitude", "first"), lon=("Longitude", "first"),
                                         rain_n=("v", "size"), rain_gt100=("v", lambda s: int((s > 100).sum())))
    rows = []
    for name, g in level.groupby("Station"):
        lat, lon = g["Latitude"].iloc[0], g["Longitude"].iloc[0]
        d = distance_km(lat, lon, gauges["lat"].to_numpy(), gauges["lon"].to_numpy())
        i = int(np.argmin(d))
        v = g["v"].dropna()
        rows.append({
            "level_station": name, "hours": len(v),
            "stuck_frac": round(v.value_counts(normalize=True).iloc[0], 3) if len(v) else 1.0,
            "p01": round(v.quantile(0.01), 2), "med": round(v.median(), 2), "p99": round(v.quantile(0.99), 2),
            "rain_gauge": gauges.index[i], "rain_km": round(float(d[i]), 1),
            "rain_n": int(gauges["rain_n"].iloc[i]), "rain_gt100": int(gauges["rain_gt100"].iloc[i]),
        })
    out = pd.DataFrame(rows).sort_values(["rain_km", "stuck_frac"])
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
