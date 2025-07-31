import os
import requests
import numpy as np
import pandas as pd
import io

# Output folder
output_dir = "data/wind"
os.makedirs(output_dir, exist_ok=True)

# Time range
START_YEAR = 2013
END_YEAR = 2023

# Regions defined as (name, lat_min, lat_max, lon_min, lon_max)
regions = [
    ("cal", 32.5, 35.0, -120.0, -116.0),
    ("texas", 29.0, 33.0, -101.0, -96.0),
    ("colorado", 37.0, 40.5, -106.0, -103.0),
    ("northeast", 38.5, 43.0, -78.5, -71.0)
]

# Grid resolution in degrees (e.g., 0.5 ~ 55 km, 0.25 ~ 28 km)
GRID_STEP = 0.5

# Wind parameters at 10m and 50m
WIND_PARAMS = "WS10M,WD10M,WS50M,WD50M"

def download_wind_point(region_name, index, lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": START_YEAR,
        "end": END_YEAR,
        "latitude": lat,
        "longitude": lon,
        "parameters": WIND_PARAMS,
        "format": "CSV",
        "community": "RE"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        # Read the CSV text from response, skipping header lines (12)
        raw_csv = response.text
        df = pd.read_csv(io.StringIO(raw_csv), skiprows=12)

        # Add lat and lon columns
        df.insert(0, "Longitude", lon)
        df.insert(0, "Latitude", lat)

        # Save modified CSV
        filename = f"{region_name}_{index}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        print(f"✅ Saved {filename} with lat/lon")
    except Exception as e:
        print(f"❌ Failed for {region_name} point {index} at ({lat}, {lon}): {e}")

# Loop through each region and generate grid points
for region_name, lat_min, lat_max, lon_min, lon_max in regions:
    lats = np.arange(lat_min, lat_max + GRID_STEP, GRID_STEP)
    lons = np.arange(lon_min, lon_max + GRID_STEP, GRID_STEP)
    index = 0
    for lat in lats:
        for lon in lons:
            download_wind_point(region_name, index, lat, lon)
            index += 1
