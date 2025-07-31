import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------- CONFIG ----------------------
regions = ["cal", "colorado", "texas", "northeast"]
output_root = "data/wind_daily"
os.makedirs(output_root, exist_ok=True)

for region in regions:
    print(f"\n🌎 Processing region: {region}")
    smoke_dir = f"data/smoke/{region}"
    wind_dir = f"data/wind/{region}"
    output_dir = os.path.join(output_root, region)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get valid PM2.5 dates
    valid_dates = set()
    for f in os.listdir(smoke_dir):
        if f.endswith(".csv"):
            date_str = f.replace(".csv", "")
            valid_dates.add(date_str)

    print(f"📅 Found {len(valid_dates)} PM2.5 days in {region}")

    # Step 2: Map {date: list of point dataframes}
    date_to_rows = {date: [] for date in valid_dates}

    # Step 3: Read wind point files
    wind_files = [f for f in os.listdir(wind_dir) if f.endswith(".csv")]

    for wf in tqdm(wind_files, desc=f"🌀 Reading wind files for {region}"):
        file_path = os.path.join(wind_dir, wf)

        try:
            # Read CSV normally, pandas will infer header
            df = pd.read_csv(file_path)

            # Filter needed columns and drop missing
            cols = ["Latitude", "Longitude", "YEAR", "MO", "DY", "WS10M", "WD10M", "WS50M", "WD50M"]
            df = df[cols].dropna()

            # Prepare date column string: YYYY_MM_DD
            df["date"] = df.apply(lambda r: f"{int(r.YEAR)}_{int(r.MO):02d}_{int(r.DY):02d}", axis=1)

            # Compute wind U/V components
            def wind_uv(ws, wd_deg):
                wd_rad = np.radians(wd_deg)
                u = -ws * np.sin(wd_rad)
                v = -ws * np.cos(wd_rad)
                return u, v

            u10, v10 = wind_uv(df["WS10M"], df["WD10M"])
            u50, v50 = wind_uv(df["WS50M"], df["WD50M"])

            df["wind_u10m"] = u10
            df["wind_v10m"] = v10
            df["wind_u50m"] = u50
            df["wind_v50m"] = v50

            # For each valid date, append the corresponding rows to the dict
            for date in valid_dates:
                day_df = df[df["date"] == date]
                if not day_df.empty:
                    date_to_rows[date].append(
                        day_df[[
                            "Latitude", "Longitude",
                            "wind_u10m", "wind_v10m",
                            "wind_u50m", "wind_v50m"
                        ]]
                    )

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            continue

    # Step 4: Save daily output files
    for date, row_list in tqdm(date_to_rows.items(), desc=f"💾 Writing daily files for {region}"):
        if row_list:
            df_day = pd.concat(row_list, ignore_index=True)
            df_day.insert(0, "date", date.replace("_", "-"))
            output_path = os.path.join(output_dir, f"{date}.csv")
            df_day.to_csv(output_path, index=False)
