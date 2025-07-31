import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.spatial import cKDTree
import tempfile
import shutil
import gc
import concurrent.futures
from collections import defaultdict

# -------------------- CONFIG --------------------
DATA_PATH = "data/processed"
OUTPUT_PATH = "data/processed/horizon_datasetsNEW"
REGIONS = ["cal", "texas", "northeast", "colorado"]
HORIZONS = [1, 3, 7, 14]
SPATIAL_RADIUS_KM = 50
BATCH_SIZE = 60

MAX_SAMPLES_PER_REGION_PER_HORIZON = 6000
MAX_SAMPLES_PER_REGION_PER_YEAR = 900
MAX_SAMPLES_PER_REGION_PER_MONTH = 80

PLUME_CACHE = {}

# ------------------ UTILITIES -------------------
def parse_date(s): return datetime.strptime(s, "%Y_%m_%d")
def format_date(dt): return dt.strftime("%Y_%m_%d")

def gaussian_plume_estimate(source_coords, point_coords, u, v, Q=1000, H=10, sigma_y=50, sigma_z=20):
    dx = point_coords[0] - source_coords[0]
    dy = point_coords[1] - source_coords[1]
    wind_mag = np.sqrt(u**2 + v**2) + 1e-6
    wind_dir_x = u / wind_mag
    wind_dir_y = v / wind_mag
    x = dx * wind_dir_x + dy * wind_dir_y
    y = -dx * wind_dir_y + dy * wind_dir_x
    if x <= 0:
        return 0.0
    exponent = -0.5 * (y / sigma_y) ** 2
    vertical = np.exp(-0.5 * (H / sigma_z) ** 2)
    denom = 2 * np.pi * wind_mag * sigma_y * sigma_z
    C = (Q / denom) * np.exp(exponent) * vertical
    return round(C, 5)

def compute_plume(lat, lon, u10, v10, u50, v50):
    u = round((u10 + u50) / 2, 5)
    v = round((v10 + v50) / 2, 5)
    key = (round(lat, 5), round(lon, 5), u, v)
    if key not in PLUME_CACHE:
        PLUME_CACHE[key] = gaussian_plume_estimate((lon, lat), (lon, lat), u, v)
    return PLUME_CACHE[key]

# ------------------ DATA LOAD -------------------
def load_all_regions():
    all_dfs = []
    for region in REGIONS:
        file = os.path.join(DATA_PATH, f"{region}_dataset.csv")
        if not os.path.exists(file):
            print(f"[WARNING] Missing dataset: {file}")
            continue
        df = pd.read_csv(file)
        df["region"] = region
        df["date"] = df["date"].apply(parse_date)
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No data loaded.")
    return pd.concat(all_dfs, ignore_index=True)

def save_date_dfs_to_temp(date_to_df):
    temp_dir = tempfile.mkdtemp(prefix="date_dfs_")
    date_files = {}
    for date, df in list(date_to_df.items()):
        path = os.path.join(temp_dir, f"{format_date(date)}.pkl")
        df.to_pickle(path)
        date_files[date] = path
        del date_to_df[date]
    return temp_dir, date_files

def prepare_date_data(date, date_files):
    data = {"today": date_files[date]}
    for h in HORIZONS:
        f = date + timedelta(days=h)
        if f in date_files:
            data[h] = date_files[f]
    return date, data

# ------------------ PROCESSING -------------------
def process_one_date(args):
    date, data_files = args
    today_df = pd.read_pickle(data_files["today"])
    horizons_df = {h: pd.read_pickle(data_files[h]) if h in data_files else None for h in HORIZONS}

    lat_rad = np.radians(today_df["lat"].values)
    scale_lat = 111.0
    scale_lon = 111.0 * np.cos(lat_rad)
    pts_xy = np.vstack([today_df["lat"] * scale_lat, today_df["lon"] * scale_lon]).T
    tree = cKDTree(pts_xy)

    partial_results = {h: [] for h in HORIZONS}

    for idx, row in today_df.iterrows():
        center_lat, center_lon = row["lat"], row["lon"]
        center_xy = np.array([
            center_lat * 111.0,
            center_lon * 111.0 * np.cos(np.radians(center_lat))
        ])
        neighbor_idxs = tree.query_ball_point(center_xy, SPATIAL_RADIUS_KM)
        if not neighbor_idxs:
            continue

        context = []
        for n_idx in neighbor_idxs:
            n = today_df.iloc[n_idx]
            u = round((n["wind_u10m"] + n["wind_u50m"]) / 2, 5)
            v = round((n["wind_v10m"] + n["wind_v50m"]) / 2, 5)
            src = (round(n["lon"], 5), round(n["lat"], 5))
            tgt = (round(center_lon, 5), round(center_lat, 5))
            key = (*src, *tgt, u, v)
            if key not in PLUME_CACHE:
                PLUME_CACHE[key] = gaussian_plume_estimate(src, tgt, u, v)
            plume_val = PLUME_CACHE[key]
            context.append([
                n["lat"] - center_lat,
                n["lon"] - center_lon,
                n["wind_u10m"], n["wind_v10m"],
                n["wind_u50m"], n["wind_v50m"],
                n["elevation"], n["nlcd"],
                n["pm25_target"], plume_val
            ])

        for h in HORIZONS:
            df_fut = horizons_df[h]
            if df_fut is None:
                continue
            lat_r = round(center_lat, 5)
            lon_r = round(center_lon, 5)
            match = df_fut[(df_fut["lat_round5"] == lat_r) & (df_fut["lon_round5"] == lon_r)]
            if match.empty:
                continue
            target_pm25 = match.iloc[0]["pm25_target"]
            partial_results[h].append({
                "center_lat": center_lat,
                "center_lon": center_lon,
                "date": format_date(date),
                "region": row["region"],
                "target": target_pm25,
                "current_pm25": row["pm25_target"],
                "plume_pred": row["plume_pred"],
                "wind_u10m": row["wind_u10m"],
                "wind_v10m": row["wind_v10m"],
                "wind_u50m": row["wind_u50m"],
                "wind_v50m": row["wind_v50m"],
                "elevation": row["elevation"],
                "nlcd": row["nlcd"],
                "context": context
            })

    return partial_results

# ------------------ FLUSHING -------------------
def flush_to_csv(datasets, overwrite=False):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for h in HORIZONS:
        csv_path = os.path.join(OUTPUT_PATH, f"dataset_h{h}.csv")
        df = pd.DataFrame(datasets[h])
        if df.empty:
            continue
        mode = 'w' if overwrite or not os.path.exists(csv_path) else 'a'
        df.to_csv(csv_path, mode=mode, header=(mode == 'w'), index=False)
        datasets[h] = []

# ------------------ MAIN PIPELINE -------------------
def build_datasets_parallel(df_all):
    df_all = df_all.dropna(subset=[
        "lat", "lon", "elevation", "nlcd",
        "wind_u10m", "wind_v10m", "wind_u50m", "wind_v50m", "pm25_target"
    ]).reset_index(drop=True)

    print("Precomputing plume...")
    df_all["plume_pred"] = df_all.apply(lambda r: compute_plume(
        r["lat"], r["lon"], r["wind_u10m"], r["wind_v10m"], r["wind_u50m"], r["wind_v50m"]
    ), axis=1)

    grouped = df_all.groupby("date")
    date_to_df = {}
    for date, df in grouped:
        df = df.copy()
        df["lat_round5"] = df["lat"].round(5)
        df["lon_round5"] = df["lon"].round(5)
        date_to_df[date] = df.reset_index(drop=True)

    print("Saving date-wise data...")
    temp_dir, date_files = save_date_dfs_to_temp(date_to_df)
    date_data_list = [prepare_date_data(d, date_files) for d in sorted(date_files.keys())]

    datasets = {h: [] for h in HORIZONS}
    samples_per_region_horizon = {h: defaultdict(int) for h in HORIZONS}
    samples_per_year = {h: defaultdict(lambda: defaultdict(int)) for h in HORIZONS}
    samples_per_month = {h: defaultdict(lambda: defaultdict(int)) for h in HORIZONS}

    processed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for results in tqdm(executor.map(process_one_date, date_data_list), total=len(date_data_list)):
            for h in HORIZONS:
                for sample in results[h]:
                    region = sample["region"]
                    year = int(sample["date"][:4])
                    month = int(sample["date"][5:7])

                    if samples_per_region_horizon[h][region] >= MAX_SAMPLES_PER_REGION_PER_HORIZON:
                        continue
                    if samples_per_year[h][region][year] >= MAX_SAMPLES_PER_REGION_PER_YEAR:
                        continue
                    if samples_per_month[h][region][(year, month)] >= MAX_SAMPLES_PER_REGION_PER_MONTH:
                        continue

                    datasets[h].append(sample)
                    samples_per_region_horizon[h][region] += 1
                    samples_per_year[h][region][year] += 1
                    samples_per_month[h][region][(year, month)] += 1

            processed += 1
            if processed % BATCH_SIZE == 0:
                print(f"Flushing batch at {processed} dates...")
                flush_to_csv(datasets, overwrite=(processed == BATCH_SIZE))
                gc.collect()

    flush_to_csv(datasets)
    print("Saving final pickle files...")
    for h in HORIZONS:
        path = os.path.join(OUTPUT_PATH, f"dataset_h{h}.csv")
        df = pd.read_csv(path)
        df.to_pickle(path.replace(".csv", ".pkl"))
    shutil.rmtree(temp_dir)
    print("Done.")

# ------------------ ENTRY POINT -------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print("Loading regional data...")
    df_all = load_all_regions()
    print(f"Total records: {len(df_all)}")
    build_datasets_parallel(df_all)
