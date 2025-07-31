import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.spatial import cKDTree
import pickle
import concurrent.futures
import gc
import tempfile
import shutil
from collections import defaultdict

# Manual cache for plume values
PLUME_CACHE = {}

# Gaussian plume estimate function
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

# CONFIG
DATA_PATH = "data/processed"
OUTPUT_PATH = "data/processed/horizon_datasets"
SPATIAL_RADIUS_KM = 50
HORIZONS = [1, 3, 7, 14]
BATCH_SIZE = 60  # Number of dates processed before flushing to disk
MAX_SAMPLES_PER_REGION_PER_HORIZON = 5000  # 1000 samples per region per horizon, total 4000 per horizon

def parse_date(s):
    return datetime.strptime(s, "%Y_%m_%d")

def format_date(dt):
    return dt.strftime("%Y_%m_%d")

def compute_plume(lat, lon, u10, v10, u50, v50):
    u = round((u10 + u50) / 2, 5)
    v = round((v10 + v50) / 2, 5)
    key = (round(lat, 5), round(lon, 5), u, v)
    if key not in PLUME_CACHE:
        PLUME_CACHE[key] = gaussian_plume_estimate((lon, lat), (lon, lat), u, v)
    return PLUME_CACHE[key]

def load_all_regions():
    all_dfs = []
    for region in ["cal", "texas", "northeast", "colorado"]:
        file = os.path.join(DATA_PATH, f"{region}_dataset.csv")
        if not os.path.exists(file):
            print(f"[WARNING] Dataset CSV not found for region {region} at {file}, skipping.")
            continue
        df = pd.read_csv(file)
        df["region"] = region
        df["date"] = df["date"].apply(parse_date)
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No data loaded from any region!")
    return pd.concat(all_dfs, ignore_index=True)

def clean_context(row):
    return [
        [round(float(x), 3) if isinstance(x, (float, int, np.floating, np.integer)) else int(x)
         for x in context_row]
        for context_row in row
    ]

def process_one_date(args):
    """
    args = (date, data_files_dict)
    data_files_dict = {
        "today": path_to_pickle,
        1: path_to_pickle_for_date_plus_1_day,
        3: ...
        7: ...
        14: ...
    }
    """
    date, data_files = args
    import pandas as pd

    # Load only required dataframes from disk
    today_df = pd.read_pickle(data_files["today"])
    horizons_df = {}
    for h in HORIZONS:
        if h in data_files:
            horizons_df[h] = pd.read_pickle(data_files[h])
        else:
            horizons_df[h] = None

    lat_rad = np.radians(today_df["lat"].values)
    scale_lat = 111.0
    scale_lon = 111.0 * np.cos(lat_rad)
    pts_xy = np.vstack([
        today_df["lat"].values * scale_lat,
        today_df["lon"].values * scale_lon
    ]).T
    tree = cKDTree(pts_xy)
    partial_results = {h: [] for h in HORIZONS}

    for idx, row in today_df.iterrows():
        center_lat, center_lon = row["lat"], row["lon"]
        center_x = center_lat * 111.0
        center_y = center_lon * 111.0 * np.cos(np.radians(center_lat))
        center_xy = np.array([center_x, center_y])

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
            future_df = horizons_df[h]
            if future_df is None:
                continue
            lat_r = round(center_lat, 5)
            lon_r = round(center_lon, 5)
            match = future_df[
                (future_df["lat_round5"] == lat_r) &
                (future_df["lon_round5"] == lon_r)
            ]
            if match.empty:
                continue
            target_pm25 = match.iloc[0]["pm25_target"]

            sample = {
                "center_lat": center_lat,
                "center_lon": center_lon,
                "date": format_date(date),
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
            }
            partial_results[h].append(sample)

    del today_df, lat_rad, pts_xy, tree
    gc.collect()

    return partial_results

def save_date_dfs_to_temp(date_to_df):
    temp_dir = tempfile.mkdtemp(prefix="date_dfs_")
    date_files = {}
    for date, df in list(date_to_df.items()):  # list() to avoid "dictionary changed size" error
        path = os.path.join(temp_dir, f"{format_date(date)}.pkl")
        df.to_pickle(path)
        date_files[date] = path
        del date_to_df[date]  # Delete the DataFrame from the dictionary to free memory
    return temp_dir, date_files


def prepare_date_data(date, date_files):
    """
    For given date, return (date, dict_of_paths) where dict_of_paths contains keys
    'today' and horizon days with values as paths to pickle files.
    """
    data = {"today": date_files[date]}
    for h in HORIZONS:
        future_date = date + timedelta(days=h)
        if future_date in date_files:
            data[h] = date_files[future_date]
    return date, data


def limit_samples(datasets):
    """
    Limit datasets to max 1000 samples per region per horizon,
    i.e. 4000 samples per horizon total.
    Returns new limited datasets dict.
    """
    limited = {h: [] for h in HORIZONS}
    counters = {h: defaultdict(int) for h in HORIZONS}

    for h in HORIZONS:
        # Shuffle to randomize selection
        np.random.shuffle(datasets[h])

        for sample in datasets[h]:
            region = sample.get("region", "unknown")
            if counters[h][region] < MAX_SAMPLES_PER_REGION_PER_HORIZON:
                limited[h].append(sample)
                counters[h][region] += 1

            # Stop early if limit reached for all regions
            if all(counters[h][r] >= MAX_SAMPLES_PER_REGION_PER_HORIZON for r in ["cal", "texas", "northeast", "colorado"]):
                break

    return limited
MAX_SAMPLES_PER_REGION_PER_HORIZON = 1000  # max per region per horizon
MAX_TOTAL_SAMPLES_PER_HORIZON = MAX_SAMPLES_PER_REGION_PER_HORIZON * 4  # 4 regions total

def build_datasets_parallel(df):
    df = df.dropna(subset=[
        "lat", "lon", "elevation", "nlcd",
        "wind_u10m", "wind_v10m", "wind_u50m", "wind_v50m", "pm25_target"
    ]).reset_index(drop=True)

    print("Computing plume_pred for each source point (source->self)...")
    df["plume_pred"] = df.apply(lambda row: compute_plume(
        row["lat"], row["lon"],
        row["wind_u10m"], row["wind_v10m"],
        row["wind_u50m"], row["wind_v50m"]
    ), axis=1)

    grouped = df.groupby("date")
    all_dates = sorted(grouped.groups.keys())
    date_to_df = {}
    for d in all_dates:
        date_df = grouped.get_group(d).copy()
        date_df["lat_round5"] = date_df["lat"].round(5)
        date_df["lon_round5"] = date_df["lon"].round(5)
        date_df = date_df.reset_index(drop=True)
        date_to_df[d] = date_df

    print("Saving per-date DataFrames to temp files...")
    temp_dir, date_files = save_date_dfs_to_temp(date_to_df)

    # Initialize counters
    samples_per_region_horizon = {h: defaultdict(int) for h in HORIZONS}
    samples_per_year = defaultdict(int)
    samples_per_month = defaultdict(int)

    datasets = {h: [] for h in HORIZONS}

    print("Building datasets for horizons in parallel...")
    processed_dates = 0

    date_data_list = [prepare_date_data(date, date_files) for date in all_dates]

    def enough_samples():
        # Check if for all horizons the total samples reached max allowed
        return all(
            sum(samples_per_region_horizon[h].values()) >= MAX_TOTAL_SAMPLES_PER_HORIZON
            for h in HORIZONS
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for partial_results in tqdm(executor.map(process_one_date, date_data_list), total=len(date_data_list)):
            # Early break if reached enough samples
            if enough_samples():
                print("Reached max total samples for all horizons, stopping early.")
                break

            for h in HORIZONS:
                # For each sample in partial_results[h], filter based on quotas
                for sample in partial_results[h]:
                    region = sample.get("region", None)
                    if region is None:
                        continue

                    year = int(sample["date"][:4])
                    month = int(sample["date"][5:7])

                    # Check if region quota reached
                    if samples_per_region_horizon[h][region] >= MAX_SAMPLES_PER_REGION_PER_HORIZON:
                        continue
                    # Optionally limit samples per year/month, e.g.:
                    if samples_per_year[year] >= 400:  # arbitrary yearly cap, adjust as needed
                        continue
                    if samples_per_month[(year, month)] >= 35:  # arbitrary monthly cap, adjust as needed
                        continue

                    # Add sample
                    datasets[h].append(sample)
                    samples_per_region_horizon[h][region] += 1
                    samples_per_year[year] += 1
                    samples_per_month[(year, month)] += 1

            processed_dates += 1
            if processed_dates % BATCH_SIZE == 0 or processed_dates == len(date_data_list):
                print(f"Flushing batch at date count {processed_dates} to CSV and clearing memory...")
                flush_to_csv(datasets, overwrite=(processed_dates == BATCH_SIZE))
                for h in HORIZONS:
                    datasets[h] = []
                gc.collect()

    # Flush any remaining samples after loop
    if any(len(datasets[h]) > 0 for h in HORIZONS):
        print("Flushing remaining samples after processing all dates...")
        flush_to_csv(datasets)
        gc.collect()

    # Save final combined pickles
    print("Saving final combined pickle files...")
    for h in HORIZONS:
        csv_path = os.path.join(OUTPUT_PATH, f"dataset_h{h}.csv")
        df_h = pd.read_csv(csv_path)
        pkl_path = os.path.join(OUTPUT_PATH, f"dataset_h{h}.pkl")
        df_h.to_pickle(pkl_path)
        print(f"Saved pickle for horizon {h} with {len(df_h)} samples.")

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    return datasets

def flush_to_csv(datasets, overwrite=False):
    for h in HORIZONS:
        csv_path = os.path.join(OUTPUT_PATH, f"dataset_h{h}.csv")
        df_h = pd.DataFrame(datasets[h])
        if df_h.empty:
            continue
        if overwrite or not os.path.exists(csv_path):
            df_h.to_csv(csv_path, mode='w', header=True, index=False)
        else:
            df_h.to_csv(csv_path, mode='a', header=False, index=False)
        datasets[h] = []  # clear after saving


def csv_to_pkl_conversion(csv_folder=OUTPUT_PATH):
    """Utility to convert all CSV datasets in a folder to Pickle format."""
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv") and filename.startswith("dataset_h"):
            csv_path = os.path.join(csv_folder, filename)
            pkl_path = csv_path.replace(".csv", ".pkl")
            print(f"Converting {csv_path} to {pkl_path}...")
            df = pd.read_csv(csv_path)
            df.to_pickle(pkl_path)
    print("All CSV files converted to Pickle.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print("Loading all regions data...")
    df_all = load_all_regions()
    print(f"Total points loaded: {len(df_all)}")
    print("Building datasets with spatial context and plume predictions...")
    horizon_datasets = build_datasets_parallel(df_all)
