import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter  # Added import here
import csv
from scipy.spatial import cKDTree


def prepare_dataset(region):

    smoke_dir = f"data/smoke/{region}"
    wind_dir = f"data/wind_daily/{region}"
    elev_path = f"data/elevation/{region}.tif"
    nlcd_path = f"data/nlcd/{region}_nlcd_2021.tif"

    if not os.path.isdir(smoke_dir):
        raise FileNotFoundError(f"Missing smoke directory for region: {region}")
    if not os.path.isdir(wind_dir):
        raise FileNotFoundError(f"Missing wind directory for region: {region}")
    if not os.path.exists(elev_path):
        raise FileNotFoundError(f"Missing elevation file for region: {region}")
    if not os.path.exists(nlcd_path):
        raise FileNotFoundError(f"Missing NLCD file for region: {region}")

    elev_src = rasterio.open(elev_path)
    nlcd_src = rasterio.open(nlcd_path)

    samples = []

    pm25_files = sorted([f for f in os.listdir(smoke_dir) if f.endswith(".csv")])

    for pm25_file in tqdm(pm25_files, desc=f"Processing {region} PM2.5 files"):
        date = pm25_file.replace(".csv", "")
        smoke_path = os.path.join(smoke_dir, pm25_file)
        wind_path = os.path.join(wind_dir, f"{date}.csv")

        if not os.path.exists(wind_path):
            continue

        df_pm25 = pd.read_csv(smoke_path)
        df_pm25 = df_pm25.dropna(subset=["lat", "lon", "smokePM_pred"])
        df_pm25 = df_pm25[df_pm25["smokePM_pred"] >= 0]

        df_pm25["lat"] = df_pm25["lat"].astype(float)
        df_pm25["lon"] = df_pm25["lon"].astype(float)

        df_wind = pd.read_csv(wind_path)
        df_wind.columns = [c.lower() for c in df_wind.columns]
        df_wind["latitude"] = df_wind["latitude"].astype(float)
        df_wind["longitude"] = df_wind["longitude"].astype(float)

        # Build KDTree on wind coords (lat, lon)
        wind_coords = np.vstack((df_wind["latitude"], df_wind["longitude"])).T
        tree = cKDTree(wind_coords)

        # Query nearest wind point for each PM2.5 point
        pm25_coords = np.vstack((df_pm25["lat"], df_pm25["lon"])).T
        dists, idxs = tree.query(pm25_coords, k=1)

        filtered_pm25 = []
        filtered_wind = []

        for i, idx in enumerate(idxs):
            filtered_pm25.append(df_pm25.iloc[i])
            filtered_wind.append(df_wind.iloc[idx])

        if len(filtered_pm25) == 0:
            continue

        df_pm25_filtered = pd.DataFrame(filtered_pm25).reset_index(drop=True)
        df_wind_filtered = pd.DataFrame(filtered_wind).reset_index(drop=True)

        points = list(zip(df_pm25_filtered["lon"], df_pm25_filtered["lat"]))
        elev_vals = np.array([val[0] for val in elev_src.sample(points)])
        nlcd_vals = np.array([val[0] for val in nlcd_src.sample(points)])

        valid_mask = (~np.isnan(elev_vals)) & (~np.isnan(nlcd_vals)) & (elev_vals != -32768)

        df_pm25_filtered = df_pm25_filtered[valid_mask].reset_index(drop=True)
        df_wind_filtered = df_wind_filtered[valid_mask].reset_index(drop=True)
        elev_vals = elev_vals[valid_mask]
        nlcd_vals = nlcd_vals[valid_mask]

        for i in range(len(df_pm25_filtered)):
            features = [
                df_pm25_filtered.loc[i, "lat"],
                df_pm25_filtered.loc[i, "lon"],
                elev_vals[i],
                nlcd_vals[i],
                df_wind_filtered.loc[i, "wind_u10m"],
                df_wind_filtered.loc[i, "wind_v10m"],
                df_wind_filtered.loc[i, "wind_u50m"],
                df_wind_filtered.loc[i, "wind_v50m"],
                date  # add date as a feature
            ]
            target = df_pm25_filtered.loc[i, "smokePM_pred"]
            samples.append((features, target))

    elev_src.close()
    nlcd_src.close()

    return samples


def save_dataset_csv(samples, save_path):
    feature_names = [
        "lat", "lon", "elevation", "nlcd",
        "wind_u10m", "wind_v10m", "wind_u50m", "wind_v50m", "date"
    ]
    data = []
    for features, target in samples:
        data.append(features + [target])

    df = pd.DataFrame(data, columns=feature_names + ["pm25_target"])
    df.to_csv(save_path, index=False)


def read_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        transform = src.transform
    return data, transform


def read_wind_data(wind_dir, valid_dates=None):

    all_dfs = []
    for f in os.listdir(wind_dir):
        if f.endswith(".csv"):
            path = os.path.join(wind_dir, f)
            df = pd.read_csv(path, parse_dates=["date"])

            if valid_dates is not None:
                df = df[df["date"].dt.date.isin(valid_dates)]

            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    wind_df = pd.concat(all_dfs, ignore_index=True)
    return wind_df


def latlon_to_pixel(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def simulate_plume(pm25_csv_path,
                   wind_10m_speed, wind_10m_dir,
                   wind_50m_speed, wind_50m_dir,
                   elev, transform, nlcd):
    # Load PM2.5 CSV
    df = pd.read_csv(pm25_csv_path)
    lats = df['lat'].values
    lons = df['lon'].values
    pm25_vals = df['smokePM_pred'].values

    grid = np.zeros_like(elev, dtype=np.float32)

    for lat, lon, val in zip(lats, lons, pm25_vals):
        if val < 10:
            continue
        row, col = latlon_to_pixel(lat, lon, transform)
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] += val

    avg_speed = (wind_10m_speed + wind_50m_speed) / 2
    u = (wind_10m_speed * np.cos(np.radians(wind_10m_dir)) + wind_50m_speed * np.cos(np.radians(wind_50m_dir))) / 2
    v = (wind_10m_speed * np.sin(np.radians(wind_10m_dir)) + wind_50m_speed * np.sin(np.radians(wind_50m_dir))) / 2
    avg_dir_rad = np.arctan2(v, u) 
    sigma_x = 3 + 10 * np.abs(np.cos(avg_dir_rad)) * (10 / avg_speed)
    sigma_y = 3 + 10 * np.abs(np.sin(avg_dir_rad)) * (10 / avg_speed)

    plume = gaussian_filter(grid, sigma=[sigma_y, sigma_x])

    return plume

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
    return C


def load_dataset_csv(path, source_coords=(-100.0, 38.0)):
    import numpy as np
    samples = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            features = np.array([float(x) for x in row[:-1]], dtype=np.float32)
            target = float(row[-1])

            lat, lon = features[0], features[1]
            u, v = features[4], features[5]

            plume = gaussian_plume_estimate(source_coords, (lon, lat), u, v)
            extended_features = np.append(features, plume)

            samples.append((extended_features, target))

    return samples

