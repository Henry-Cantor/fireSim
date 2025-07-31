import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from tqdm import tqdm

# --------------------- CONFIG ---------------------
regions = {
    "cal": {
        "lat_range": (32.5, 37.5),
        "lon_range": (-120.0, -114.0)
    },
    "colorado": {
        "lat_range": (37.0, 41.0),
        "lon_range": (-109.0, -102.0)
    },
    "texas": {
        "lat_range": (25.8, 33.0),
        "lon_range": (-106.7, -93.5)
    },
    "northeast": {
        "lat_range": (36.5, 45.5),
        "lon_range": (-82.0, -67.0)
    }
}

start_year = 2013
end_year = 2023
max_days_per_region = 250

smoke_folder = 'data/smoke'
output_folder = 'data/smoke/downscaled'
os.makedirs(output_folder, exist_ok=True)

def load_region_data(region):
    elev_file = f"data/elevation/{region}.tif"
    nlcd_file = f"data/nlcd/{region}_nlcd_2021.tif"
    
    with rasterio.open(elev_file) as elev_src:
        elev = elev_src.read(1)
        elev_affine = elev_src.transform
    
    with rasterio.open(nlcd_file) as lc_src:
        lc = lc_src.read(1, out_shape=elev.shape, resampling=Resampling.bilinear)
        
    norm_elev = (elev - np.nanmean(elev)) / np.nanstd(elev)
    norm_lc = (lc - np.nanmean(lc)) / np.nanstd(lc)
    
    return norm_elev, norm_lc, elev_affine

def latlon_to_rowcol_vectorized(lats, lons, transform, shape):
    rows, cols = rowcol(transform, lons, lats)
    rows = np.array(rows)
    cols = np.array(cols)
    valid_mask = (
        (rows >= 0) & (rows < shape[0]) &
        (cols >= 0) & (cols < shape[1])
    )
    return rows, cols, valid_mask

def process_file(file, region, bounds, norm_elev, norm_lc, elev_affine, grid_x, grid_y):
    lat_min, lat_max = bounds['lat_range']
    lon_min, lon_max = bounds['lon_range']

    date = file.replace('.csv', '')
    filepath = os.path.join(smoke_folder, region, file)
    df = pd.read_csv(filepath)

    df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
    if df.empty:
        return  # skip empty

    points = df[['lon', 'lat']].values
    values = df['smokePM_pred'].values

    nearest_interp = NearestNDInterpolator(points, values)
    linear_interp = LinearNDInterpolator(points, values)

    lons_flat = grid_x.flatten()
    lats_flat = grid_y.flatten()

    interp_linear = linear_interp(lons_flat, lats_flat)
    interp_nearest = nearest_interp(lons_flat, lats_flat)

    interp = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
    interp = interp.reshape(grid_x.shape)

    rows, cols, valid_mask = latlon_to_rowcol_vectorized(lats_flat, lons_flat, elev_affine, norm_elev.shape)
    sampled_elev = np.zeros_like(lats_flat, dtype=np.float32)
    sampled_lc = np.zeros_like(lats_flat, dtype=np.float32)
    sampled_elev[valid_mask] = norm_elev[rows[valid_mask], cols[valid_mask]]
    sampled_lc[valid_mask] = norm_lc[rows[valid_mask], cols[valid_mask]]
    sampled_elev = sampled_elev.reshape(grid_x.shape)
    sampled_lc = sampled_lc.reshape(grid_x.shape)

    downscaled = interp + 0.1 * sampled_elev + 0.05 * sampled_lc

    out_df = pd.DataFrame({
        'lat': grid_y.flatten(),
        'lon': grid_x.flatten(),
        'smokePM_downscaled': downscaled.flatten(),
        'norm_elevation': sampled_elev.flatten(),
        'norm_nlcd': sampled_lc.flatten()
    })

    out_path = os.path.join(output_folder, f'{region}_{date}.csv')
    out_df.to_csv(out_path, index=False)

def main():
    for region, bounds in regions.items():
        print(f"Processing region: {region}")
        norm_elev, norm_lc, elev_affine = load_region_data(region)
        lat_min, lat_max = bounds['lat_range']
        lon_min, lon_max = bounds['lon_range']

        grid_res_deg = 0.001  # 100m
        grid_x, grid_y = np.meshgrid(
            np.arange(lon_min, lon_max, grid_res_deg),
            np.arange(lat_min, lat_max, grid_res_deg)
        )

        region_smoke_folder = os.path.join(smoke_folder, region)
        daily_files = sorted([f for f in os.listdir(region_smoke_folder) if f.endswith('.csv')])

        filtered_files = []
        for f in daily_files:
            date = f.replace('.csv', '')
            year = int(date[:4])
            if start_year <= year <= end_year:
                filtered_files.append(f)
        filtered_files = filtered_files[:max_days_per_region]

        for file in tqdm(filtered_files):
            process_file(file, region, bounds, norm_elev, norm_lc, elev_affine, grid_x, grid_y)

if __name__ == "__main__":
    main()
