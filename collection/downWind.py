import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from tqdm import tqdm
from datetime import datetime

wind_input_folder = "data/wind"
elevation_folder = "data/elevation"
output_folder = "data/windDown"
os.makedirs(output_folder, exist_ok=True)

regions = {
    "cal": ("california", "cal.tif", 54),       # cal_0.csv to cal_53.csv
    "colorado": ("colorado", "colorado.tif", 56),
    "northeast": ("northeast", "northeast.tif", 108),
    "texas": ("texas", "texas.tif", 99),
}

start_date = pd.Timestamp('2013-01-01')
end_date = pd.Timestamp('2023-12-31')

def compute_slope(elevation):
    dx, dy = np.gradient(elevation)
    slope = np.sqrt(dx**2 + dy**2)
    return slope



def read_wind_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find the index of the header row
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("YEAR"):
            header_line = i
            break
    else:
        raise ValueError("Header row with 'YEAR' not found.")

    header_lines = lines[:header_line]

    # Read CSV starting at the header row
    df = pd.read_csv(path, skiprows=header_line)

    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()

    # Check required columns
    required_cols = {'YEAR', 'MO', 'DY', 'WS10M'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Construct date column
    df["DATE"] = pd.to_datetime(df[["YEAR", "MO", "DY"]], errors='coerce')

    return df, header_lines





for prefix, (region, elev_file, max_index) in regions.items():
    print(f"Loading elevation for {region}a from {elev_file}")
    elev_path = os.path.join(elevation_folder, elev_file)
    with rasterio.open(elev_path) as elev_src:
        elev = elev_src.read(1)
        transform = elev_src.transform
        meta = elev_src.meta.copy()
        slope = compute_slope(elev)
    rows, cols = elev.shape

    print(f"Collecting wind station data for {region}...")
    points = []
    wind_data_per_point = []

    dates_index = None  # to unify dates across all points

    for i in tqdm(range(max_index)):
        csv_filename = f"{prefix}_{i}.csv"
        csv_path = os.path.join(wind_input_folder, csv_filename)
        if not os.path.exists(csv_path):
            # Missing file, skip
            continue
        try:
            df, header_lines = read_wind_csv(csv_path)

            # Make columns uppercase, strip spaces for safety
            df.columns = df.columns.str.strip().str.upper()

            lat = None
            lon = None
            for line in header_lines:
                if line.startswith("Location:"):
                    parts = line.split()
                    try:
                        lat_idx = parts.index('latitude') + 1
                        lon_idx = parts.index('longitude') + 1
                        lat = float(parts[lat_idx])
                        lon = float(parts[lon_idx])
                    except Exception:
                        pass
                    break

            if lat is None or lon is None:
                print(f"  Could not parse lat/lon for {csv_filename}, skipping")
                continue

            # Now parse date with uppercase column names
            df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']])
            df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]


            # If no data in range, skip
            if df.empty:
                continue

            # Wind speed column, WS10M, replace missing -999 with nan
            ws = df['WS10M'].replace(-999, np.nan).values

            # Collect points and data
            points.append((lon, lat))
            wind_data_per_point.append(ws)

            # Store dates_index (once)
            if dates_index is None:
                dates_index = df['DATE'].reset_index(drop=True)
            else:
                # Sanity check that dates align
                if not dates_index.equals(df['DATE'].reset_index(drop=True)):
                    print(f"  Warning: dates mismatch in {csv_filename}")

        except Exception as e:
            print(f"  Error reading {csv_filename}: {e}")

    if len(points) == 0:
        print(f"No valid data for {region}, skipping region.")
        continue

    points = np.array(points)  # Nx2 array (lon, lat)
    wind_data_per_point = np.array(wind_data_per_point)  # shape (N_points, N_dates)

    print(f"Interpolating and downscaling daily wind data for {region}...")

    # Prepare grid coords for interpolation
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs_mesh, ys_mesh = np.meshgrid(xs, ys)
    xs_world, ys_world = rasterio.transform.xy(transform, ys_mesh, xs_mesh, offset='center')
    xs_world = np.array(xs_world)
    ys_world = np.array(ys_world)

    for day_idx, day in enumerate(tqdm(dates_index)):
        values = wind_data_per_point[:, day_idx]

        # Remove nan points to avoid interpolation errors
        valid_mask = ~np.isnan(values)
        valid_points = points[valid_mask]
        valid_values = values[valid_mask]

        if len(valid_points) == 0:
            print(f"  No valid data for {region} on {day.date()}, skipping day.")
            continue

        # Interpolate spatially
        grid_z = griddata(valid_points, valid_values, (xs_world, ys_world), method='linear', fill_value=np.nan)

        # Fill nan with nearest neighbor interpolation
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_z[nan_mask] = griddata(valid_points, valid_values, (xs_world[nan_mask], ys_world[nan_mask]), method='nearest')

        # Apply slope adjustment
        adjusted = grid_z * (1 + 0.3 * slope / np.nanmax(slope))
        adjusted = np.clip(adjusted, 0, None)

        # Update metadata
        meta.update(
            dtype=rasterio.float32,
            height=rows,
            width=cols,
            count=1,
            compress='lzw'
        )

        out_name = f"{region}_wind_{day.strftime('%Y%m%d')}.tif"
        out_path = os.path.join(output_folder, out_name)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(adjusted.astype(np.float32), 1)

print("All done!")
