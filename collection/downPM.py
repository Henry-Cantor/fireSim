import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# --- Load and Prepare Grid Info ---
grid_shapefile = "data/smoke/10km_grid_wgs84.shp"
print("Reading grid shapefile...")
grid_gdf = gpd.read_file(grid_shapefile)
print("Original CRS:", grid_gdf.crs)

grid_gdf = grid_gdf.to_crs(epsg=4326)  # Ensure CRS is WGS84 for lat/lon
print("Converted CRS:", grid_gdf.crs)

grid_gdf['lon'] = grid_gdf.geometry.centroid.x
grid_gdf['lat'] = grid_gdf.geometry.centroid.y

# Rename ID column for consistency
grid_gdf = grid_gdf.rename(columns={"ID": "grid_id_10km"})

# Save ID and lat/lon only
grid_lookup = grid_gdf[['grid_id_10km', 'lat', 'lon']].copy()

# --- Define Regions ---
region_bounds = {
    "cal": (-125, -113.5, 32, 42),
    "colorado": (-109, -101, 36.5, 41),
    "texas": (-106.65, -93.5, 25.8, 36.5),
    "northeast": (-80, -66.9, 37, 47.5),  # includes northern VA
}

def assign_region(row):
    lon, lat = row['lon'], row['lat']
    for region, (min_lon, max_lon, min_lat, max_lat) in region_bounds.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return region
    return None

# Assign region
print("Assigning regions...")
grid_lookup['region'] = grid_lookup.apply(assign_region, axis=1)
num_outside = grid_lookup['region'].isnull().sum()
print(f"⚠️ Warning: {num_outside} grid points are outside all defined bounding boxes")

# --- Load PM2.5 Data ---
pm25_csv_path = "data/smoke/pm25.csv"
print("Reading PM2.5 data...")
pm25_df = pd.read_csv(pm25_csv_path)

# Make sure grid_id_10km is string type for merge
pm25_df['grid_id_10km'] = pm25_df['grid_id_10km'].astype(str)
grid_lookup['grid_id_10km'] = grid_lookup['grid_id_10km'].astype(str)

# Merge with lat/lon/region
merged_df = pm25_df.merge(grid_lookup, on='grid_id_10km', how='left')
print("Rows after merge:", len(merged_df))
print("Rows with missing coordinates:", merged_df[['lat', 'lon']].isnull().any(axis=1).sum())

# Convert 'date' column (format 'YYYYMMDD') to datetime
merged_df['date'] = pd.to_datetime(merged_df['date'], format='%Y%m%d', errors='coerce')

# Filter date and region
filtered_pm25 = merged_df[
    (merged_df['date'].dt.year >= 2013) &
    (merged_df['date'].dt.year <= 2023) &
    merged_df['region'].notnull()
]
print(f"Filtered PM2.5 data from {len(merged_df)} → {len(filtered_pm25)} rows after date and null filtering\n")

# --- Group and Save by Region and Date ---
for region in region_bounds.keys():
    region_df = filtered_pm25[filtered_pm25['region'] == region].copy()
    if region_df.empty:
        print(f"⚠️ No PM2.5 points found in region {region}")
        continue

    region_df['year'] = region_df['date'].dt.year
    region_df['month'] = region_df['date'].dt.month
    region_df['day'] = region_df['date'].dt.day

    print(f"🚧 Processing PM2.5 for {region}...")

    for (y, m, d), group in tqdm(region_df.groupby(['year', 'month', 'day']), desc=f"{region} dates"):
        out_dir = f"data/smoke/processed/{region}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{y}_{m:02d}_{d:02d}.csv")
        group[['grid_id_10km', 'smokePM_pred', 'lat', 'lon']].to_csv(out_path, index=False)

print("\n✅ PM2.5 regional extraction complete.")
