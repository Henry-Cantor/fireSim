import geopandas as gpd
import pandas as pd

# Paths
grid_shapefile_path = "data/smoke/gridInfo.shp"

# Load shapefile
grid_gdf = gpd.read_file(grid_shapefile_path)
print("Original CRS:", grid_gdf.crs)

# Set CRS if missing
if grid_gdf.crs is None:
    grid_gdf = grid_gdf.set_crs(epsg=4326)

# Reproject to projected CRS for accurate centroid calculation (UTM zone 11N is common for California)
projected_crs = 32611
grid_gdf_proj = grid_gdf.to_crs(epsg=projected_crs)

# Calculate centroids and convert back to lat/lon (EPSG:4326)
centroids = grid_gdf_proj.geometry.centroid.to_crs(epsg=4326)
grid_gdf['lon'] = centroids.x
grid_gdf['lat'] = centroids.y

# Rename ID column to grid_id_10km if needed
if 'ID' in grid_gdf.columns:
    grid_gdf = grid_gdf.rename(columns={'ID': 'grid_id_10km'})

# Define California bounding box
cal_bbox = (-125, 32, -114, 43)  # (minx, miny, maxx, maxy)

# Filter points inside California bbox
in_cal = grid_gdf[
    (grid_gdf['lon'] >= cal_bbox[0]) & (grid_gdf['lon'] <= cal_bbox[2]) &
    (grid_gdf['lat'] >= cal_bbox[1]) & (grid_gdf['lat'] <= cal_bbox[3])
]

print(f"Number of grid points in California bbox: {len(in_cal)}")
print("Sample grid IDs in California bbox:")
print(in_cal['grid_id_10km'].head(20).values)
