import os
import rasterio
from rasterio.windows import from_bounds

# Paths to your full downloaded NLCD files (update filenames if needed)
NLCD_2017_PATH = "data/nlcd/full/Annual_NLCD_LndCov_2017_CU_C1V1.tif"
NLCD_2021_PATH = "data/nlcd/full/Annual_NLCD_LndCov_2021_CU_C1V1.tif"

# Output directory (same as input folder)
OUTPUT_DIR = "data/nlcd/clipped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regions defined as (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "california": (-121.5, 32.0, -114.0, 35.0),
    "texas": (-101.0, 28.5, -94.0, 33.5),
    "colorado": (-107.5, 36.5, -102.0, 41.5),
    "northeast": (-79.5, 36.0, -71.0, 45.0),
}

from rasterio.warp import transform_bounds

def clip_region(input_tif, region_name, bbox):
    with rasterio.open(input_tif) as src:
        raster_crs = src.crs

        # Convert bbox from EPSG:4326 (lon/lat) to raster CRS
        bbox_proj = transform_bounds('EPSG:4326', raster_crs, *bbox, densify_pts=21)

        left, bottom, right, top = src.bounds
        print(f"Raster bounds: left={left}, bottom={bottom}, right={right}, top={top}")
        print(f"Projected bbox for {region_name}: {bbox_proj}")

        intersect_left = max(bbox_proj[0], left)
        intersect_right = min(bbox_proj[2], right)
        intersect_bottom = max(bbox_proj[1], bottom)
        intersect_top = min(bbox_proj[3], top)

        if intersect_right <= intersect_left or intersect_top <= intersect_bottom:
            print(f"WARNING: No intersection between bbox and raster for {region_name}")
            return

        intersect_bbox = (intersect_left, intersect_bottom, intersect_right, intersect_top)

        window = rasterio.windows.from_bounds(*intersect_bbox, transform=src.transform)
        if window.width <= 0 or window.height <= 0:
            print(f"WARNING: Computed zero-sized window for {region_name}, skipping")
            return

        data = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
            "compress": "lzw"
        })

        out_path = os.path.join(OUTPUT_DIR, f"nlcd_{region_name}_{os.path.basename(input_tif)}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)

        print(f"Saved clipped NLCD for {region_name} to {out_path}")


def main():
    for year_path in [NLCD_2017_PATH, NLCD_2021_PATH]:
        if not os.path.exists(year_path):
            print(f"ERROR: File not found: {year_path}")
            continue

        for region_name, bbox in REGIONS.items():
            print(f"Clipping {os.path.basename(year_path)} for {region_name}...")
            clip_region(year_path, region_name, bbox)

if __name__ == "__main__":
    main()
