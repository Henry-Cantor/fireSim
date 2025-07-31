import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from main2 import PlumeNetDeep  # Your model class

device = torch.device("cpu")

# ---- Load data ----
csv_path = "visuals/heatMap.csv"
df = pd.read_csv(csv_path)

required_cols = [
    "current_pm25", "plume_pred",
    "wind_u10m", "wind_v10m",
    "wind_u50m", "wind_v50m",
    "elevation", "nlcd",
    "context", "center_lat", "center_lon"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ---- Prepare features ----
center_features = df[[
    "current_pm25", "plume_pred",
    "wind_u10m", "wind_v10m",
    "wind_u50m", "wind_v50m",
    "elevation", "nlcd"
]].values.astype(np.float32)

scaler = StandardScaler()
scaler.fit(center_features)
center_features_scaled = scaler.transform(center_features)

max_neighbors = 20
context_list = []
for ctx in df["context"]:
    if pd.isna(ctx) or ctx == "[]":
        ctx_feats = np.zeros((max_neighbors, 10), dtype=np.float32)
    else:
        ctx_feats = np.array(eval(ctx, {"np": np}), dtype=np.float32)
        if ctx_feats.shape[0] > max_neighbors:
            ctx_feats = ctx_feats[:max_neighbors]
        else:
            pad_len = max_neighbors - ctx_feats.shape[0]
            pad = np.zeros((pad_len, 10), dtype=np.float32)
            ctx_feats = np.vstack([ctx_feats, pad])
    ctx_feats = (ctx_feats - ctx_feats.mean(axis=0)) / (ctx_feats.std(axis=0) + 1e-6)
    context_list.append(ctx_feats)

context_features = np.stack(context_list)

center_tensor = torch.tensor(center_features_scaled, dtype=torch.float32).to(device)
context_tensor = torch.tensor(context_features, dtype=torch.float32).to(device)

models = {}
for horizon in [1, 3, 7]:
    model = PlumeNetDeep().to(device)
    model.load_state_dict(torch.load(f"finals/plumeNet{horizon}_improved.pt", map_location=device))
    model.eval()
    models[horizon] = model

preds = {}
preds[0] = df["current_pm25"].values

with torch.no_grad():
    for h, model in models.items():
        outputs = model(center_tensor, context_tensor).cpu().numpy()
        preds[h] = outputs.squeeze() if outputs.ndim > 1 else outputs

lats = df["center_lat"].values
lons = df["center_lon"].values

# --- Create uniform grid covering bounding box ---
lat_min, lat_max = lats.min(), lats.max()
lon_min, lon_max = lons.min(), lons.max()
grid_lat = np.linspace(lat_min, lat_max, 200)
grid_lon = np.linspace(lon_min, lon_max, 200)
LON, LAT = np.meshgrid(grid_lon, grid_lat)

def grid_pm25(lat_points, lon_points, pm25_values):
    points = np.vstack([lon_points, lat_points]).T
    grid_pm = griddata(points, pm25_values, (LON, LAT), method='cubic')
    nan_mask = np.isnan(grid_pm)
    if np.any(nan_mask):
        grid_pm[nan_mask] = griddata(points, pm25_values, (LON, LAT), method='nearest')[nan_mask]
    return grid_pm

def in_hull(points, hull):
    return hull.find_simplex(points) >= 0

points = np.vstack([lons, lats]).T
hull = Delaunay(points)
grid_points = np.vstack([LON.ravel(), LAT.ravel()]).T
mask = in_hull(grid_points, hull).reshape(LON.shape)

# ---- Wind vectors (average u10/v10 into grid bins) ----
U = griddata(points, df["wind_u10m"].values, (LON, LAT), method="nearest")
V = griddata(points, df["wind_v10m"].values, (LON, LAT), method="nearest")

# Subsample wind for clarity
step = 50
LON_q = LON[::step, ::step]
LAT_q = LAT[::step, ::step]
U_q = U[::step, ::step]
V_q = V[::step, ::step]

# ---- Compute common color scale ----
all_grids = []
for day in [0, 1, 3, 7]:
    all_grids.append(grid_pm25(lats, lons, preds[day]))
combined_vals = np.concatenate([g[mask] for g in all_grids])
vmin = np.nanmin(combined_vals)
vmax = np.nanmax(combined_vals)

# --- Plot heatmaps ---
cmap = plt.cm.inferno.reversed()  # lighter = lower PM2.5

for i, day in enumerate([0, 1, 3, 7]):
    plt.figure(figsize=(8, 6))
    grid_vals = grid_pm25(lats, lons, preds[day])
    grid_vals_masked = np.where(mask, grid_vals, np.nan)
    
    pcm = plt.pcolormesh(LON, LAT, grid_vals_masked, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    plt.quiver(LON_q, LAT_q, U_q, V_q, scale=100, width=0.005, alpha=0.7, color='cyan')
    
    plt.colorbar(pcm, label="PM2.5 (μg/m³)")
    plt.title(f"Day {day} PM2.5 Heatmap with Wind Vectors")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"heatmap_day{day}_masked_region_wind.png")
    plt.close()

print("✅ Done. Heatmaps with shared color scale and wind vectors saved.")
