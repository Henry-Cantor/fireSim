import pickle
import numpy as np

DATASET_DIR = "data/processed/horizon_datasets"
HORIZONS = [1, 3, 7, 14]

for h in HORIZONS:
    path = f"{DATASET_DIR}/dataset_h{h}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"--- Horizon {h} dataset ---")
    print(f"Total samples: {len(data)}")

    # Sample some random entries
    sample_entries = np.random.choice(data, size=min(5, len(data)), replace=False)

    for i, entry in enumerate(sample_entries):
        print(f"Sample #{i+1}:")
        print(f"  Center lat/lon: ({entry['center_lat']:.4f}, {entry['center_lon']:.4f})")
        print(f"  Region: {entry['region']}")
        print(f"  Date: {entry['date']}")
        print(f"  Current PM2.5: {entry['current_pm25']:.3f}")
        print(f"  Plume prediction: {entry['plume_pred']:.3f}")
        print(f"  Target PM2.5 (horizon {h} days): {entry['target']:.3f}")
        print(f"  Wind 10m (u,v): ({entry['wind_u10m']:.3f}, {entry['wind_v10m']:.3f})")
        print(f"  Wind 50m (u,v): ({entry['wind_u50m']:.3f}, {entry['wind_v50m']:.3f})")
        print(f"  Elevation: {entry['elevation']:.1f}")
        print(f"  NLCD: {entry['nlcd']}")
        print(f"  Number of neighbors in context: {len(entry['context'])}")
        if len(entry['context']) > 0:
            neighbor0 = entry['context'][0]
            print(f"  First neighbor context features (Δlat, Δlon, wind_u10, wind_v10, wind_u50, wind_v50, elev, nlcd, pm25, plume):")
            print(f"    {neighbor0}")
        print()

    # Check for any missing target values or NaNs
    targets = [e['target'] for e in data]
    print(f"Targets min/max: {np.min(targets):.3f} / {np.max(targets):.3f}")
    print(f"Any NaNs in targets? {np.any(np.isnan(targets))}")

    print("\n")
