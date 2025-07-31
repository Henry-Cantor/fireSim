import torch
import os
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from operator import itemgetter
from math import radians, cos, sin, asin, sqrt, atan2, degrees

def parse_date(date_str):
    s = str(date_str).strip()
    if "_" in s:
        return datetime.strptime(s, "%Y_%m_%d").date()
    else:
        s_int = str(int(float(s)))
        return datetime.strptime(s_int, "%Y%m%d").date()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    brng = atan2(x, y)
    brng = degrees(brng)
    return (brng + 360) % 360

def round_coord(x, decimals=1):
    return round(float(x), decimals)

class LSTMSequenceDatasetMultiHorizon(Dataset):
    def __init__(self, samples, seq_len=5, horizons=[1,3,7,14], max_neighbors=20, radius_km=50, cache_path=None):
        self.seq_len = seq_len
        self.horizons = horizons
        self.max_neighbors = max_neighbors
        self.radius_km = radius_km
        self.samples = []

        if cache_path and os.path.exists(cache_path):
            print(f"[INFO] Loading cached dataset from {cache_path}")
            self.samples = torch.load(cache_path)
            print(f"[INFO] Loaded {len(self.samples)} samples from cache.")
            return

        grouped = defaultdict(list)
        for s in samples:
            features = s[0]
            target = s[1]
            # ROUND the lat/lon to 1 decimal here!
            lat_lon = (round_coord(features[0], 1), round_coord(features[1], 1))
            try:
                date = parse_date(features[8])
            except Exception:
                continue
            grouped[lat_lon].append((date, features, target))

        for latlon, records in grouped.items():
            records.sort(key=itemgetter(0))

        self.spatial_points = list(grouped.keys())

        print("Now going to compute neighbors.")
        self.neighbor_map = {}
        for latlon in self.spatial_points:
            lat_c, lon_c = latlon
            neighbors = []
            for nlat, nlon in self.spatial_points:
                dist = haversine(lat_c, lon_c, nlat, nlon)
                if dist <= self.radius_km:
                    neighbors.append((nlat, nlon, dist))
            neighbors.sort(key=lambda x: x[2])  # sort by distance
            self.neighbor_map[latlon] = neighbors[:self.max_neighbors]
            print("Neighbor loop done.")
        print("Neighbors mapped.")

        self.features_by_point_date = {}
        self.targets_by_point_date = {}
        for latlon, records in grouped.items():
            for date, features, target in records:
                self.features_by_point_date[(latlon[0], latlon[1], date)] = features
                self.targets_by_point_date[(latlon[0], latlon[1], date)] = target
            print("One lookup done.")
        print("Lookup dicts made.")

        for latlon, records in grouped.items():
            dates = [r[0] for r in records]

            max_horizon = max(horizons)
            for i in range(len(records) - seq_len - max_horizon + 1):
                seq_dates = dates[i:i+seq_len]
                last_date = seq_dates[-1]

                seq_input = []
                for day_date in seq_dates:
                    day_neighbors = []
                    lat_c, lon_c = latlon

                    for nlat, nlon, dist in self.neighbor_map[latlon]:
                        fkey = (nlat, nlon, day_date)
                        if fkey not in self.features_by_point_date:
                            continue
                        nf = self.features_by_point_date[fkey]
                        brng = bearing(lat_c, lon_c, nlat, nlon)
                        nf_ext = np.append(nf, [dist, brng])
                        day_neighbors.append(nf_ext)

                    if len(day_neighbors) < self.max_neighbors:
                        feat_dim = len(day_neighbors[0]) if day_neighbors else len(records[0][1]) + 2
                        day_neighbors.extend([np.zeros(feat_dim)] * (self.max_neighbors - len(day_neighbors)))
                    else:
                        day_neighbors = day_neighbors[:self.max_neighbors]

                    seq_input.append(np.stack(day_neighbors))

                seq_input = np.stack(seq_input)

                target_y = []
                valid_targets = False
                for h in horizons:
                    target_date = last_date + timedelta(days=h)
                    tkey = (latlon[0], latlon[1], target_date)
                    if tkey not in self.targets_by_point_date:
                        target_y.append(np.nan)
                    else:
                        target_y.append(self.targets_by_point_date[tkey])
                        valid_targets = True

                if valid_targets:
                    self.samples.append((seq_input, np.array(target_y, dtype=np.float32)))

            print(f"Processed sequence for {latlon} with dates {seq_dates[0]} to {last_date}, targets: {target_y}")

        print(f"[INFO] Created LSTMSequenceDatasetMultiHorizon with {len(self.samples)} samples")

        if cache_path:
            print(f"[INFO] Saving processed dataset to {cache_path}")
            torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y = self.samples[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
