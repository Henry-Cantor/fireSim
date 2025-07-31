import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from plume.gaussian_plume import simulate_plume, read_wind_data
from plume.gaussian_plume import read_tiff

class PlumeDataset(Dataset):
    def __init__(self, region, smoke_dir, wind_dir, elev_path, nlcd_path, dates, forecast_days=[1,3,7]):
        self.region = region
        self.smoke_path = smoke_dir  # Already region-specific path from main.py
        self.wind_path = wind_dir    # Region-specific wind dir from main.py
        self.elev_path = elev_path
        self.nlcd_path = nlcd_path
        self.dates = dates
        self.forecast_days = forecast_days

        self.elev, self.transform = read_tiff(elev_path)
        self.nlcd, _ = read_tiff(nlcd_path)

        # Convert PM2.5 date strings ('YYYY_MM_DD') to datetime.date
        self.pm25_dates_dt = pd.to_datetime([d.replace("_", "-") for d in dates]).date

        # Load and preprocess all wind data once:
        # returns a DataFrame with columns: date, WS10M, WD10M, WS50M, WD50M
        self.wind_df = read_wind_data(self.wind_path, valid_dates=set(self.pm25_dates_dt))

        self.samples = self.build_samples()
        self.eval_dates = [s['date'] for s in self.samples]

    def build_samples(self):
        samples = []
        for i in range(len(self.dates)):
            for fd in self.forecast_days:
                if i + fd < len(self.dates):
                    d0 = self.dates[i]
                    d1 = self.dates[i + fd]
                    input_csv = os.path.join(self.smoke_path, d0 + ".csv")
                    target_csv = os.path.join(self.smoke_path, d1 + ".csv")
                    if not os.path.exists(input_csv) or not os.path.exists(target_csv):
                        continue

                    input_date_dt = pd.to_datetime(d0.replace("_", "-")).date()
                    wind_row = self.wind_df[self.wind_df['date'] == input_date_dt]
                    if wind_row.empty:
                        continue

                    sample = {
                        'input_date': d0,
                        'target_date': d1,
                        'forecast_day': fd,
                        'date': d0
                    }
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_csv = os.path.join(self.smoke_path, s['input_date'] + ".csv")
        target_csv = os.path.join(self.smoke_path, s['target_date'] + ".csv")

        # Convert date string to datetime.date to lookup wind features
        input_date_dt = pd.to_datetime(s['input_date'].replace("_", "-")).date()

        # Get wind features row matching input date (assumes unique dates)
        wind_row = self.wind_df[self.wind_df['date'] == input_date_dt]
        if wind_row.empty:
            raise ValueError(f"No wind data for date {input_date_dt} in region {self.region}")

        wind_10m_speed = wind_row['WS10M'].values[0]
        wind_10m_dir = wind_row['WD10M'].values[0]
        wind_50m_speed = wind_row['WS50M'].values[0]
        wind_50m_dir = wind_row['WD50M'].values[0]

        try:
            input_plume = simulate_plume(
                input_csv, 
                wind_10m_speed, wind_10m_dir,
                wind_50m_speed, wind_50m_dir,
                self.elev, self.transform,
                self.nlcd
            )

            target_plume = simulate_plume(
                target_csv,
                wind_10m_speed, wind_10m_dir,
                wind_50m_speed, wind_50m_dir,
                self.elev, self.transform,
                self.nlcd
            )
        except Exception as e:
            raise RuntimeError(f"Simulation failed for sample {s}: {e}")

        # Stack plume, elevation, NLCD
        input_tensor = np.stack([
            input_plume / 500.0,  # normalize
            self.elev / 3000.0,
            self.nlcd / 100.0
        ])

        target_tensor = target_plume / 500.0  # normalize

        return {
            'input': torch.tensor(input_tensor, dtype=torch.float32),
            'target': torch.tensor(target_tensor, dtype=torch.float32),
            'meta': s
        }
