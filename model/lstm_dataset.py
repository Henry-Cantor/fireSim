import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from plume.gaussian_plume import simulate_plume, read_wind_data
from plume.gaussian_plume import read_tiff

class LSTMSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples, seq_len=5):
        self.seq_len = seq_len
        self.samples = []

        # Group by grid ID and sort by date
        from collections import defaultdict
        from operator import itemgetter

        grouped = defaultdict(list)
        for s in samples:
            grid_id = int(s[3])  # assuming index 3 is grid ID
            date = s[4]          # assuming index 4 is datetime object
            grouped[grid_id].append((date, s))

        for grid_samples in grouped.values():
            grid_samples.sort(key=itemgetter(0))  # sort by date
            features = [s[1] for s in grid_samples]
            targets = [s[2] for s in grid_samples]  # PM2.5

            for i in range(len(features) - seq_len + 1):
                seq_x = features[i:i+seq_len]
                seq_y = targets[i+seq_len-1]  # predict PM2.5 of last day
                self.samples.append((np.stack(seq_x), seq_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y = self.samples[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
