# ----- (Imports and Config remain unchanged) -----
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
DATASET_DIR = "data/processed/horizon_datasetsNEW"
HORIZON = 7
PICKLE_PATH = os.path.join(DATASET_DIR, f"dataset_h{HORIZON}.pkl")
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f"plumeNet{HORIZON}_improved.pt"


# ---------------- Dataset ----------------
class PlumeDataset(Dataset):
    def __init__(self, data_list, max_neighbors=20, scaler=None):
        self.data = data_list
        self.max_neighbors = max_neighbors
        self.scaler = scaler

        self.center_features = []
        for sample in self.data.itertuples():
            feats = [
                sample.current_pm25, sample.plume_pred,
                sample.wind_u10m, sample.wind_v10m,
                sample.wind_u50m, sample.wind_v50m,
                sample.elevation, sample.nlcd
            ]
            self.center_features.append(feats)

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.center_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        center_feats = np.array([
            sample["current_pm25"], sample["plume_pred"],
            sample["wind_u10m"], sample["wind_v10m"],
            sample["wind_u50m"], sample["wind_v50m"],
            sample["elevation"], sample["nlcd"]
        ], dtype=np.float32)

        center_feats = self.scaler.transform([center_feats])[0]

        context = sample.get("context", [])
        if len(context) == 0:
            context_feats = np.zeros((self.max_neighbors, 10), dtype=np.float32)
        else:
            context_feats = np.array(eval(context, {"np": np}), dtype=np.float32)
            if context_feats.shape[0] > self.max_neighbors:
                context_feats = context_feats[:self.max_neighbors]
            else:
                padding = np.zeros((self.max_neighbors - context_feats.shape[0], 10), dtype=np.float32)
                context_feats = np.vstack([context_feats, padding])

        context_feats = (context_feats - context_feats.mean(axis=0)) / (context_feats.std(axis=0) + 1e-6)
        target = np.float32(sample["target"])

        return (
            torch.tensor(center_feats, dtype=torch.float32),
            torch.tensor(context_feats, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

# ---------------- Model ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + res)

class PlumeNetDeep(nn.Module):
    def __init__(self, center_feat_dim=8, context_feat_dim=10, max_neighbors=20, hidden_dim=256):
        super().__init__()

        self.context_net = nn.Sequential(
            ResidualBlock(context_feat_dim, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool1d(1),
        )

        self.temporal_net = nn.LSTM(
            input_size=center_feat_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, center_feats, context_feats):
        x_context = context_feats.permute(0, 2, 1)
        context_vec = self.context_net(x_context).squeeze(-1)

        x_center = center_feats.unsqueeze(1)
        _, (h_n, _) = self.temporal_net(x_center)
        lstm_vec = torch.cat([h_n[-2], h_n[-1]], dim=1)

        combined = torch.cat([context_vec, lstm_vec], dim=1)
        return self.fc(combined).squeeze(1)

# ---------------- Training ----------------
def train_model(model, train_loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for center_feats, context_feats, targets in tqdm(train_loader, desc="Training"):
        center_feats = center_feats.to(DEVICE)
        context_feats = context_feats.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(center_feats, context_feats)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)

    scheduler.step(total_loss)
    return total_loss / len(train_loader.dataset)

class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target)))

# ---------------- Evaluation ----------------
def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for center_feats, context_feats, targets in data_loader:
            center_feats = center_feats.to(DEVICE)
            context_feats = context_feats.to(DEVICE)
            preds = model(center_feats, context_feats)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_targets)

    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    mask = trues > 5
    mape = np.mean(np.abs((preds[mask] - trues[mask]) / (trues[mask] + 1e-6))) * 100

    return preds, trues, mae, r2, mape

# ----- plot_metrics, plot_predictions stay unchanged -----


# ---------------- Visualization ----------------
def plot_metrics(train_losses, horizon):
    plt.plot(train_losses, label="Train Loss")
    plt.title(f"Training Loss - Horizon {horizon} Day(s)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(preds, trues, horizon):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(x=trues, y=preds, fill=True, cmap="viridis", thresh=0.02)
    max_val = max(preds.max(), trues.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label="Perfect")
    plt.xlabel("True PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title(f"Predicted vs True - Horizon {horizon}d")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    residuals = preds - trues
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------- Main ----------------
def run():
    print(f"[INFO] Using device: {DEVICE}")
    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(f"Missing pickle file: {PICKLE_PATH}")

    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] Loaded {len(data)} samples")

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    temp_ds = PlumeDataset(train_data)
    scaler = temp_ds.scaler
    train_ds = PlumeDataset(train_data, scaler=scaler)
    test_ds = PlumeDataset(test_data, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = PlumeNetDeep().to(DEVICE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    # criterion = LogCoshLoss()

    # train_losses = []
    # for epoch in range(EPOCHS):
    #     loss = train_model(model, train_loader, optimizer, scheduler, criterion)
    #     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
    #     train_losses.append(loss)
    #     torch.save(model.state_dict(), MODEL_SAVE_PATH)

    model.load_state_dict(torch.load("plumeNet3_improved.pt", map_location=torch.device("cpu")))  # Set model to evaluation mode (important!)

    preds, trues, mae, r2, mape = evaluate_model(model, test_loader)
    print(f"[RESULT] MAE: {mae:.4f}, R²: {r2:.4f}, MAPE (>5 µg/m³): {mape:.2f}%")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    plot_predictions(preds, trues, HORIZON)

if __name__ == "__main__":
    run()
