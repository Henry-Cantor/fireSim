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
HORIZON = 3
PICKLE_PATH = os.path.join(DATASET_DIR, f"dataset_h{HORIZON}.pkl")
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f"plumeNet{HORIZON}.pt"


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
class PlumeCNNLSTM(nn.Module):
    def __init__(self, center_feat_dim=8, context_feat_dim=10, max_neighbors=20, hidden_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(context_feat_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.lstm = nn.LSTM(
            input_size=center_feat_dim,
            hidden_size=hidden_dim,
            num_layers=5,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, center_feats, context_feats):
        cnn_input = context_feats.permute(0, 2, 1)  # (B, C, T)
        context_out = self.cnn(cnn_input).squeeze(-1)

        lstm_input = center_feats.unsqueeze(1)  # (B, 1, F)
        _, (h_n, _) = self.lstm(lstm_input)
        lstm_out = h_n[-1]  # (B, H)

        combined = torch.cat([context_out, lstm_out], dim=1)
        return self.fc(combined).squeeze(1)


# ---------------- Loss Function ----------------
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target)))


# ---------------- Training ----------------
def train_model(model, train_loader, optimizer, criterion):
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
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(train_loader.dataset)


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
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-6))) * 100

    return preds, trues, mae, r2, mape


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

    model = PlumeCNNLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = LogCoshLoss()

    train_losses = []
    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
        train_losses.append(loss)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")

    # model.load_state_dict(torch.load("plumeNet3.pt", map_location=torch.device("cpu")))  # Set model to evaluation mode (important!)

    preds, trues, mae, r2, mape = evaluate_model(model, test_loader)
    print(f"[RESULT] MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")

    # # plot_metrics(train_losses, HORIZON)
    plot_predictions(preds, trues, HORIZON)
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
