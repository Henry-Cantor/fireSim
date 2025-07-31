# model/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

def train_model(model, dataset, epochs=5, lr=1e-3):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_seq, y in loader:
            optimizer.zero_grad()
            y_pred = model(x_seq).squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")
    return model
