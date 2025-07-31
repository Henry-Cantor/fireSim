import torch
import torch.nn as nn

class PlumeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq):
        # x_seq: [batch_size, seq_len, input_dim]
        _, (hidden, _) = self.lstm(x_seq)  # hidden: [num_layers, batch, hidden_dim]
        out = self.fc(hidden[-1])  # use the last layer's hidden state
        return out
