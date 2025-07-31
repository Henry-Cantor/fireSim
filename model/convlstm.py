import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels, kernel_size,
                              padding=padding)

        self.hidden_channels = hidden_channels

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        conv_out = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, list) else [hidden_channels] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else self.hidden_channels[i-1]
            cell = ConvLSTMCell(in_ch, self.hidden_channels[i], self.kernel_size[i])
            self.cells.append(cell)

    def forward(self, input_seq):
        # input_seq shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = input_seq.size()
        h = [torch.zeros(batch_size, self.hidden_channels[i], height, width, device=input_seq.device)
             for i in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels[i], height, width, device=input_seq.device)
             for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x = input_seq[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x, h[i], c[i])
                x = h[i]
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_channels, height, width)
        return outputs[:, -1]  # return last time step output



class PlumeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=4):  # output_dim=4 for 4 horizons
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim), take last layer
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out  # shape: (batch, output_dim)
