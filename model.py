import torch
import torch.nn as nn

class GRUNetwork(nn.Module):
    def __init__(self, input_size, linear_hidden_size, gru_hidden_size, num_layers, dropout=0.):
        super(GRUNetwork, self).__init__()
        self.linear = nn.Linear(input_size, linear_hidden_size)
        self.gru = nn.GRU(linear_hidden_size, gru_hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(gru_hidden_size, 1)

    def forward(self, x1, x2):
        # x: [B, L, D]

        x1 = self.linear(x1)
        x2 = self.linear(x2)
        # x: [B, L, H]

        h10 = torch.zeros(self.gru.num_layers, x1.size(0), self.gru.hidden_size).to(x1.device)
        h20 = torch.zeros(self.gru.num_layers, x2.size(0), self.gru.hidden_size).to(x2.device)
        # h: [num_layers, B, H]

        out1, h1 = self.gru(x1, h10)
        out2, h2 = self.gru(x2, h20)
        # out: [B, L, H]

        out1 = self.fc(out1)
        out2 = self.fc(out2)
        # out: [B, L, 1]

        out = torch.cat((out1, out2), dim=2)
        # out: [B, L, 2]

        out = torch.softmax(out, dim=2)
        # out: [B, L, 2]

        return out



if __name__ == "__main__":
    model = GRUNetwork(10, 20, 30, 2)
    x1 = torch.randn(5, 100, 10)
    x2 = torch.randn(5, 100, 10)
    import time
    begin_time = time.time()
    out = model(x1, x2)
    print("Time: ", time.time() - begin_time)
    print(out)
