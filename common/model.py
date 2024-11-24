# common/model.py

import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Layer 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  # Layer 2

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
