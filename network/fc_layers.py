import torch
import torch.nn as nn


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.fc1 = nn.Linear(1792, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 28)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))

        return x
