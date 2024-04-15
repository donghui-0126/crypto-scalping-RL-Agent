import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import math

learning_rate = 0.0001

class Classifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(Classifier, self).__init__()

        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=(12, 2), stride=(12, 2))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 17), stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9), stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.lstm1 = nn.LSTM(512, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape([-1, 1, 30, 64])
        
        x = F.relu(self.bn1(self.conv2d_1(x)))
        x = F.relu(self.bn2(self.conv2d_2(x)))
        x = F.relu(self.bn3(self.conv2d_3(x)))

        # Assuming x has shape (batch_size, channels, height, width)
        x = x.view(x.size(0), -1)
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out[:, :]  # Take the last time step's output

        x = F.relu(self.fc1(lstm_out))
        output = self.fc2(x)

        return F.softmax(output, dim=-1)
