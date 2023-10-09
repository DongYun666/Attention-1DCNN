import torch
import torch.nn as nn

class DeepPacket(nn.Module):
    def __init__(self,num_classes):
        super(DeepPacket, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv1d(1,200,5,3),
            nn.ReLU(),
        )
        self.cov2 = nn.Sequential(
            nn.Conv1d(200,200,4,3),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Sequential(
            nn.Linear(200*83, 100),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(50, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cov1(x.unsqueeze(2).transpose(1,2)) # B 1500 1  -> B 200 499
        x = self.cov2(x) # B 200 500 -> B 200 166
        x = self.max_pool(x) # B 200 167 -> B 200 83
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
