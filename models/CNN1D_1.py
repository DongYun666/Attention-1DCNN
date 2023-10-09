import torch
import torch.nn as nn
from models.MLP import MLP

from models.embed import DataEmbedding


class CNN1D(nn.Module):
    def __init__(self,num_classes):
        super(CNN1D, self).__init__()

        self.conv1d1 = nn.Conv1d(1,32,25,stride=1,padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=3,stride=3,padding=1)
        self.conv1d2 = nn.Conv1d(32,64,25,stride=1,padding='same')
        self.maxpool2 = nn.MaxPool1d(3,3,padding=1)
        self.linear1 = nn.Linear(88 * 64,1024)
        self.linear2 = nn.Linear(1024,num_classes)
        self.rel = nn.ReLU()
        # self.apply(self._init_weight_)

    # def _init_weight_(self,m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.conv1d1(x.unsqueeze(2).transpose(1,2))
        x = self.rel(x)
        x = self.maxpool1(x)
        x = self.conv1d2(x)
        x = self.rel(x)
        x = self.maxpool2(x)
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
