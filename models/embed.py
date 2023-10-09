import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LinearEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(LinearEmbedding, self).__init__()
        self.linear = nn.Linear(in_features=c_in, out_features=d_model)
        kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.linear(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, feature_num, d_model,device, dropout=0.1):  # c_in 对应初始数据的特征数量  d_model 对应的是最后输出的维度
        super(DataEmbedding, self).__init__()
        self.device = device
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.linear_embedding = LinearEmbedding(c_in=feature_num, d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.linear_embedding(x) + self.position_embedding(x)
        x = self.linear_embedding(x)
        # x = self.position_embedding(x)
        # x = self.norm(x)
        
        return x
