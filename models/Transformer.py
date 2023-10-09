import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from models.MLP import MLP
from models.embed import DataEmbedding

class Transformer(nn.Module):
    def __init__(self, feature_num, e_layers, num_heads, num_classes,win_size, device, dropout = 0.1,d_model = 512):
        super(Transformer,self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,nhead=num_heads)
        self.Transformer = TransformerEncoder(encoder_layer, e_layers)
        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(win_size)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)


        self.norm = nn.LayerNorm(d_model)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = MLP(d_model//4 * win_size, num_classes, dropout)

    def forward(self, x):
        x = self.enc_embedding(x)

        out = self.Transformer(x)

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = torch.flatten(out,1)

        out = self.head(out)

        return out
