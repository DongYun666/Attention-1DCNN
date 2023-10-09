import torch
import torch.nn as nn


class TrafficADEncoderLayer(nn.Module):
    def __init__(self,attention,d_model,dropout):
        super(TrafficADEncoderLayer, self).__init__()
        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.GELU()

    def forward(self,x):

        x = self.attention(x) + x

        if self.training:
            x = self.dropout(x)

        x = self.norm1(x)

        x = self.feedforward(x) + x

        x = self.norm2(x)

        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in,win_size):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(win_size)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1)).transpose(1,2)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x

class TrafficADEncoderLayer3(nn.Module):
    def __init__(self,attention,d_model,win_size,dropout):
        super(TrafficADEncoderLayer3, self).__init__()
        self.attention = attention

        self.norm1 = nn.BatchNorm1d(win_size)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.GELU()

    def forward(self,x):

        x = self.attention(x) + x

        if self.training:
            x = self.dropout(x)

        x = self.norm1(x)

        x = self.act1(x)

        x = self.maxpool(x)

        return x


class TrafficADEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TrafficADEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
    def forward(self, x):
        for attn_layer in  self.attn_layers:
            x = attn_layer(x)
        return x
    
class TrafficADEncoder2(nn.Module):
    def __init__(self, stage_layers,win_size):
        super(TrafficADEncoder2, self).__init__()
        self.attn_layers = nn.ModuleList(stage_layers)
        self.norm1 = nn.BatchNorm1d(win_size)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
    def forward(self, x):
        for layer in self.attn_layers:
            x = layer(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.maxpool(x)
        return x
    

class TrafficADEncoder3(nn.Module):
    def __init__(self, stage_layers):
        super(TrafficADEncoder3, self).__init__()
        self.attn_layers = nn.ModuleList(stage_layers)

    def forward(self, x):
        for layer in self.attn_layers:
            x = layer(x)
        return x