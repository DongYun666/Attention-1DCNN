import torch
import torch.nn as nn

from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention
from models.AttentionLayer.WaveletAttentionLayer3 import WaveletAttention

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.embed import DataEmbedding


class WaveTransformerCNN1D(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes, win_size, device, dropout = 0.1,d_model = 512, wavelet_method = 'db4',trainable = True):
        super(WaveTransformerCNN1D, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
            
                    WaveletAttention(
            
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        L = win_size,
                        num_heads = num_heads,
                        wavename = wavelet_method,
                        d_model = d_model,
                        device = device,
                        trainable = trainable
                    ),
                    d_model=d_model,
                    dropout=dropout
                ) for i in range(e_layers)
            ],
        )

        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(win_size)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        
        self.head = MLP(d_model//4 * win_size, num_classes, dropout)
        # self.linear1 = nn.Linear(d_model//4 * win_size, num_classes,bias=True)

    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        out = self.encoder(x)             

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = torch.flatten(out,1) # 32 128

        # out = self.linear1(out)
        out = self.head(out)

        return out



