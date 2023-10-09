import torch
import torch.nn as nn
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention
from models.AttentionLayer.WaveletAttentionLayer2 import WaveletAttention

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.embed import DataEmbedding

class WaveletADFormer2(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes, win_size, device, dropout = 0.1,d_model = 512, wavelet_method = 'db4',trainable = True):
        super(WaveletADFormer2, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
            
                    WaveletAttention(
            
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        L = win_size,
                        level = num_heads,
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

        self.norm = nn.LayerNorm(d_model)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = MLP(d_model, num_classes, dropout)
    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        encoder_out = self.encoder(x)             

        out = self.norm(encoder_out)   # B W D

        out = self.avgpool(out.transpose(1,2))  # 32 128 1

        out = torch.flatten(out,1) # 32 128

        out = self.head(out)

        return out


