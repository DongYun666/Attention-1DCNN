import torch
import torch.nn as nn
from models.AttentionLayer.ConvAttentionLayer import ConvAttentionLayer

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.embed import DataEmbedding

from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

class CNNADFormer(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes, kernel_size,padding,device, dropout = 0.1,d_model = 512):
        super(CNNADFormer, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
            
                    ConvAttentionLayer(
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        d_model=d_model,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        padding = padding
                    ),

                    d_model=d_model,
                    dropout=dropout
                ) for i in range(e_layers)
            ],
        )

        self.norm = nn.LayerNorm(d_model)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes,bias=True)
        # self.head = MLP(d_model, num_classes, dropout)
    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        encoder_out = self.encoder(x)             

        out = self.norm(encoder_out)   # B W D

        out = self.avgpool(out.transpose(1,2))  # 32 128 1

        out = torch.flatten(out,1) # 32 128

        out = self.head(out)

        return out