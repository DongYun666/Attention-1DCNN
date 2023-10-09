import torch
import torch.nn as nn


from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.AttentionLayer.AttentionLayer import LinearAttentionLayer
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

from models.embed import DataEmbedding

class LinearADFormer(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes,device, dropout = 0.1,d_model = 512):
        super(LinearADFormer, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
                    LinearAttentionLayer(
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        d_model= d_model,
                        num_heads=num_heads
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


