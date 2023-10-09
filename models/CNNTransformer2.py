import torch
import torch.nn as nn
from models.AttentionLayer.ConvAttentionLayer import ConvAttentionLayer

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoder2, TrafficADEncoderLayer
from models.embed import DataEmbedding

from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

class CNNADFormer2(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes,win_size, kernel_size,padding,device, dropout = 0.1,d_model = 512):
        super(CNNADFormer2, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder2(
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
                ),
                TrafficADEncoderLayer(
                    ConvAttentionLayer(
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        d_model=d_model//2,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        padding = padding
                    ),
                    d_model=d_model//2,
                    dropout=dropout
                )
            ],
            win_size = win_size
        )

        self.head = nn.Linear(d_model//4 * win_size, num_classes,bias=True)
        # self.head = MLP(d_model, num_classes, dropout)
    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        encoder_out = self.encoder(x)     

        out = torch.flatten(encoder_out,1) 

        return self.head(out)        
