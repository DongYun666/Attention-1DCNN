import torch
import torch.nn as nn


from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.AttentionLayer.AttentionLayer import LinearAttentionLayer
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

from models.embed import DataEmbedding

class CNN1dTransformer(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes,win_size,device, dropout = 0.1,d_model = 512):
        super(CNN1dTransformer, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
                    LinearAttentionLayer(
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        d_model= d_model//4,
                        num_heads=num_heads
                    ),
                    d_model=d_model//4,
                    dropout=dropout
                ) for i in range(e_layers)
            ],
        )

        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(win_size)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model//4, num_classes,bias=True)
        # self.norm2 = nn.LayerNorm(512)
        # self.act2 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout)

        # self.linear2 = nn.Linear(512, num_classes,bias=True)
        
    def forward(self, x):

        # position embedding
        out = self.enc_embedding(x)     # B W D

        # out = self.encoder(out)         # B W D 

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.maxpool(out)

        out = self.encoder(out)         # B W D

        out = self.avgpool(out.transpose(1,2))
        out = torch.flatten(out,1) # 32 128
        out = self.head(out)
        return out


