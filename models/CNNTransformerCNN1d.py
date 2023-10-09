import torch
import torch.nn as nn
from models.AttentionLayer.ConvAttentionLayer import ConvAttentionLayer

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.embed import DataEmbedding

from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

class CNNADFormerCNN1d(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes, kernel_size,padding,win_size,device, dropout = 0.1,d_model = 512):
        super(CNNADFormerCNN1d, self).__init__()

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

        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(win_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        
        self.linear1 = nn.Linear(d_model//4 * win_size, 256,bias=True)
        self.dropout3 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(256, num_classes,bias=True)

    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        out = self.encoder(x)             

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = torch.flatten(out,1) # 32 128

        out = self.linear1(out)
        if self.training:
            out = self.dropout3(out)
        out = self.linear2(out)
        return out