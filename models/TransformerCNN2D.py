import torch
import torch.nn as nn


from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.AttentionLayer.AttentionLayer import LinearAttentionLayer
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

from models.embed import DataEmbedding

class TransformerCNN2D(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes,win_size,device, dropout = 0.1,d_model = 512):
        super(TransformerCNN2D, self).__init__()

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

        self.CNN1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(1)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.CNN2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        
        self.linear1 = nn.Linear((win_size//4) * (d_model//4), num_classes,bias=True)

    def forward(self, x):

        # position embedding
        x = self.enc_embedding(x)     # B W D

        out = self.encoder(x)         # B W D 

        out = self.CNN1(out.unsqueeze(1)).squeeze(1)
        out = self.norm1(out.unsqueeze(1)).squeeze(1)
        out = self.act1(out)
        out = self.maxpool(out.unsqueeze(1)).squeeze(1)

        out = self.CNN2(out.unsqueeze(1)).squeeze(1)
        out = self.norm1(out.unsqueeze(1)).squeeze(1)
        out = self.act1(out)
        out = self.maxpool(out.unsqueeze(1)).squeeze(1)

        out = torch.flatten(out,1) # 32 128

        out = self.linear1(out)

        return out


