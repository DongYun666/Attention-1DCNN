import torch
import torch.nn as nn


from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.AttentionLayer.AttentionLayer import LinearAttentionLayer
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention

from models.embed import DataEmbedding

class TransformerCNN1d(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes,win_size,device, dropout = 0.1,d_model = 512):
        super(TransformerCNN1d, self).__init__()

        self.enc_embedding = DataEmbedding(feature_num, d_model, device, dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoder = TrafficADEncoder(
            [
                TrafficADEncoderLayer(
                    LinearAttentionLayer(
                        FullAttention() if attention_type == 'full' else CosinAttention(num_heads=num_heads,device=device),
                        # d_model= d_model // (2 ** i),
                        d_model=d_model,
                        num_heads=num_heads
                    ),
                    # d_model=d_model//(2 ** i),
                    d_model=d_model,
                    dropout=dropout
                ) for i in range(e_layers)
            ],
        )

        # self.CNN1 = nn.Conv1d(in_channels=d_model//(2 ** (e_layers)), out_channels=d_model//(2**(e_layers)), kernel_size=3, stride=1, padding=1)
        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(win_size)
        # self.norm1 = nn.LayerNorm(d_model)
        self.act1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2) 

        # self.CNN2 = nn.Conv1d(in_channels=d_model//(2 ** (e_layers + 1)), out_channels=d_model//(2 ** (e_layers + 1)), kernel_size=3, stride=1, padding=1)
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(win_size)
        self.act2 = nn.Tanh()
        self.down = nn.Sequential(
            nn.BatchNorm1d(win_size),
            nn.Tanh(),
            nn.Dropout(p = dropout)
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.head = MLP(d_model//4 * win_size, num_classes, dropout)
        # self.linear1 = nn.Linear(d_model// (2 ** (e_layers+2))* win_size, num_classes,bias=True)
        # self.linear1 = nn.Linear(d_model//4 * win_size, d_model // 2 ,bias=True)
        # self.norm2 = nn.LayerNorm(512)
        # self.act2 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(512, num_classes,bias=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(d_model, d_model // 4 ,bias=True)

        self.head = nn.Linear(d_model//4 , num_classes,bias=True)
        # self.head = nn.Linear(d_model , num_classes,bias=True)
        
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weight_)
    def _init_weight_(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        
    def forward(self, x):

        # position embedding
        out = self.enc_embedding(x)     # B W D

        # cls token
        # cls_tokens = self.cls_token.expand(cnn_out.shape[0], -1, -1)  # B 1 D
        # cnn_out = torch.cat((cls_tokens, cnn_out), dim=1)                 # B W+1 D

        out = self.encoder(out)         # B W D 

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm1(out)
        out = self.act1(out)
        # out = self.down(out)
        out = self.maxpool1(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm2(out)
        # out = self.down(out)
        out = self.act1(out)
        out = self.maxpool2(out) 

        out = self.avgpool(out.transpose(1,2))
        out = torch.flatten(out,1) # 32 128
        # out = self.linear(out)
        if self.training:
            out = self.dropout(out)

        out = self.head(out)

        return out


