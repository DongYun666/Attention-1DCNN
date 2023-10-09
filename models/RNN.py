import torch
import torch.nn as nn
from models.MLP import MLP

from models.embed import DataEmbedding


class RNN(nn.Module):
    def __init__(self,feature_num,e_layers,d_model,num_classes,win_size,device,dropout=0.1):
        super(RNN, self).__init__()
        e_layers = 1
        self.d_model = d_model
        self.enc_embedding = DataEmbedding(feature_num, d_model,device, dropout)

        self.rnn1 = nn.RNN(input_size=feature_num, hidden_size=d_model, num_layers=e_layers, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn2 = nn.RNN(input_size=d_model, hidden_size=d_model, num_layers=e_layers, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(dropout)

        self.e_layers = e_layers

        self.linear1 = nn.Linear(d_model * win_size, 512,bias=True)
        self.LayerNorm1 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256,bias=True)
        self.LayerNorm2 = nn.LayerNorm(256)
        self.dropout4 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_classes,bias=True)

        self.apply(self._init_weight_)

    def _init_weight_(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        
        out, _ = self.rnn1(x)
        if self.training:
            out = self.dropout1(out)
        out, _ = self.rnn2(out)
        if self.training:
            out = self.dropout2(out)
    

        out = torch.flatten(out,1) # 32 128

        out = self.linear1(out)
        out = self.LayerNorm1(out)
        if self.training:
            out = self.dropout3(out)
        
        out = self.linear2(out)
        out = self.LayerNorm2(out)
        if self.training:
            out = self.dropout4(out)
        out = self.linear3(out)

        return out
        