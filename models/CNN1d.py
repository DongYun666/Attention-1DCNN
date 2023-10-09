import torch
import torch.nn as nn
from models.MLP import MLP

from models.embed import DataEmbedding


class CNN1d(nn.Module):
    def __init__(self,feature_num,e_layers,d_model,num_classes,win_size,device,dropout=0.1):
        super(CNN1d, self).__init__()

        # self.enc_embedding = DataEmbedding(feature_num, d_model,device, dropout)
        self.linear = nn.Linear(feature_num, d_model,bias=True)
        self.linear3 = nn.Linear(d_model, d_model,bias=True)
        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)         
        self.norm = nn.BatchNorm1d(win_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        
        # self.head = MLP(d_model//4 * 10, num_classes, dropout)

        self.e_layers = e_layers
        self.hidden_size = d_model//2

        self.linear1 = nn.Linear(d_model//4 * 10, 256,bias=True)
        
        self.head = nn.Linear(d_model//4 , num_classes,bias=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.LayerNorm = nn.LayerNorm(256)
        self.dropout3 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(256, num_classes,bias=True)

        self.apply(self._init_weight_)

    def _init_weight_(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

    def forward(self, x):

        # out = self.enc_embedding(x)
        out = self.linear(x)

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        # out = self.norm(out)
        # out = self.relu(out)
        out = self.maxpool(out)

        if self.training:
            out = self.dropout1(out)
        
        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        # out = self.norm(out)
        # out = self.relu(out)
        out = self.maxpool(out)

        if self.training:
            out = self.dropout2(out)
            
        out = self.avgpool(out.transpose(1,2))
        out = torch.flatten(out,1) # 32 128

        if self.training:
            out = self.dropout3(out)
        out = self.head(out)
        return out
