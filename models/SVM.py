import torch
import torch.nn as nn
from models.MLP import MLP

from models.embed import DataEmbedding


class SVM(nn.Module):
    def __init__(self,feature_num,e_layers,d_model,num_classes,device,dropout=0.1):
        super(SVM, self).__init__()
        self.enc_embedding = DataEmbedding(feature_num, d_model,device, dropout)

        self.hidden1 = nn.Linear(d_model, 2*d_model,bias=True)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.hidden2 = nn.Linear(2*d_model, d_model,bias=True)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.head = MLP(d_model, num_classes, dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.apply(self._init_weight_)

    def _init_weight_(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)


    def forward(self, x):
        
        x = self.enc_embedding(x)   # 32 50 128

        out = self.hidden1(x)            
        out = self.act1(out)
        if self.training:
            out = self.dropout1(out)

        out = self.hidden2(out)
        out = self.act2(out) + x
        if self.training:
            x = self.dropout2(out)

        out = self.avgpool(out.transpose(1,2))  # 32 128 1

        out = torch.flatten(out,1) # 32 128

        out = self.head(out)
        return out

