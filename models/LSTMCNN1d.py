import torch
import torch.nn as nn
from models.MLP import MLP

from models.embed import DataEmbedding


class LSTMCNN1d(nn.Module):
    def __init__(self,feature_num,e_layers,d_model,num_classes,win_size,device,dropout=0.1):
        super(LSTMCNN1d, self).__init__()
        e_layers = 1
        self.e_layers = e_layers
        self.d_model = d_model
        self.enc_embedding = DataEmbedding(feature_num, d_model,device, dropout)

        self.lstm1 = nn.LSTM(input_size=feature_num, hidden_size=d_model, num_layers=e_layers, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=e_layers, batch_first=True, bidirectional=False) 
        self.dropout2 = nn.Dropout(dropout)

        
        self.CNN1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(win_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2) 
        self.CNN2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(d_model//4 ,num_classes,bias=True)

        self.linear1 = nn.Linear(d_model//4 * win_size, 256,bias=True)
        self.dropout3 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(256, num_classes,bias=True)

        self.apply(self._init_weight_)

    def _init_weight_(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        
        # x = self.enc_embedding(x)
        h1 = torch.randn(self.e_layers, x.size(0), self.d_model,requires_grad=True).to(x.device)
        c1 = torch.randn(self.e_layers, x.size(0), self.d_model,requires_grad=True).to(x.device)
        
        h2 = torch.randn(self.e_layers, x.size(0), self.d_model,requires_grad=True).to(x.device)
        c2 = torch.randn(self.e_layers, x.size(0), self.d_model,requires_grad=True).to(x.device)

        out, _ = self.lstm1(x,(h1,c1))
        if self.training:
            out = self.dropout1(out)
        out, _ = self.lstm2(out,(h2,c2))
        if self.training:
            out = self.dropout2(out)
    

        out = self.CNN1(out.transpose(1,2)).transpose(1,2)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.CNN2(out.transpose(1,2)).transpose(1,2)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.avgpool(out.transpose(1,2))
        out = torch.flatten(out,1) # 32 128
        out = self.head(out)

        return out
        