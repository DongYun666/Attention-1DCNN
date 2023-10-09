import torch
import torch.nn as nn
from models.Attention.Attention import FullAttention
from models.Attention.CosineAttention import CosinAttention
from models.AttentionLayer.WaveletAttentionLayer2 import WaveletAttention

from models.MLP import MLP
from models.encoder import TrafficADEncoder, TrafficADEncoderLayer
from models.embed import DataEmbedding

#未实现
class WaveletADFormer2(nn.Module):
    def __init__(self,attention_type, feature_num, e_layers, num_heads, num_classes, win_size, device, dropout = 0.1,d_model = 512, wavelet_method = 'db4',trainable = True):
        super(WaveletADFormer2, self).__init__()

       
    def forward(self, x):

        return out


