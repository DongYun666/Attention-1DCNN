from einops import rearrange
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# 具体通过modwt 产生QKV ,QKV 视为三级分解后的值

class WaveletAttention(nn.Module):
    def __init__(self,attention,L,num_heads, wavename,d_model,device,trainable = False):
        super(WaveletAttention, self).__init__()
        self.attention = attention
        wavelet = pywt.Wavelet(wavename)
        h = torch.tensor(wavelet.dec_hi,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        g = torch.tensor(wavelet.dec_lo,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        self.trainable = trainable
        self.num_heads = num_heads
        self.w_dec_filter = nn.Parameter(self.get_dec_filter(h, L),requires_grad=self.trainable) # 生成小波滤波器的系数矩阵
        self.v_dec_filter = nn.Parameter(self.get_dec_filter(g, L),requires_grad=self.trainable) # 生成小波滤波器的低通分解系数矩阵


    def get_dec_filter(self, wavelet, L):
        wavelet_len = wavelet.shape[0]
        filter = torch.zeros(3,L,L,device=wavelet.device)
        wl = torch.arange(wavelet_len)
        for j in range(3):
            for t in range(L):
                index = torch.remainder(t - 2**j * wl, L)
                hl = torch.zeros(L)
                for i,idx in enumerate(index):
                    hl[idx] = wavelet[i]
                filter[j][t] = hl
        return filter # 返回滤波器的系数矩阵 [3,L,L]
    
    def modwt(self,x):
        B,L,D = x.shape
        x = x.permute(0,2,1)
        v_j = x
        v = []                                  # 用于存储小波变换的结果
        for j in range(3):             # 遍历小波分解的层数
            v_j = torch.einsum('ml,bdl->bdm', self.v_dec_filter[j], v_j) # 小波变换的公式 将上一层的结果与下采样小波滤波器的系数矩阵卷积 ，得到当前层的结果
            v.append(v_j)                       # 将小波变换的结果存储到v中
        v = torch.stack(v, dim=0)

        w = torch.einsum('jml,jbdl->jbdm', self.w_dec_filter, v) # (B, D, level, L) 将v_prime与小波滤波器的系数矩阵卷积
        
        return w.permute(0,1,3,2)  
    
    def forward(self, x):
        b,w,_ = x.shape
        QKV = self.modwt(x)
        query = rearrange(QKV[0], 'b w (h d) -> b h w d',h = self.num_heads)
        key = rearrange(QKV[1], 'b w (h d) -> b h w d',h = self.num_heads)
        value = rearrange(QKV[2], 'b w (h d) -> b h w d',h = self.num_heads)

        out = self.attention(query, key, value)

        return rearrange(out, 'b h w d -> b w (h d)',h = self.num_heads)
    
    

