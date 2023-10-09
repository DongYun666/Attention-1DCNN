import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class WaveletAttention(nn.Module):
    def __init__(self,attention,L, level, wavename,d_model,device,trainable = False):
        super(WaveletAttention, self).__init__()
        self.attention = attention
        self.level = level
        wavelet = pywt.Wavelet(wavename)
        h = torch.tensor(wavelet.dec_hi,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        g = torch.tensor(wavelet.dec_lo,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        self.w_dec_filter = nn.Parameter(self.get_dec_filter(h, L),requires_grad=trainable)  # 生成小波滤波器的系数矩阵
        self.v_dec_filter = nn.Parameter(self.get_dec_filter(g, L),requires_grad=trainable)  # 生成小波滤波器的低通分解系数矩阵
        self.w_rec_filter = nn.Parameter(self.get_rec_filter(h, L),requires_grad=trainable)
        self.v_rec_filter = nn.Parameter(self.get_rec_filter(g, L),requires_grad=trainable)

        self.proj_out = nn.Linear(d_model*(level+1),d_model)

        # self.weight = nn.Parameter(torch.tensor([0.8,0.2,2],dtype=torch.float64).reshape(1,level,1,1))
        self.weight = nn.Parameter(torch.ones((1,level + 1,1,1)))


    def get_dec_filter(self, wavelet, L):
        wavelet_len = wavelet.shape[0]
        filter = torch.zeros(self.level,L,L,device=wavelet.device)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t - 2**j * wl, L)
                hl = torch.zeros(L)
                for i,idx in enumerate(index):
                    hl[idx] = wavelet[i]
                filter[j][t] = hl
        return filter # 返回滤波器的系数矩阵 [level,L,L]
    

    def get_rec_filter(self, wavelet, L):
        wavelet_len = wavelet.shape[0]
        filter = torch.zeros(self.level, L, L, device=wavelet.device)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t + 2 ** j * wl, L)
                hl = torch.zeros(L)
                for i, idx in enumerate(index):
                    hl[idx] = wavelet[i]
                filter[j][t] = hl
        return filter  # 返回滤波器的系数矩阵 [level,L,L]
    
    def modwt(self,x):
        B,L,D = x.shape
        x = x.permute(0,2,1)
        v_j = x
        v = []                                  # 用于存储小波变换的结果
        for j in range(self.level):             # 遍历小波分解的层数
            v_j = torch.einsum('ml,bdl->bdm', self.v_dec_filter[j], v_j) # 小波变换的公式 将上一层的结果与下采样小波滤波器的系数矩阵卷积 ，得到当前层的结果
            v.append(v_j)                       # 将小波变换的结果存储到v中
        v = torch.stack(v, dim=2)  # (B, D, level, L) 将v中的结果转换为一个四维的张量
        v_prime = torch.cat([x.reshape(B, D, 1, L), v[..., :-1, :]], dim=2)  # (B, D, level, L) 将v中的结果与原始信号进行拼接
        w = torch.einsum('jml,jbdl->bdjm', self.w_dec_filter, v_prime.permute(2, 0, 1, 3)) # (B, D, level, L) 将v_prime与小波滤波器的系数矩阵卷积
        wavecoeff = torch.cat([w, v[..., -1, :].reshape(B, D, 1, L)], dim=2)  # (B, D, level + 1, L) 将w与v中的最后一层结果进行拼接
        return wavecoeff.permute(0, 2, 3, 1)  # (B, level + 1, L, D)
    
    def imodwt(self,wave):
        wave = wave.permute(0, 3, 1, 2)
        w_rec_filter = self.w_rec_filter.to(wave)  # (level, L, L)
        v_rec_filter = self.v_rec_filter.to(wave)  # (level, L, L)
        w = wave[..., :-1, :]  # (B, D, level, L)
        v_j = wave[..., -1, :]  # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(2)  # (B, D, 1, L)
        for j in range(self.level)[::-1]:
            detail_j = torch.einsum('ml,bdrl->bdrm', w_rec_filter[j], w[..., j, :].unsqueeze(2))
            scale_cat = torch.cat([detail_j, scale_j], dim=2)
            scale_j = torch.einsum('bdrl->bdl', scale_cat)
            if j > 0:
                scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[j - 1], scale_j).unsqueeze(2)  # (B, D, 1, L)
        return scale_j  # (B, D, L)
    
    def forward(self, x):
        b,w,_ = x.shape
        query = self.modwt(x)
        key = self.modwt(x)
        value = self.modwt(x)
        attn_out = self.attention(query, key, value)

        out = self.imodwt(attn_out)
        return out.permute(0, 2, 1)
    
        # out = torch.mean(self.weight * attn_out,dim = 1)
        # return out 


        # out = torch.mean(attn_out,dim = 1)
        # return out

    
    

