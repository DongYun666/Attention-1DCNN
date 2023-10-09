import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ModwtFunction(Function):
    @staticmethod
    def forward(ctx, x, w_dec_filter, v_dec_filter, w_rec_filter,v_rec_filter, level):
        ctx.save_for_backward(w_dec_filter,v_dec_filter, w_rec_filter, v_rec_filter)
        B,L,D = x.shape
        x = x.permute(0,2,1)
        v_j = x
        v = []                                  # 用于存储小波变换的结果
        for j in range(level):             # 遍历小波分解的层数
            v_j = torch.einsum('ml,bdl->bdm', v_dec_filter[j], v_j) # 小波变换的公式 将上一层的结果与下采样小波滤波器的系数矩阵卷积 ，得到当前层的结果
            v.append(v_j)                       # 将小波变换的结果存储到v中
        v = torch.stack(v, dim=2)  # (B, D, level, L) 将v中的结果转换为一个四维的张量
        v_prime = torch.cat([x.reshape(B, D, 1, L), v[..., :-1, :]], dim=2)  # (B, D, level, L) 将v中的结果与原始信号进行拼接
        w = torch.einsum('jml,jbdl->bdjm', w_dec_filter, v_prime.permute(2, 0, 1, 3)) # (B, D, level, L) 将v_prime与小波滤波器的系数矩阵卷积
        wavecoeff = torch.cat([w, v[..., -1, :].reshape(B, D, 1, L)], dim=2)  # (B, D, level + 1, L) 将w与v中的最后一层结果进行拼接
        return wavecoeff.permute(0, 2, 3, 1)  # (B, level + 1, L, D)

    @staticmethod
    def backward(ctx, grad_input):
        w_dec_filter,v_dec_filter, w_rec_filter, v_rec_filter = ctx.saved_variables
        grad_input = grad_input.permute(0, 3, 1, 2) # (B, D, level + 1, L)
        w = grad_input[..., :-1, :]  # (B, D, level, L)
        v_j = grad_input[..., -1, :]  # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(2)  # (B, D, 1, L)
        level = grad_input.shape[2]-1
        for j in range(level)[::-1]:
            detail_j = torch.einsum('ml,bdrl->bdrm', w_rec_filter[j], w[..., j, :].unsqueeze(2))
            scale_cat = torch.cat([detail_j, scale_j], dim=2)
            scale_j = torch.einsum('bdrl->bdl', scale_cat)
            if j > 0:
                scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[j - 1], scale_j).unsqueeze(2)  # (B, D, 1, L)
        grad_output = torch.autograd.Variable(scale_j.permute(0, 2, 1))        
        return grad_output, w_dec_filter,v_dec_filter,w_rec_filter,w_rec_filter,None # (B, L, D)
        # return grad_output, None,None,None,None,None # (B, L, D)


class Wavelet(nn.Module):
    def __init__(self,L, level, wavename,device,trainable = False):
        super(Wavelet, self).__init__()
        self.level = level
        wavelet = pywt.Wavelet(wavename)
        h = torch.tensor(wavelet.dec_hi,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        g = torch.tensor(wavelet.dec_lo,dtype=torch.float, device=device) / torch.sqrt(torch.tensor(2.0,dtype=torch.float, device=device))
        self.w_dec_filter = nn.Parameter(self.get_dec_filter(h, L),requires_grad=trainable)  # 生成小波滤波器的系数矩阵
        self.v_dec_filter = nn.Parameter(self.get_dec_filter(g, L),requires_grad=trainable)  # 生成小波滤波器的低通分解系数矩阵
        self.w_rec_filter = nn.Parameter(self.get_rec_filter(h, L),requires_grad=trainable)
        self.v_rec_filter = nn.Parameter(self.get_rec_filter(g, L),requires_grad=trainable)
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
    
    def forward(self, x):
        return ModwtFunction.apply(x, self.w_dec_filter, self.v_dec_filter, self.w_rec_filter,self.v_rec_filter,self.level)


class WaveletAttention(nn.Module):
    def __init__(self,attention,d_model,num_heads,winsize,wavelet_method,device,trainable):
        super(WaveletAttention,self).__init__()
        self.inner_attention = attention

        self.wave_conv_poj_q = Wavelet(L=winsize,level = num_heads,wavename=wavelet_method,device=device,trainable=trainable)
        self.wave_conv_poj_k = Wavelet(L=winsize,level = num_heads,wavename=wavelet_method,device=device,trainable=trainable)
        self.wave_conv_poj_v = Wavelet(L=winsize,level = num_heads,wavename=wavelet_method,device=device,trainable=trainable)
        
        self.proj_out = nn.Linear(d_model*(num_heads+1),d_model)

    def forward(self,x):
        b, w, _ = x.shape
        query =  self.wave_conv_poj_q(x)
        key = self.wave_conv_poj_k(x)
        value = self.wave_conv_poj_v(x)
        att_out = self.inner_attention(query,key,value)      # b h w l
        out = att_out.transpose(1,2).contiguous().view(b,w,-1)
        return self.proj_out(out)