import torch
import torch.nn as nn
from torch.nn import functional as F

class CosinAttention(nn.Module):
    def __init__(self,num_heads,device):
        super(CosinAttention, self).__init__()
        self.device = device
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1),device=self.device)), requires_grad=True)
    def forward(self, queries, keys, values):
        B, H, W, L = queries.shape
        _, _, S, D = values.shape
        # 使用Swin中提出的方法
        logit_scale = torch.exp(torch.clamp(self.logit_scale, max=torch.log(torch.sqrt(torch.tensor(L,device=self.device)))))
        # cosins 相似度
        attn = (F.normalize(queries.contiguous(), dim=-1) @ F.normalize(keys.contiguous(), dim=-1).transpose(-2, -1))  # 先计算P范式
        attn = attn * logit_scale

        A = torch.softmax(attn, dim=-1)
        V = (A @ values.contiguous())
        return V.contiguous()