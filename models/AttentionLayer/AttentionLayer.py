import torch
import torch.nn as nn

from einops import rearrange

class LinearAttentionLayer(nn.Module):
    def __init__(self,attention,d_model,num_heads):
        super(LinearAttentionLayer,self).__init__()
        self.inner_attention = attention
        self.num_heads = num_heads
        self.linear_poj_q = nn.Linear(d_model,d_model*num_heads)
        self.linear_poj_k = nn.Linear(d_model,d_model*num_heads)
        self.linear_poj_v = nn.Linear(d_model,d_model*num_heads)
        self.proj_out = nn.Linear(d_model*num_heads,d_model)

    def forward(self,x):
        b, w, _ = x.shape
        query = rearrange(self.linear_poj_q(x),'b w (h l) -> b h w l',h = self.num_heads)
        key = rearrange(self.linear_poj_k(x),'b w (h l) -> b h w l',h = self.num_heads)
        value = rearrange(self.linear_poj_v(x),'b w (h l) -> b h w l',h = self.num_heads)
        att_out = self.inner_attention(query,key,value) # b h w l
        out = att_out.transpose(1,2).contiguous().view(b,w,-1)
        return self.proj_out(out)