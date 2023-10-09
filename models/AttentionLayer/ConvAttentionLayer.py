import torch.nn as nn
from einops import rearrange

class ConvAttentionLayer(nn.Module):
    def __init__(self,attention,d_model,num_heads,kernel_size,padding):
        super(ConvAttentionLayer,self).__init__()
        self.inner_attention = attention
        self.num_heads = num_heads
        self.conv_poj_q = nn.Conv1d(d_model,d_model*num_heads,kernel_size=kernel_size,padding=padding,bias = True)
        self.conv_poj_k = nn.Conv1d(d_model,d_model*num_heads,kernel_size=kernel_size,padding=padding,bias = True)
        self.conv_poj_v = nn.Conv1d(d_model,d_model*num_heads,kernel_size=kernel_size,padding=padding,bias = True)
        self.proj_out = nn.Linear(d_model*num_heads,d_model)

    def forward(self,x):
        b, w, _ = x.shape

        x = x.transpose(1,2)
        query = rearrange(self.conv_poj_q(x).permute(0,2,1),'b w (h l) -> b h w l',h = self.num_heads)
        key = rearrange(self.conv_poj_k(x).permute(0,2,1),'b w (h l) -> b h w l',h = self.num_heads)
        value = rearrange(self.conv_poj_v(x).permute(0,2,1),'b w (h l) -> b h w l',h = self.num_heads)

        att_out = self.inner_attention(query,key,value) # b h w l
        out = att_out.transpose(1,2).contiguous().view(b,w,-1)

        return self.proj_out(out)