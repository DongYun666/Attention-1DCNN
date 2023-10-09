import torch
import torch.nn as nn

from math import sqrt
from torch.nn import functional as F




class FullAttention(nn.Module):
    def __init__(self):
        super(FullAttention, self).__init__()

    def forward(self, queries, keys, values):
        B, H, W, L = queries.shape
        _, _, S, D = values.shape
        scale = 1. / sqrt(L)
        scores = torch.einsum("bhle,bhse->bhls", queries, keys)
        weight = torch.softmax(scale * scores, dim=-1)
        out = torch.einsum("bhls,bhsd->bhld", weight, values)
        return out





