import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, targets):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weights, p=2, dim=1)
        
        # Compute cosine similarity between embeddings and weights
        cosine_sim = F.linear(embeddings, weights)
        
        # Compute the theta value (angle)
        theta = torch.acos(cosine_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        
        # Compute the arc margin
        marginal_theta = theta + self.margin
        
        # Compute the modified cosine and sine values
        cos_m = torch.cos(marginal_theta)
        sin_m = torch.sin(marginal_theta)
        
        # Apply the modified cosine similarity
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        
        logits = self.scale * (one_hot * cos_m - cosine_sim * sin_m)
        
        # Compute the final cross-entropy loss
        loss = F.cross_entropy(logits, targets)
        
        return loss


# Generate some dummy data
num_classes = 5
emb_size = 64
batch_size = 32
num_samples = 1000

features = torch.randn(num_samples, emb_size) # 生成正态分布的数据 1000*64
targets = torch.randint(0, num_classes, (num_samples,))

# Create the loss function
loss_fn = ArcFaceLoss(emb_size, num_classes)
loss = loss_fn(features, targets)