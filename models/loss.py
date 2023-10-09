import math
import torch
from torch import nn
from torch.nn import functional as F

# 当前使用
class  CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def forward(self,predict,target):
        B, W = predict.shape
        predict = F.softmax(predict, dim=-1) # softmmax获取预测概率 
        logsoftmax = torch.log(predict)
        loss = torch.mean(-torch.sum(target * logsoftmax,dim=-1))
        return loss

class CrossEntropyLoss2(nn.Module):
    def __init__(self):            
        super(CrossEntropyLoss2, self).__init__()
    def forward(self,predict,target):
        B, W = predict.shape       
        p = F.softmax(predict, dim=-1) # softmmax获取预测概率
        # p = torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)
        loss = -torch.log(p) * target -torch.log((1 - p)) * (1 - target)
        loss = torch.mean(torch.sum(loss,dim=-1))
        return loss                
                                    
    

class CrossEntropyLoss3(nn.Module):
    def __init__(self,labels_num_count,device):
        super(CrossEntropyLoss3, self).__init__()
        self.device = device
        length = len(labels_num_count)
        # penalty_matrix = [[0] * length for _ in range(length)]
        penalty_matrix = torch.zeros(length,length,device=device)
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1
                else:
                    # penalty_matrix[i][j] = min(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
                    penalty_matrix[i][j] = labels_num_count[i]/(labels_num_count[i]+labels_num_count[j]) 

        # self.penalty_matrix = torch.tensor(penalty_matrix,device=device)
        self.register_buffer('penalty_matrix', penalty_matrix)
    def forward(self,predict,target):
        B, W = predict.shape
        p = F.softmax(predict, dim=-1) # softmmax获取预测概率
        # p = torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)
        
        # 将p转换为one-hot编码
        p_onehot = torch.zeros_like(p)
        p_onehot.scatter_(1, p.argmax(dim=-1, keepdim=True), 1)
        
        target_res = target.argmax(dim=-1).view(-1,1)

        alpha = torch.empty((B,W),device=target.device)
        
        for i in range(W):
            temp = torch.full((B,1),i,dtype=torch.long,device=target.device)
            alpha[:,i] = self.penalty_matrix[target_res.squeeze(),temp.squeeze()]

        # mask = (target == p_onehot).all(dim=1)
        # resoult_True = torch.zeros_like(target)
        # resoult_True[mask] = target[mask]
        # resoult_False = torch.zeros_like(target)
        # resoult_False[~mask] = 1-target[~mask]

        # loss = -torch.log(p) * resoult_True - torch.log(1-p) * resoult_False * (torch.exp(1-p*alpha) - 1)


        # loss = -torch.log(p) * target * (torch.exp(1-p*alpha) - 1) * torch.tensor(1/1.7,device=self.device)
        
        # loss = -torch.log(p) * target - torch.log(1-p) * (1-target) * torch.exp(1-p*alpha)
        loss = -torch.log(p) * target * (torch.exp(1 - p*alpha) - 1)
        # loss = -torch.log(p) * target * (torch.exp((1 - p*alpha) / torch.log(torch.tensor(2,device=self.device))))

        # loss = -torch.log(p) * target * alpha

        # loss1 = -torch.log(p) * target 
        # loss2 = -torch.log((1 - p)) * (1 - target) 
        # loss2 = loss2 * (torch.exp(1- p * alpha) - 1)
        # loss = loss1 + loss2
        loss = torch.mean(torch.sum(loss,dim=-1))

        return loss


class CrossEntropyLossWeight(nn.Module):
    def __init__(self,labels_num_count,device):
        super(CrossEntropyLossWeight,self).__init__()

        length = len(labels_num_count)
        penalty_matrix = [[0] * length for _ in range(length)]
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1
                else:
                    penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

    def forward(self,predict,target):
        B, W = predict.shape
        logpredict = torch.log(torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)) 
        
        target_ids = target.argmax(dim=-1).tolist()
        predict_ids = predict.argmax(dim=-1).tolist()
        alpha = []
        for i in range(len(target_ids)):
            alpha.append(self.penalty_matrix[target_ids[i]][predict_ids[i]])
        alpha = torch.tensor(alpha,device=target.device).view(-1,1)
        
        # target 传入的是one-hot编码
        logpredict_ = -torch.sum(target * logpredict,dim=-1)
        # 对每一个样本计算惩罚矩阵
        target_ = torch.argmax(target,dim=-1)

        penalty_matrix_ = self.penalty_matrix[target_]
        loss = torch.mean(torch.sum(logpredict_ * penalty_matrix_,dim=-1))
        return loss

class CrossEntropyLossWeight2(nn.Module):
    def __init__(self,labels_num_count,device):
        super(CrossEntropyLossWeight2,self).__init__()

        length = len(labels_num_count)
        penalty_matrix = [[0] * length for _ in range(length)]
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1
                else:
                    penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

    def forward(self,predict,target):
        B, W = predict.shape
        logpredict = torch.log(torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)) 
        
        target_ids = target.argmax(dim=-1).tolist()
        predict_ids = predict.argmax(dim=-1).tolist()
        alpha = []
        for i in range(len(target_ids)):
            alpha.append(self.penalty_matrix[target_ids[i]][predict_ids[i]])
        alpha = torch.tensor(alpha,device=target.device).view(-1,1)
        
        # target 传入的是one-hot编码
        logpredict_ = -torch.sum(target * logpredict,dim=-1)

        loss = torch.mean(torch.sum(logpredict_ * alpha,dim=-1))
        return loss

# 错误 
class CrossEntropyLossWeight3(nn.Module):
    def __init__(self,labels_num_count,device):
        super(CrossEntropyLossWeight3,self).__init__()

        length = len(labels_num_count)
        penalty_matrix = [[0] * length for _ in range(length)]
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1
                else:
                    penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

    def forward(self,predict,target):
        B, W = predict.shape
        logpredict = torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)  # logsoftmax

        preids = predict.argmax(dim=-1)
        tarids = target.argmax(dim=-1)
        predict_onehot = torch.zeros_like(predict)
        predict_onehot.scatter_(1, predict.argmax(dim=-1, keepdim=True), 1) 
        for i in range(len(preids)):
            if preids[i] != tarids[i]:
                predict_onehot[i][preids[i]] = torch.tensor(self.penalty_matrix[tarids[i]][preids[i]])
            else:
                predict_onehot[i][preids[i]] = torch.tensor(0)
        #使用 predict_onehot 作为 target

        loss = torch.mean(torch.sum(predict_onehot * logpredict,dim=-1))
        return loss


class costloss(nn.Module):
    def __init__(self,labels_num_count,device):
        super(costloss,self).__init__()
        self.correct_loss = CrossEntropyLoss()
        self.error_loss = CrossEntropyLossWeight2(labels_num_count,device)
    def forward(self,predict,target):
        res = (torch.max(predict,dim=-1,keepdim=True)[0] == predict).int()
        correct = (res == target).all(dim=1).nonzero()[:, 0]
        error = (res != target).any(dim=1).nonzero()[:, 0]
        correct_loss = self.correct_loss(predict[correct],target[correct])
        error_loss = self.error_loss(predict[error],target[error])
        return correct_loss + error_loss






# 添加惩罚矩阵的focal 损失
class penalty_focal_loss(nn.Module):

    def __init__(self,labels_num_count,device,gamma=2):
        super(penalty_focal_loss,self).__init__()
        self.gamma = gamma
        penalty_matrix = []
        labels_sum = sum(labels_num_count)
        for i in range(len(labels_num_count)):
            # penalty_matrix.append(labels_num_count[i]/labels_sum)
            penalty_matrix.append(1/labels_num_count[i])

        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

    def forward(self,predict,target):
        pt = F.softmax(predict, dim=-1) # softmmax获取预测概率
        ids = target.argmax(dim=-1).view(-1, 1) 
        # alpha = self.penalty_matrix[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        alpha = self.penalty_matrix[ids.data.view(-1)].view(-1,1)
        probs = (pt * target).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), 2)) * log_p

        return loss.mean()
    

class penalty_focal_loss2(nn.Module):
    def __init__(self,labels_num_count,device,gamma=2):
        super(penalty_focal_loss2,self).__init__()
        self.gamma = gamma
        length = len(labels_num_count)
        penalty_matrix = [[0] * length for _ in range(length)]
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1 
                else:
                    penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
                    # penalty_matrix[i][j] = labels_num_count[i]/(labels_num_count[i]+labels_num_count[j])

        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)
        # self.register_buffer('penalty_matrix', penalty_matrix)


    def forward(self,predict,target):
        pt = F.softmax(predict, dim=-1) # softmmax获取预测概率
        target_ids = target.argmax(dim=-1).tolist()
        predict_ids = predict.argmax(dim=-1).tolist()
        alpha = []
        for i in range(len(target_ids)):
            alpha.append(self.penalty_matrix[target_ids[i]][predict_ids[i]])
        alpha = torch.tensor(alpha,device=target.device).view(-1,1)
        probs = (pt * target).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), 2)) * log_p
        # print(self.penalty_matrix)
        return loss.mean()
    
# class penalty_focal_loss3(nn.Module):
#     def __init__(self,labels_num_count,device,gamma=2):
#         super(penalty_focal_loss3,self).__init__()
#         self.gamma = gamma
#         self.alpha = 0.25


#     def forward(self,preds,labels):
#         labels = labels.argmax(dim=-1)
#         bce_loss = F.cross_entropy(preds, labels) 
#         probs = torch.softmax(preds, dim=1)
#         class_mask = labels.bool()
#         probs = probs[class_mask]
#         labels = labels[class_mask]
        
#         alpha = torch.ones(labels.shape[0]) * self.alpha
        
#         # 从类别不平衡角度对正样本损失加权
#         alpha = torch.where(labels, alpha, 1-alpha)
#         probs = probs + 1e-8 # 防止0导致nan
#         focal_loss = -alpha * (1-probs)**self.gamma * probs.log() 
#         return focal_loss.mean()
    
class penalty_focal_arcloss(nn.Module):
    def __init__(self, labels_num_count, emb_size,device, s=100.0, m=1, gamma=2, easy_margin = False):
        super(penalty_focal_arcloss, self).__init__()
        
        self.gamma = gamma
        num_classes = len(labels_num_count)

        self.num_classes = num_classes
        self.emb_size = emb_size
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin
        self.loss = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.randn(num_classes,emb_size,device=device))
        nn.init.xavier_uniform_(self.weight)

        penalty_matrix = [[0] * num_classes for _ in range(num_classes)]
        for i in range(len(labels_num_count)):
            for j in range(len(labels_num_count)):
                if i == j:
                    penalty_matrix[i][j] = 1
                else:
                    penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
        self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

    def forward(self, predict, target):
        B, W = predict.shape

        predict = F.normalize(predict,p = 2, dim=1)

        weights = F.normalize(self.weight,p = 2, dim=1)

        cosine = F.linear(predict, weights)
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th,phi,cosine - self.mm)
        
        logits = self.s * (target * phi + (1.0 - target) * cosine)
        
        target = target.argmax(dim=-1)

        return self.loss(logits,target)
        # logsoftmax = torch.log(torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1).reshape(B,1))

        # loss = torch.mean(-torch.sum(logsoftmax,dim=-1))

        # return loss

        # pt = F.softmax(logits, dim=-1) # softmmax获取预测概率
        # target_ids = target.argmax(dim=-1).tolist()
        # logits_ids = logits.argmax(dim=-1).tolist()
        # alpha = []
        # for i in range(len(target_ids)):
        #     alpha.append(self.penalty_matrix[target_ids[i]][logits_ids[i]])
        
        # alpha = torch.tensor(alpha,device=target.device).view(-1,1)
        # probs = (pt * target).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        # log_p = probs.log()
        # # 同样，原始ce上增加一个动态权重衰减因子
        # loss = -alpha * (torch.pow((1 - probs), 2)) * log_p
        
        # loss = torch.mean(-torch.sum(loss,dim=-1))

        # return loss
    

class CEFLoss(nn.Module):
    def __init__(self,labels_num_count,device):
        super(CEFLoss,self).__init__()
        self.correct_loss = CrossEntropyLoss()
        self.error_loss = penalty_focal_loss2(labels_num_count,device)
    def forward(self,predict,target):
        res = (torch.max(predict,dim=-1,keepdim=True)[0] == predict).int()
        correct = (res == target).all(dim=1).nonzero()[:, 0]
        error = (res != target).any(dim=1).nonzero()[:, 0]
        correct_loss = self.correct_loss(predict[correct],target[correct])
        error_loss = self.error_loss(predict[error],target[error])
        return correct_loss + error_loss


