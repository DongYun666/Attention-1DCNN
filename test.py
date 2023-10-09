# # from scapy.all import *

# # path = "H://数据集//BoT_IoT数据集//BOT-IOT-pcap//test"
# # flows = {}
# # for filename in os.listdir(path):
# #     if filename.endswith(".pcap"):
# #         filepath = os.path.join(path, filename)
# #         packets = rdpcap(filepath)
# #         for packet in packets:
# #             if "a" in flows:
# #                 flows["a"].append(packet)
# #             else:
# #                 flows["a"]=[packet]
# # for packets in flows.values():
# #     packets = sorted(packets, key=lambda x: -len(x))
# #     # print(len(packets))
# #     print(type(packets))


# # import os
# # from scapy.all import *
# # path = "H://数据集//TOR//TorPcaps//Pcaps//tor"
# # savepath = "H://数据集//TOR//TorPcaps//整合数据集"

# # for file in os.listdir(path):
# #     filepath = os.path.join(path,file)
# #     print("正在处理文件：",filepath)
# #     packets = []
# #     for pcapfile in os.listdir(filepath):
# #         pcapfilepath = os.path.join(filepath,pcapfile)
# #         packets += rdpcap(pcapfilepath)
# #     # 保存文件

# #     wrpcap(savepath+"//"+file+".pcap",packets)
# #     print("保存文件：",savepath+"//"+file+".pcap")
# #     # 打印处理完毕
# # print("文件处理完毕")

# # from scapy.all import *
# # from scapy.layers.inet import TCP, UDP
# # path = "F://数据集//dataset//BOT_IOT//整合//zhenghe.pcap"
# # packets = rdpcap(path)
# # for packet in packets:
# #     if packet.haslayer(TCP) and packet.payload:
# #         print(packet[TCP].payload)
# #     elif packet.haslayer(UDP) and packet.payload:
# #         print(packet[UDP].payload)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class penalty_focal_loss2(nn.Module):
#     def __init__(self,labels_num_count,device,gamma=2):
#         super(penalty_focal_loss2,self).__init__()
#         self.gamma = gamma
#         length = len(labels_num_count)
#         penalty_matrix = [[0] * length for _ in range(length)]
#         for i in range(len(labels_num_count)):
#             for j in range(len(labels_num_count)):
#                 if i == j:
#                     penalty_matrix[i][j] = 1
#                     # penalty_matrix[i][j] = 0
#                 else:
#                     penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
#         self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

#     def forward(self,predict,target):
#         pt = F.softmax(predict, dim=-1) # softmmax获取预测概率
#         target_ids = target.argmax(dim=-1).tolist()
#         predict_ids = predict.argmax(dim=-1).tolist()
#         # alpha = self.penalty_matrix[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
#         # alpha = self.penalty_matrix[ids.data.view(-1)].view(-1,1)
#         alpha = []
#         for i in range(len(target_ids)):
#             alpha.append(self.penalty_matrix[target_ids[i]][predict_ids[i]])
#         alpha = torch.tensor(alpha,device=target.device).view(-1,1)
            
#         probs = (pt * target).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
#         log_p = probs.log()
#         # 同样，原始ce上增加一个动态权重衰减因子
#         loss = -alpha * (torch.pow((1 - probs), 2)) * log_p
#         return loss.mean()
    

# class CrossEntropyLossWeight(nn.Module):
#     def __init__(self,labels_num_count,device):
#         super(CrossEntropyLossWeight,self).__init__()

#         length = len(labels_num_count)
#         penalty_matrix = [[0] * length for _ in range(length)]
#         for i in range(len(labels_num_count)):
#             for j in range(len(labels_num_count)):
#                 if i == j:
#                     penalty_matrix[i][j] = 1
#                 else:
#                     penalty_matrix[i][j] = max(labels_num_count[i],labels_num_count[j])/(labels_num_count[i]+labels_num_count[j])
#         self.penalty_matrix = torch.tensor(penalty_matrix,device=device)

#     def forward(self,predict,target):
#         B, W = predict.shape
#         logpredict = torch.log(torch.exp(predict) / torch.sum(torch.exp(predict), dim=-1).reshape(B,1)) 
        
#         target_ids = target.argmax(dim=-1).tolist()
#         predict_ids = predict.argmax(dim=-1).tolist()
#         alpha = []
#         for i in range(len(target_ids)):
#             alpha.append(self.penalty_matrix[target_ids[i]][predict_ids[i]])
#         alpha = torch.tensor(alpha,device=target.device).view(-1,1)

#         # target 传入的是one-hot编码
#         logpredict_ = -torch.sum(target * logpredict,dim=-1)

#         loss = torch.mean(torch.sum(logpredict_ * alpha,dim=-1))
#         return loss


# if __name__ == "__main__":
#     predict = torch.tensor([
#         [0.1, 0.8, 0.3,0.1],
#         [0.1, 0.2, 0.9,0.1],
#         [0.8, 0.3, 0.7,0.2],
#     ], dtype=torch.float32)
#     target = torch.tensor(
#         [[0,1,0,0],
#         [0,0,1,0],
#         [0,1,0,0]]
#     )
#     class_num = 4
#     loss = CrossEntropyLossWeight([20,40,100,10],torch.device("cpu"))
#     print(loss(predict,target))


from matplotlib import pyplot as plt

with open('VPN_NOVPN_length.txt', 'r') as f:
    data = [int(line.strip()) for line in f.readlines()]
# 求出现在各个范围内的个数
y = [0 for i in range(0, 2000, 50)]
# 统计 0-3000 之间的数据
for i in data:
    if i < 2000:
        y[i//50] += 1
    else:
        y[-1] += 1
# 求解频率
y = [i/sum(y) for i in y]
# 绘制直方图
x = [i for i in range(0, 2000, 50)]

plt.bar(x, y, width=20)

plt.xticks([0,100,200,300,400,500,1000,1500,2000],rotation = 90)
# 设置显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.ylabel('概率质量函数')
# plt.xlabel('数据包长度')

plt.show()
print(x)
print(y)

# fig, ax = plt.subplots()
# b = ax.bar(x, y)
# plt.title('Recommended song list score')
# for a, b in zip(x, y):
#     ax.text(a, b+1, b, ha='center', va='bottom')

# plt.xlim((0,30))
# plt.ylim((0,1))
# plt.xticks(range(len(x)+2))
# plt.xlabel('playlist number')
# plt.ylabel('score')
# plt.legend()
# plt.show()



# x = range(1,11)
# y = [84,87,78,93,26,88,74,92,69,86]
# fig, ax = plt.subplots()
# # 截尾平均数
# means = sum(sorted(y)[1:-1])/len(y[1:-1])
# b = ax.bar(x, y, label='{}'.format(means))
# plt.title('Recommended song list score')
# for a, b in zip(x, y):
#     ax.text(a, b+1, b, ha='center', va='bottom')

# plt.xlim((1,10))
# plt.ylim((1,100))
# plt.xticks(range(len(x)+2))
# plt.xlabel('playlist number')
# plt.ylabel('score')
# plt.legend()
# plt.show()
