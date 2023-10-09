import json
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 窗口内归一化
# def load_data(data_path,mode,win_size):
#     dataset = []
#     labels = []
#     for filename in os.listdir(data_path):
#         if filename.split(".")[0].endswith(mode):
#             data = np.load(data_path+"//"+filename,allow_pickle=True)
#             if data.shape[0] % win_size:
#                 data = np.concatenate([data,data[:win_size-data.shape[0]%win_size]],axis=0)
#             # np.random.shuffle(data)
#             for i in range(data.shape[0]//win_size):
#                 scancel = StandardScaler()
#                 scancel_value = scancel.fit_transform(data[i*win_size:(i+1)*win_size])
#                 dataset.append(scancel_value)
#                 label = filename.split(".")[0]
#                 labels.append(label[:-6] if mode == "Train" else label[:-5])
#                 # 每个窗口对应一个标签
#     dataset = np.array(dataset)
#     labels = np.array(labels)
#     return dataset,labels

# 整体数据归一化
# def load_data(data_path,mode,win_size):
#     dataset = []
#     labels = []
#     for filename in os.listdir(data_path):
#         if filename.split(".")[0].endswith(mode):
#             data = np.load(data_path+"//"+filename,allow_pickle=True)
#             if data.shape[0] % win_size:
#                 data = np.concatenate([data,data[:win_size-data.shape[0]%win_size]],axis=0)
#             # 是否随机打乱数据
#             # if mode == "Train":
#                 # np.random.shuffle(data)
#             np.random.shuffle(data)
#             # 整体归一化
#             scancel = StandardScaler()
#             data = scancel.fit_transform(data)
            
#             #再划分窗口
#             for i in range(data.shape[0]//win_size):
#                 dataset.append(data[i*win_size:(i+1)*win_size])
#                 label = filename.split(".")[0]
#                 labels.append(label[:-6] if mode == "Train" else label[:-5])
#                 # 每个窗口对应一个标签
#     dataset = np.array(dataset)
#     labels = np.array(labels)
#     return dataset,labels

def load_data(data_path,mode,win_size):
    dataset = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.split(".")[0].endswith(mode):
            print("正在加载{}数据集".format(filename))
            data = np.load(data_path+"//"+filename,allow_pickle=True)
            dataset += data.tolist()
            labels += [filename.split(".")[0][:-6] if mode == "Train" else filename.split(".")[0][:-5]] * data.shape[0]
            # 每个窗口对应一个标签
    dataset = np.array(dataset)
    labels = np.array(labels)
    # 随机打乱
    #index = np.arange(dataset.shape[0])
    #np.random.shuffle(index)
    #dataset = dataset[index]
    #labels = labels[index]
    return dataset,labels

def processdata(data_path,mode,win_size):
    dataset,labels = load_data(data_path,mode,win_size)
    # 对数据进行上采样
    # if mode == "Train":
    #     from imblearn.over_sampling import SMOTE
    #     if data_path.split("/")[-1].startswith("BOT_IOT"):
    #         sm = SMOTE(sampling_strategy = {"data_theft":30000}, random_state=42,k_neighbors=2)
    #         data, label = sm.fit_resample(dataset.reshape(dataset.shape[0],-1), labels)
    #         dataset = data.reshape(-1,dataset.shape[-2],dataset.shape[-1])
    #         labels = label

    # 对标签使用one-hot 编码
    one_hot_encoder = OneHotEncoder(sparse=False)
    label_encoder = one_hot_encoder.fit_transform(labels.reshape((-1,1)))

    # 标签 ： 数量
    labels_name,labels_num_count = np.unique(labels, return_counts=True)

    return dataset,label_encoder,labels_name,labels_num_count

class Dataset(object):
    def __init__(self,dataset,labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        return self.dataset[index],self.labels[index]
    
def get_dataloader(data_path,batch_size,win_size,mode):
    dataset,labels,labels_name,labels_num_count = processdata(data_path,mode,win_size)
    dataset = Dataset(dataset,labels)
    if mode == "Train":
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return dataloader,labels_name,labels_num_count



