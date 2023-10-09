import os
import time
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch import nn

from data_factory.data_loader2 import get_dataloader
from models.BuildModel import build
from models.loss import CEFLoss, CrossEntropyLoss, CrossEntropyLoss2, CrossEntropyLoss3, CrossEntropyLossWeight, CrossEntropyLossWeight2, CrossEntropyLossWeight3, costloss, penalty_focal_arcloss, penalty_focal_loss, penalty_focal_loss2
import warnings
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# from apex.optimizers import FusedAdam
# from torchsummary import summary

warnings.filterwarnings("ignore") 


# 动态调整学习率
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (1 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))

# def adjust_learning_rate(optimizer):
#     lr = optimizer.param_groups[0]["lr"] * 1
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     print("Updating learning rate to {}".format(lr))


# 早停机制 可能需要修改  如果val_loss
class EarlyStopping:
    def __init__(self, usepenalty,patience=7, verbose=False, dataset_name='', delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.var_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.usepenalty = usepenalty

    def __call__(self, val_loss, model, path, num_experiment,model_name,win_size,usepenalty,optimizer,lr,epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment,model_name,win_size,usepenalty,optimizer)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # if self.counter % 5 == 0:
            #     adjust_learning_rate(optimizer)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment,model_name,win_size,usepenalty,optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, num_experiment,model_name,win_size,usepenalty,optimizer):
        if self.verbose:
            print(f'====Validation loss decreased ({self.var_loss_min:.6f} --> {val_loss:.6f}). Saving model====')
        torch.save(model.state_dict(),os.path.join(path, str(model_name)+"_" + str(self.dataset) +"_"+str(win_size)+"_"+str(num_experiment)+"_"+str(usepenalty)+'_checkpoint_.pth'))
        self.var_loss_min = val_loss
        # adjust_learning_rate(optimizer)




# 实验主程序
class Experiment(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Experiment.DEFAULTS, **config)
        self.same_seeds(seed=self.random_seed)
        self.device = torch.device(self.device if self.device != "cpu" else "cpu")

        if self.mode == "Train":
            self.train_loader,self.classes,self.labels_num_count = get_dataloader(data_path=self.data_path,batch_size=self.batch_size, win_size=self.win_size,mode = self.mode)
            self.vali_loader,_,_ = get_dataloader(data_path=self.data_path,batch_size=self.batch_size, win_size=self.win_size,mode = "Test")
        else:
            self.test_loader,self.classes,self.labels_num_count = get_dataloader(data_path=self.data_path,batch_size=self.batch_size, win_size=self.win_size,mode = self.mode)    
        self.num_classes = len(self.classes)


        if self.usepenalty:
            # self.loss = CrossEntropyLossWeight3(self.labels_num_count,self.device)
            # self.loss = CrossEntropyLossWeight2(self.labels_num_count,self.device)
            # self.loss = CrossEntropyLossWeight(self.labels_num_count,self.device)
            # self.loss = costloss(self.labels_num_count,self.device)
            # self.loss = penalty_focal_loss2(self.labels_num_count,self.device,self.gamma)
            # self.loss = penalty_focal_loss3(self.labels_num_count,self.device,self.gamma)
            
            # self.loss = penalty_focal_arcloss(self.labels_num_count, len(self.labels_num_count), device=self.device)
            # self.loss = CEFLoss(self.labels_num_count,self.device)
            self.loss = CrossEntropyLoss3(self.labels_num_count,self.device)
        else:
            self.loss = CrossEntropyLoss()
            # self.loss = CrossEntropyLoss2()


        self.model = build(self.model_name,
                           self.attention_type,
                           self.feature_num,
                           self.e_layers,
                           self.num_heads,
                           self.num_classes,
                           self.win_size,
                           self.device,
                           self.dropout,
                           self.d_model,
                           self.kernel_size,
                           self.padding,
                           self.wavelet_method,
                           self.trainable
                           )
        if self.model == None:
            raise Exception("模型不存在")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2,eps=1e-8)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2,eps=1e-8)
        # self.optimizer = FusedAdam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2,eps=1e-8)
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.model.to(self.device)
        
        # 打印模型结构
        # summary(model = self.model,input_size=(50,42),batch_size=32,device='cuda')


    def same_seeds(self, seed):
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(seed)  # 为当前GPU设置
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = True

    def vali(self,vali_loader):
        self.model.eval()
        outputs = []
        labels = []
        for i, (input_data, label) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            label = label.float().to(self.device)
            output = self.model(input)
            outputs.extend(output.detach().cpu().argmax(dim=-1).tolist())
            labels.extend(label.detach().cpu().argmax(dim=-1).tolist())
        f1 = f1_score(labels,outputs, average='macro')
        accuracy = accuracy_score(labels,outputs)
        return accuracy,f1
            
    def train(self):
        print('====================Train  model======================')
        time_now = time.time()
        path = self.model_save_path

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        # 需要修改
        early_stopping = EarlyStopping(usepenalty= self.usepenalty,patience=self.patience, verbose=True, dataset_name=self.data_path.split("/")[-1])
        
        train_steps = len(self.train_loader)
        train_loss_record = []
        accuracy_score_record = []
        f1_score_record = []
        for epoch in range(self.num_epochs):
            iter_count = 0
            train_loss = 0
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, label) in enumerate(self.train_loader):
                iter_count += 1
                input = input_data.float().to(self.device)
                label = label.float().to(self.device)
                output = self.model(input)

                # loss = self.loss(output,label)
                loss = self.loss(output,label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = train_loss / train_steps
            train_loss_record.append(train_loss.detach().cpu().numpy())
            vali_accuracy,f1 = self.vali(self.vali_loader)
            accuracy_score_record.append(vali_accuracy)
            f1_score_record.append(f1)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali accuracy {3:.7f}".format(epoch + 1, train_steps,train_loss,vali_accuracy))

            early_stopping(f1, self.model, path, self.num_Experiment,self.model_name,self.win_size,self.usepenalty,self.optimizer,self.lr,epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        # 保存损失值以及准确率
        np.save("train_loss_record_"+str(self.model_name)+"_"+str(self.usepenalty)+".npy",train_loss_record)
        np.save("accuracy_score_record_"+str(self.model_name)+"_"+str(self.usepenalty)+".npy",accuracy_score_record)
        np.save("f1_score_record_"+str(self.model_name)+"_"+str(self.usepenalty)+".npy",f1_score_record)
        
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path),str(self.model_name)+"_"+ str(self.data_path.split("/")[-1])+"_"+str(self.win_size) +"_"+ str(self.num_Experiment)+"_"+str(self.usepenalty)+ '_checkpoint_.pth')
            )
        )
        self.model.eval()
        print("=====================Test model============================")
        labels = []
        outputs = []
        for i, (input_data, label) in enumerate(self.test_loader):

            input = input_data.float().to(self.device)
            output = self.model(input).to("cpu")

            outputs.extend(output.detach().cpu().argmax(dim=-1).tolist())
            labels.extend(label.detach().cpu().argmax(dim=-1).tolist())


        np.save("outputs.npy",outputs)
        np.save("labels.npy",labels)

        from sklearn.metrics import classification_report
        report = classification_report(labels, outputs,target_names=[c for c in self.classes],digits=8)
        print("测试报告：")
        print(report)

        cm = confusion_matrix(y_true=labels, y_pred=outputs, labels=[i for i in range(len(self.classes))])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c[3:] for c in self.classes])
        # disp.plot()
        # plt.show()
        # print(disp.confusion_matrix)
        print("混淆矩阵：")
        print(cm)

        def false_positive_rate(y_real,y_pred):
            cm = confusion_matrix(y_true=y_real, y_pred=y_pred)
            fpr = []
            for class_index in range(cm.shape[0]):
                fp = sum(cm[:, class_index]) - cm[class_index, class_index]
                tn = np.sum(cm) - np.sum(cm[class_index, :]) - np.sum(cm[:, class_index]) + cm[class_index, class_index]
                fpr.append(fp / (fp + tn))
            return fpr
        fpr = false_positive_rate(outputs,labels)
        print("误报率 : ",fpr)


