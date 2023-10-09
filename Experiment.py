import os
import time
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch import nn

from data_factory.data_loader import get_dataloader
from models.BuildModel import build
from models.loss import my_loss, penalty_focal_loss

import warnings
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score

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


# 早停机制 可能需要修改  如果val_loss
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.var_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path, num_experiment,model_name,win_size):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment,model_name,win_size)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, num_experiment,model_name,win_size)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, num_experiment,model_name,win_size):
        if self.verbose:
            print(f'====Validation loss decreased ({self.var_loss_min:.6f} --> {val_loss:.6f}). Saving model====')
        torch.save(model.state_dict(),os.path.join(path, str(model_name)+"_" + str(self.dataset) +"_"+str(win_size)+"_"+str(num_experiment)+'_checkpoint_.pth'))
        self.var_loss_min = val_loss


# 实验主程序
class Experiment(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Experiment.DEFAULTS, **config)
        self.same_seeds(seed=self.random_seed)

        self.data_loader,self.classes,self.labels_num_count = get_dataloader(data_path=self.data_path,batch_size=self.batch_size, win_size=self.win_size,mode = self.mode)
        
        self.num_classes = len(self.classes)

        if self.device is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_num

        self.device = torch.device(self.device+self.device_num if self.device is not None else "cpu")

        if self.usepenalty:
            self.loss =  penalty_focal_loss(self.labels_num_count,self.device,self.gamma)
        else:
            self.loss = my_loss()
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
        
        if torch.cuda.is_available():
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

    def train(self):
        print('====================Train  model======================')
        time_now = time.time()
        path = self.model_save_path

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        # 需要修改
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, dataset_name=self.data_path.split("/")[-1])
        
        train_steps = len(self.data_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            epoch_time = time.time()
            self.model.train()

            for i, (input_data, label) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                label = label.float().to(self.device)
                output = self.model(input)

                rec_loss = self.loss(output,label)

                loss_list.append(rec_loss)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                rec_loss.backward()

                self.optimizer.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = torch.mean(torch.tensor(loss_list))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps,train_loss))


            early_stopping(train_loss, self.model, path, self.num_Experiment,self.model_name,self.win_size)
            if early_stopping.early_stop:
               print("Early stopping")
               break
            # adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path),str(self.model_name)+"_"+ str(self.data_path.split("/")[-1])+"_"+str(self.win_size) +"_"+ str(self.num_Experiment)+ '_checkpoint_.pth')
            )
        )
        self.model.eval()
        print("=====================Test model============================")
        labels = []
        outputs = []
        for i, (input_data, label) in enumerate(self.data_loader):

            input = input_data.float().to(self.device)
            output = self.model(input).to("cpu")

            # output = torch.exp(output) / torch.sum(torch.exp(output), dim=-1).reshape(output.shape[0],1)

            output = output.max(dim=-1)[1]
            label = label.max(dim=-1)[1]
            outputs.append(output.detach().cpu().numpy().reshape(-1))
            labels.append(label.detach().cpu().numpy().reshape(-1))

        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        print(outputs.shape)
        print(outputs)
        print(type(outputs))
        print(labels.shape)
        print(type(labels))
        print(labels)

        accuracy = accuracy_score(labels, outputs)
        print("accuracy : ", accuracy)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, outputs)
        print(accuracy, precision, recall, f_score)
        from sklearn.metrics import classification_report
        report = classification_report(labels, outputs,target_names=[c for c in self.classes],digits=8)
        print(report)

        cm = confusion_matrix(y_true=labels, y_pred=outputs, labels=[i for i in range(len(self.classes))])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c[3:] for c in self.classes])
        # disp.plot()
        # plt.show()
        # print(disp.confusion_matrix)
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
        print("fpr: ",fpr)


