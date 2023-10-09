import os
import argparse
import time

from torch.backends import cudnn

# from Experiment3 import Experiment

# def mkdir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# def main(config):
#     cudnn.benchmark = True
#     if (not os.path.exists(config.model_save_path)):
#         mkdir(config.model_save_path)
#     for i in range(config.num_Experiment):          
#         print("============= 进行第{}次实验=============".format(i+1))
#         experiment= Experiment(vars(config))
#         if config.mode == 'Train':
#             start_time = time.time()
#             experiment.train()
#             print("训练时间：{}".format(time.time() - start_time))
#         else:
#             experiment.test()
#     return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 模型部分参数
    # GRU   RNN   RNNCNN1d  LSTM CNN1d(跑不出结果) LSTMCNN1d BiLSTM SVM BiLSTMCNN1d Transformer Linear_Transformer CNN1dTransformer TransformerCNN1d WaveTransformerCNN1D CNN_Transformer Wavelet_Transformer1 Wavelet_Transformer2 Wavelet_Transformer3 CNN_Transformer2
    parser.add_argument('--model_name', type=str, default="LSTMCNN1d",help="模型名称")
    # parser.add_argument('--model_name', type=str, default="CNN1d",help="模型名称")    
    # 公共参数
    parser.add_argument('--feature_num', type=int, default=1500,help="特征个数")
    parser.add_argument('--d_model', type=int, default=512) #最后一维映射的维度

    parser.add_argument('--dropout', type=float, default=0,help="dropout的概率")
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    
    #使用Transformer的参数  
    parser.add_argument('--e_layers', type=int, default=3,help="encoder 的层数")
    parser.add_argument('--num_heads', type=int, default=8,help="划分的头的个数")
    parser.add_argument('--attention_type', type=str, default='full',help="attention的类型")
    
    #使用CNNTransformer的参数
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--padding', type=int, default=1)
    
    # 使用WaveletTransformer的参数 
    parser.add_argument('--wavelet_method', type=str, default='db4')
    parser.add_argument('--trainable', type=str)

    # 实验部分参数 
    parser.add_argument('--device', type=str, default='cuda:0',help="是否使用cuda")
    # parser.add_argument('--device', type=str, default='cpu',help="是否使用cuda")

    parser.add_argument('--num_Experiment', type=int, default=1 ,help="实验次数")
    parser.add_argument('--lr', type=float, default=1e-4) # 3e-4 的时候出现了一次0.95
    parser.add_argument('--num_epochs', type=int, default = 1000)

    parser.add_argument('--patience', type=int, default=50,help="早停机制中的patience参数") 
    parser.add_argument('--random_seed', type=int, default=226,help="设置随机种子")
    parser.add_argument('--mode', type=str, default='Train',help="训练模式" )
    # parser.add_argument('--mode', type=str, default='Test',help="测试模式") 
    parser.add_argument('--usepenalty', type=str,default='True',help="是否使用惩罚项")
    parser.add_argument('--gamma', type=int, default=2,help="惩罚项参数") 
    
    # 数据集部分
    parser.add_argument('--data_path',type=str,default='./processdata/notor') #需要修改
    # 窗口内归一化
    # 先阻尼增量统计后使用train_test_split 划分数据集后实验效果很好 位置 test  Linear_Transformer 0.987

    # 先划分数据集(采用末尾截取的办法) 再阻尼增量统计  位置 test2 Linear_Transformer  0.561

    # 先划分数据集(采用末尾截取的办法) 再阻尼增量统计 随机打乱 位置 test2 Linear_Transformer  0.410

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 随机打乱 位置 BOT_IOT_order  0.210

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计  位置 BOT_IOT_order  0.189

    # 整体数据归一化
    # 先阻尼增量统计后使用train_test_split 划分数据集后实验效果很好 位置 test  Linear_Transformer 1.0

    # 先划分数据集(采用末尾截取的办法) 再阻尼增量统计  位置 t
    # 
    # est2 Linear_Transformer 0.568

    # 先划分数据集(采用末尾截取的办法) 再阻尼增量统计 随机打乱 位置 test2 Linear_Transformer  0.565

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 随机打乱 位置 BOT_IOT_order Linear_Transformer 0.994

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 训练随机打乱测试不打乱 位置 BOT_IOT_order Linear_Transformer 0.973 左下 47 右上45

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 都不随加打乱 位置 BOT_IOT_order Linear_Transformer 0.9683  右上有21

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 都不随加打乱 位置 BOT_IOT_order Wavelet_Transformer1 0.972 右上和坐下 49

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 都不随机打乱 位置 BOT_IOT_order Wavelet_Transformer2 窗口为100 时需要更大的内存 窗口为50可以运行 左下104 右上90 舍弃

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 都不随机打乱 位置 BOT_IOT_order Wavelet_Transformer3 0.970 左下55 右上49

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 训练随机打乱测试不随加打乱 位置 BOT_IOT_order Wavelet_Transformer3 0.970 左下56 右上48

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 随机打乱 位置 BOT_IOT_order Wavelet_Transformer3 1.0

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 都不随机打乱 位置 BOT_IOT_order CNN_Transformer 0.973 左下55 右上49

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 随机打乱 位置 BOT_IOT_order CNN_Transformer 0.989 右上36

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 训练随机打乱测试不随机打乱 位置 BOT_IOT_order CNN_Transformer 0.964 左下39 右上86

    # 采用两个不同的数据集分别作为训练和测试集 都进行阻尼增量统计 随机打乱 位置 BOT_IOT_order SVM 0.998

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--win_size', type=int, default=10,help="窗口大小")
    config = parser.parse_args()
    if config.trainable == "True":
        config.trainable = True
    else:
        config.trainable = False
    if config.usepenalty == "True":
        config.usepenalty = True
    else:
        config.usepenalty = False
    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    # main(config)

