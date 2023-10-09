# Attention-1DCNN

使用Attention和一维卷积实现加密流量分类任务，并采用成本敏感矩阵对 少数类别的样本 进行成本损失惩罚


数据集：


1、BOT_IOT

2、TON_IOT

3、VPN_NOVPN

数据清洗：

1、删除了DNS、DHCP、LLMNR等于加密流量无关的协议

2、将原始pcap切分为会话流量

3、数据填充与截断

4、数据归一化

5、生成成本敏感矩阵

数据输入：

每个会话pcap文件为 10*1500 即10个数据包，每个数据包长度为1500字节。

models 文件夹下集成了1DCNN Attention LSTM BiLSTM RNN GRU 等不同实现。

模型运行：

python main.py

