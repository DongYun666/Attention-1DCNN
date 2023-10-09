# # from scapy.all import *

# # # 抓取或读取网络数据包
# # path = "H://数据集//VPN-nonVPN-ICSX-2016-pcap//VPN-PCAPS-01//vpn_aim_chat1a.pcap"
# # packets = rdpcap(path)
# # decimal_list = []
# # for pkt in packets:
# #     hex_str = pkt.original.hex()
# #     hex_list = [int(hex_str[i:i+2],16)/255 for i in range(0, len(hex_str), 2)]
# #     decimal_list.append(hex_list)
    
# # print(decimal_list)


import argparse
import numpy as np
from scapy.all import *
import os
from scapy.layers.inet import *
from scapy.layers.inet6 import *
from scapy.layers.l2 import *
from sklearn.model_selection import train_test_split

def main(config):
    dataset = config.dataset
    win_size = config.win_size
    length = config.length
    dataset_count = config.count

    path = "./dataset/"+dataset+"/"
    # path = "H://数据集//BoT_IoT数据集//BOT-IOT-pcap"
    save_path = "./processdata/"+dataset+"_"+str(win_size)+"_"+str(length)+"_"+str(dataset_count)
    
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)

    for filename in os.listdir(path):
        if filename.endswith(".pcap"):
            data = []
            print("Processing file: " + filename) 
            filepath = os.path.join(path, filename)
            packets = rdpcap(filepath)

            # 按照五元组分类
            flows = {}
            for packet in packets:

                if packet.haslayer(IP):
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
                    proto = packet[IP].proto
                elif packet.haslayer(IPv6):
                    src_ip = packet[IPv6].src
                    dst_ip = packet[IPv6].dst
                    proto = packet[IPv6].nh
                else:
                    src_ip = ""
                    dst_ip = ""
                    proto = ""
                if packet.haslayer(TCP):
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                elif packet.haslayer(UDP):
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                else:
                    src_port = ""
                    dst_port = ""
                
                if src_port == "":
                    if packet.haslayer(ARP):
                        src_port = "ARP"
                        dst_port = "ARP"
                        proto = "ARP"
                        src_ip = packet[ARP].psrc
                        dst_ip = packet[ARP].pdst
                        
                    elif packet.haslayer(ICMP):
                        src_port = "ICMP"
                        dst_port = "ICMP"
                if src_port == "":
                    continue
                key = (src_ip, dst_ip, src_port, dst_port, proto)
                # 将数据包添加到对应的流中
                if key in flows:
                    flows[key].append(packet)
                else:
                    flows[key] = [packet]

            flows = dict(sorted(flows.items(), key=lambda x: -len(x[1]))) # 按照流中数据包数量降序排序
            flows_count = 0
            # 打印每个流中的数据包数量
            for packets in flows.values():
                # packets = sorted(packets, key=lambda x: -len(x)) # 按照数据包长度降序排序
                temp = []                 
                count = 0                 
                for pkt in packets:       
                    hex_str = pkt.original.hex()
                    hex_list = [int(hex_str[i:i+2],16)/255 for i in range(0, len(hex_str), 2)]
                    if len(hex_list) < length:
                        hex_list = hex_list + [0]*(length-len(hex_list))
                    else:                 
                        hex_list = hex_list[:length]
                    temp.append(hex_list) 
                    count += 1            
                    if count >= win_size: 
                        break             
                if len(temp) < win_size:  
                    temp = temp + temp*(win_size // len(temp) - 1) + temp[:win_size % len(temp)]
                data.append(temp)         
                flows_count += 1          
                if flows_count >= dataset_count:
                    break                 
            dataset = np.array(data)      
            # 先划分数据集再保存               
            train,test = train_test_split(dataset, test_size=0.1, random_state=10)

            np.save(save_path+"/"+filename[:-5]+"_Train.npy", train)
            np.save(save_path+"/"+filename[:-5]+"_Test.npy", test)
            print("保存成功{}数据集".format(filename))
    print("数据集处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模型部分参数
    # SVM LSTM Transformer Linear_Transformer CNN_Transformer Wavelet_Transformer1 Wavelet_Transformer2 Wavelet_Transformer3
    parser.add_argument('--dataset', type=str, default="BOT_IOT",help="数据集名称")
    parser.add_argument('--win_size', type=int, default=10,help="窗口大小")
    parser.add_argument('--length', type=int, default=1024 ,help="截断长度")
    parser.add_argument('--count', type=int, default=10000 ,help="每个类型对应的流数量")

    config = parser.parse_args()    
    main(config)





# filepath = "H://数据集//BoT_IoT数据集//BOT-IOT-pcap//test//zhenghe.pcap"
# packets = rdpcap(filepath)

# # 按照五元组分类
# flows = {}
# count = 0
# for packet in packets:

#     if packet.haslayer(IP):
#         src_ip = packet[IP].src
#         dst_ip = packet[IP].dst
#         proto = packet[IP].proto
#     elif packet.haslayer(IPv6):
#         src_ip = packet[IPv6].src
#         dst_ip = packet[IPv6].dst
#         proto = packet[IPv6].nh
#     else:
#         src_ip = ""
#         dst_ip = ""
#         proto = ""
#     if packet.haslayer(TCP):
#         src_port = packet[TCP].sport
#         dst_port = packet[TCP].dport
#     elif packet.haslayer(UDP):
#         src_port = packet[UDP].sport
#         dst_port = packet[UDP].dport
#     else:
#         src_port = ""
#         dst_port = ""
    
#     if src_port == "":
#         if packet.haslayer(ARP):
#             src_port = "ARP"
#             dst_port = "ARP"
#             proto = "ARP"
#             src_ip = packet[ARP].psrc
#             dst_ip = packet[ARP].pdst
            
#         elif packet.haslayer(ICMP):
#             src_port = "ICMP"
#             dst_port = "ICMP"
#     if src_port == "":
#         count += 1
#         continue
#     key = (src_ip, dst_ip, src_port, dst_port, proto)
#     # 将数据包添加到对应的流中
#     if key in flows:
#         flows[key].append(packet)
#     else:
#         flows[key] = [packet]
# print(flows.keys())
# print(count)