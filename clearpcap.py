import os
import numpy as np
from scapy.all import *
from scapy.layers.inet import *
from scapy.layers.inet6 import *
from scapy.layers.l2 import *

path = "J:/dataset1/ISCX_VPN_NOVPN/NOVPN/novpn_voip/remove_abnormalTCP_pcap_diff"
savepath = "J:/dataset1/ISCX_VPN_NOVPN/NOVPN/novpn_voip/clearpcap"
# otherpath = "J:/dataset1/ISCX_VPN_NOVPN/NOVPN/novpn_chat/otherpcap"
if not os.path.exists(savepath):
    os.makedirs(savepath)
count = 0
for filname in os.listdir(path):
    pcappath = os.path.join(path, filname)
    packets = rdpcap(pcappath)
    count += 1
    if count % 100 == 0:
        print("Start processing "+str(count)+"...")
    if packets[0].haslayer(TCP):  # TCP 数据包 
        if len(packets) < 6 or "S" not in packets[0][TCP].flags:
            # wrpcap(otherpath+"//"+pcappath.split("\\")[-1],packets)
            continue
        else:
            wrpcap(savepath+"//"+pcappath.split("\\")[-1],packets)
    else:
        if packets[0][Ether].src == 'ff:ff:ff:ff:ff:ff' or packets[0][Ether].dst == 'ff:ff:ff:ff:ff:ff':
            # wrpcap(otherpath+"//"+pcappath.split("\\")[-1],packets)
            continue
        else:
            wrpcap(savepath+"//"+pcappath.split("\\")[-1],packets)

print("Finish clear all data!")
