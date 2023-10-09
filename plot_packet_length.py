

from matplotlib import pyplot as plt

with open('packet_length.txt', 'r') as f:
    data = [int(line.strip()) for line in f.readlines()]
# 求出现在各个范围内的个数
y = [0 for i in range(0,21)]
# 统计 0-3000 之间的数据
for i in data:
    if i < 20:
        y[i] += 1
    else:
        y[-1] += 1
# 绘制直方图
x = [i for i in range(0,21)]

# 计算概率质量函数
y = [i/sum(y) for i in y]

plt.bar(x, y, width=0.3)

plt.xticks([x for x in range(0,21)],rotation = 90)
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
