import numpy as np
import matplotlib.pyplot as plt

def Friedman(n, k, data_matrix):
    '''
    Friedman 检验
    :param n:数据集个数
    :param k: 算法种数
    :param data_matrix:排序矩阵
    :return:T1
    '''

    # 计算每个算法的平均序值
    row, col = data_matrix.shape  # 获取矩阵的行和列
    xuzhi_mean = list()
    for i in range(col):  # 计算平均序值
        xuzhi_mean.append(data_matrix[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
    sum_mean = np.array(xuzhi_mean)  # 转成 numpy.ndarray 格式方便运算

    sum_ri2_mean = (sum_mean ** 2).sum()  # 整个矩阵内的元素逐个平方后，得到的值相加起来
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * ((k + 1) ** 2)) / 4)) / (k * (k + 1))  # P42页的公式
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)  # P42页的公式
    return result_Tf


def nemenyi(n, k, q):
    '''
    Nemenyi 后续检验
    :param n:数据集个数
    :param k:算法种数
    :param q:直接查书上2.7的表
    :return:
    '''
    cd = q * (np.sqrt((k * (k + 1) / (6 * n))))
    return cd

n = 8 #数据集个数
k = 6 #算法个数

data = np.array([[8.546,19.570,6.544,9.140,5.688,4.573],#
                 [7.896,11.081,6.369,7.826,6.325,6.287],#
                 [141.613,52.546,41.907,46.661,59.529,35.281],
                 [89.223,83.138,71.371,80.159,93.915,62.017],
                 [28.025,20.505,17.291,19.410,30.462,16.544],
                 [79.969,62.166,55.214,39.839,48.327,44.617],
                 [177.646,48.795,37.002,35.199,47.897,35.604],
                 [18.249,11.127,9.531,10.847,13.039,8.283]])#

# data = np.array([[41.803,38.961,38.467,37.966,37.938,37.449],
#                 [79.059,66.720,65.018,64.370,63.925,64.080],
#                 [18.861,17.688,17.302,17.125,16.998,17.902],
#                 [58.769,47.644,49.579,49.662,50.714,54.478],
#                 [42.583,38.105,36.913,36.683,36.293,36.429]])
    # ([[6.730,6.669,6.636,6.770,6.779,6.875],
    #             [6.530,6.504,6.430,6.414,6.432,6.646],
    #              [11.663,10.505,9.713,9.765,10.510,10.871]])
                #


T1 = Friedman(n, k, data)
cd = nemenyi(n, k, 2.850)
print('tf={}'.format(T1))
print('cd={}'.format(cd))

# 画出CD图
row, col = data.shape  # 获取矩阵的行和列
xuzhi_mean = list()
for i in range(col):  # 计算平均序值
    xuzhi_mean.append(data[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
sum_mean = np.array(xuzhi_mean)
# 这一句可以表示上面sum_mean： rank_x = list(map(lambda x: np.mean(x), data.T))  # 均值 [1.0, 2.125, 2.875]
name_y = ['Camargo_LSTM','CRTP_LSTM','GRU_NP','Auto-encoded','Process Transformer','FCPM']#
# name_y = ['Emb_size=4','Emb_size=8','Emb_size=16','Emb_size=32','Emb_size=64','Emb_size=128']
# 散点左右的位置
min_ = sum_mean - cd / 2
max_ = sum_mean + cd / 2
# 因为想要从高出开始画，所以数组反转一下
name_y.reverse()
sum_mean = list(sum_mean)
sum_mean.reverse()
max_ = list(max_)
max_.reverse()
min_ = list(min_)
min_.reverse()
# 开始画图
# plt.title("Friedman")
plt.figure(figsize=(7, 3.5))
plt.scatter(sum_mean, name_y)  # 绘制散点图
plt.hlines(name_y, max_, min_)
plt.xlabel('MAE')
plt.ylabel('Prediction Approaches')#Emb_size
plt.show()
print('end')