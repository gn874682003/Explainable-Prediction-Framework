import math
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from EHM.DataRecord import DataRecord as DR
import EHM.LogConvert as LC
import EHM.FeatureSel as FS
import EHM.CompareMethod as FC
import EHM.DivideData as DD
import EHM.Prefix as P
import EHM.multiModel as M
import EHM.word2vec as w2v
import EHM.AETS.multiset as multiset
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import copy
import pandas as pd
import torch
import optuna
EL = ['hd','Production_Data','BPIC2012','BPIC2015_1','BPIC2015_2','BPIC2015_3','BPIC2015_4','BPIC2015_5']#
Att = [[3,4,5,6,7,8,9,10,11,12,13],[3,4,5,6,7,8,9,10,11,12],[3,5],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],#
       [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

for eventlog,attribute in zip(EL,Att):
    print(eventlog)
    # 属性转换
    DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)#需转换属性值的下标
    actSum = len(DR.ConvertReflact[0])
    # 特征类别编号
    # 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    DR.State = []
    DR.State.append(0)
    for i in range(4, len(DR.Convert[0])-9):
        if i in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for i in range(6):
        DR.State.append(3)
    # 数据集划分
    DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
    # meanDuration = 0
    # for line in DR.AllData:
    #     meanDuration += line[0][-1]
    # print(meanDuration/len(DR.AllData))

    # Verify--------------------------------------------------------------------------------------
    # caseid = []
    # for line1 in DR.Test:
    #     caseid.append(line1[0][0])
    #     # for line2 in line1:
    #     #     line2.remove(line2[0])
    # dataNameFR = './Input/'+eventlog+'New.mat'
    # dataFR = scio.loadmat(dataNameFR)
    # DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)#1
    # # DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, dataFR['index'][0], -1, 1, 10)#10
    # inn = M.INN(len(dataFR['index'][0]), 32, 1, dataFR, 2)
    # print(sum(param.numel() for param in inn.parameters()))
    # inn.load_state_dict(torch.load('model/'+eventlog+'ANew.pkl'))
    # # MAE = M.TestMetricD(DR.Test_X, DR.Test_Y, inn, 2, dataFR, 1, len(dataFR['index'][0])) #
    # # print(MAE)
    # DR.TrainT, DR.Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
    # DR.TrainT_X, DR.TrainT_Y = P.changeLen(DR.TrainT, dataFR['index'][0], -1, 1)
    # M.prefixPlot(DR.Test_X, DR.Test_Y, inn, dataFR, len(dataFR['index'][0]), DR.TrainT_X, DR.TrainT_Y)
    # print('end')
    # M.viewResult(DR.Test_X, DR.Test_Y, inn, 2, dataFR, caseid, DR.ConvertReflact,attribute)  #
    # M.viewResultD(DR.Test_X,DR.Test_Y,inn,2, dataFR, DR.ConvertReflact, attribute)# caseid
    # ------------------------------------------------------------------------------------------

    # 特征选择比较方法一
    # FC.RFECV_Method(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # # 特征选择比较方法二
    # an, aii = FC.NullImportance() # aii = [13, 0, 7, 8, 10] [0, 12, 2, 3, 11]
    # FS.TestK(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))], aii)
    # 特征选择方法
    DR.TrainT, DR.Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
    if eventlog in ['BPIC2012','hd']:
        FR = FS.LightGBMNew(DR.TrainT, DR.Val, DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    else:
        FR = FS.LightGBMNew(DR.Train, DR.Train, DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    print(DR.header[:-3], len(DR.header[:-3]))

    # FR = FS.Top(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # 活动编码 训练
    DR.Train_XA, DR.Train_YA = P.cutPrefixBy(DR.Train, [0], label=-3, batchSize=20, LEN=3)  # [FR[2][0]]
    if eventlog in ['Production_Data','BPIC2012','hd']:
        ed = 16
    else:
        ed = 32
    EmbA, ACCE = w2v.word2vec(DR.Train_XA, DR.Train_YA, DR.ConvertReflact, ed)

    # dataFE = {'0': EmbA.detach().numpy(), 'name': FE[1], 'index': FE[2], 'state': [DR.State[i] for i in FE[2]],
    #           'result': FE[0], 'prefix': PE}
    # dataFD = {'0': EmbA.detach().numpy(), 'name': FD[1], 'index': FD[2], 'state': [DR.State[i] for i in FD[2]],
    #           'result': FD[0], 'prefix': PD}
    dataFR = {'0': EmbA.detach().numpy(), 'name': FR[1], 'index': FR[2], 'state': [DR.State[i] for i in FR[2]],
              'result': FR[0]}#, 'prefix': PR
    # 其他分类特征编码 随机初始化Embding
    for i in range(1, len(DR.Train[0][0]) - 3):
        if i + 3 in attribute:
            if len(DR.ConvertReflact[attribute.index(i + 3)]) > 4:#5
                eim = 16#5
                olen = len(DR.ConvertReflact[attribute.index(i + 3)])
                while olen > 16:#20
                    olen /= 4
                    eim += 4#5
                EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
                # X_tsne = TSNE(n_components=2, learning_rate=0.1).fit_transform(EmbS.weight.detach().numpy())
                # for j in range(len(X_tsne)):
                #     plt.scatter(X_tsne[j, 0], X_tsne[j, 1])
                # plt.xlabel('x')
                # plt.ylabel('y')
                # plt.title(DR.header[i])
                # plt.show()
                # if i in dataFE['index']:
                #     dataFE[str(i)] = EmbS.weight.detach().numpy()
                # if i in dataFD['index']:
                #     dataFD[str(i)] = EmbS.weight.detach().numpy()
                if i in dataFR['index']:
                    dataFR[str(i)] = EmbS.weight.detach().numpy()

    # 保存编码
    # dataNameFE = '../Data/' + eventlog + '.mat'
    # dataNameFD = '../Data/' + eventlog + '.mat'
    dataNameFR = './Input/' + eventlog + 'New.mat'
    # scio.savemat(dataNameFE, dataFE)
    # scio.savemat(dataNameFD, dataFD)
    scio.savemat(dataNameFR, dataFR)
    # 读取编码
    # dataFE = scio.loadmat(dataNameFE)
    # dataFD = scio.loadmat(dataNameFD)
    dataFR = scio.loadmat(dataNameFR)

    preType = 2  # 预测类型 1分类 2回归
    task = -1  # 预测任务 -1剩余时间 -2下一事件时间 -3下一事件
    epoch = 100
    batchSize = 100 #100
    hiddenSize = 32
    numLayer = 1
    feature = dataFR['index'][-1]

    DR.Train_batch = P.NoFill(DR.Train, feature, task, batchSize)
    DR.Val_X, DR.Val_Y = P.changeLen(DR.Test, feature, task, 1)
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, feature, task, 1)

    # 仅活动，索引编码
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 1, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，索引编码')
    # # 仅活动，One-hot编码
    # act = list(DR.ConvertReflact[0].keys())
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, len(act), hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，One-hot编码')
    # 仅活动，CBOW编码
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 0, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，CBOW编码')
    # 全部拼接，索引编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -1, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test( DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # timeE = time.time()
    # print(Metric, timeE-timeS,'s,全部拼接，索引编码')
    # 全部拼接，向量编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, hiddenSize, numLayer, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,全部拼接，向量编码')
    # 全部拼接，活动CBOW，其他索引编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -3, hiddenSize, numLayer, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,全部拼接，混合编码')

    # 可解释模型整条轨迹前缀
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    DR.Val_X, DR.Val_Y = P.changeLen(DR.Val, dataFR['index'][0], -1, 1)
    DR.TrainT_batch = P.NoFill(DR.TrainT, dataFR['index'][0], -1, batchSize)
    DR.Train_batch = P.NoFill(DR.Train, dataFR['index'][0], -1, batchSize)
    # timeS = time.time()
    # MR = M.LSTMNewTwo(DR.TrainT_batch, DR.Val_X, DR.Val_Y, epoch, 2, len(dataFR['index'][0]), 32, 1, 'inn', dataFR, 1)#64 2
    # timeE = time.time()
    # print('ExplainVal.MAE', MR[0], MR[2], MR[3], timeE - timeS, 's')
    epoch = 300
    timeS = time.time()
    MR = M.LSTMNewTwo(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]), hiddenSize, numLayer, 'inn', dataFR)  # MR[2] MR[3]
    timeE = time.time()
    print('ExplainTest', MR[2], timeE - timeS, 's')
    torch.save(MR[1].state_dict(), 'model/' + eventlog + 'ANew.pkl')
    torch.save(MR[4].state_dict(), 'model/' + eventlog + 'LNew.pkl')

    # 构造可解释训练集，全部拼接，向量编码
    # EachTrain = []
    # for i in range(len(feature)):
    #     Train = copy.deepcopy(DR.Train)
    #     for line1 in Train:
    #         lineT = []
    #         for line2 in line1:
    #             for j in range(len(line2)-3):
    #                 if j not in feature[:i + 1]:
    #                     line2[j] = 0
    #             lineT.append(line2)
    #         EachTrain.append(lineT)
    # DR.Train_batch = P.NoFill(EachTrain, feature, task, batchSize)
    # # DR.Val_X, DR.Val_Y = P.changeLen(DR.Val.copy(), feature, task, 1)
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, hiddenSize, 1, 'rnn', dataFR)
    # for i in range(len(feature)):
    #     EachTest = []
    #     Test = copy.deepcopy(DR.Test)
    #     for line1 in Test:
    #         lineT = []
    #         for line2 in line1:
    #             for j in range(len(line2)-3):
    #                 if j not in feature[:i + 1]:
    #                     line2[j] = 0
    #             lineT.append(line2)
    #         EachTest.append(lineT)
    #     DR.Test_X, DR.Test_Y = P.changeLen(EachTest, feature, task, 1)
    #     Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    #     timeE = time.time()
    #     print(Metric, timeE - timeS, 's,全部拼接，向量编码，可解释训练集')


    # 原AutoEncoder
    # timeS = time.time()
    # MR = multiset.train(DR.AllData, DR.Train, DR.Val, 'multiset', feature)
    # Metric, count = multiset.test(DR.AllData, DR.Test, MR[0], MR[2], 'multiset', feature)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's, AutoEncoder 原方法')