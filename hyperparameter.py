import optuna
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
EL = ['Production_Data','hd','BPIC2012','BPIC2015_1','BPIC2015_2','BPIC2015_3','BPIC2015_4','BPIC2015_5']#
Att = [[3,4,5,6,7,8,9,10,11,12],[3,4,5,6,7,8,9,10,11,12,13],[3,5],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],#
       [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
learnRate = [0.001,0.003,0.0008,0.004,0.004,0.004,0.004,0.004]
BS = [25,75,50,50,50,50,50,50]
for eventlog,attribute,LR,batchSize in zip(EL,Att,learnRate,BS):
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
    DR.TrainT, DR.Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
    # 读取编码
    dataNameFR = './Input/' + eventlog + 'New.mat'
    dataFR = scio.loadmat(dataNameFR)

    preType = 2  # 预测类型 1分类 2回归
    task = -1  # 预测任务 -1剩余时间 -2下一事件时间 -3下一事件
    epoch = 500
    hiddenSize = 32
    numLayer = 1
    feature = dataFR['index'][-1]

    # 可解释模型整条轨迹前缀
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    DR.Val_X, DR.Val_Y = P.changeLen(DR.Val, dataFR['index'][0], -1, 1)
    timeS = time.time()

    DR.Train_batch = P.NoFill(DR.Val, dataFR['index'][0], -1, batchSize)
    MR = M.LSTMNewTwoEarly(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]),
                           hiddenSize, numLayer, 'inn', dataFR, LR)
    timeE = time.time()
    print(MR[0], timeE - timeS, 's')
    torch.save(MR[1].state_dict(), 'model/' + eventlog + 'AHyp.pkl')
    torch.save(MR[4].state_dict(), 'model/' + eventlog + 'LHyp.pkl')

    # def obj(trail):
    #     batchSize = trail.suggest_int('batchSize', 20, 120)
    #     LR = trail.suggest_float('LR', 0.0005, 0.005)
    #     # opt = trail.suggest_categorical('opt', ['Adam', 'SGD'])
    #     DR.Train_batch = P.NoFill(DR.Val, dataFR['index'][0], -1, batchSize)
    #     MR = M.LSTMNewTwoEarly(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]),
    #                       hiddenSize, numLayer, 'inn', dataFR,LR)
    #     return MR[0]
    # stu = optuna.create_study(study_name='test', direction='minimize')
    # stu.optimize(obj, n_trials=10)
    # print(stu.best_params)
    # print(stu.best_trial)
    # print(stu.best_trial.value)
    # timeE = time.time()
    # print('ExplainTest', timeE - timeS, 's')

