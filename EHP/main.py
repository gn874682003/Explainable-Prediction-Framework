import time
import warnings
warnings.filterwarnings('ignore')
from EHP.DataRecord import DataRecord as DR
import EHP.LogConvert as LC
import EHP.FeatureSel as FS
import EHP.DivideData as DD
import EHP.Prefix as P
import EHP.Model as M
import EHP.word2vec as w2v
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# hd 3,4,5,6,7,8,9,10,11,12,13,14
# BPIC2012 3,5
# BPIC2015 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
# hospital 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
# Production_Data 3,4,5,6,7,8,9,10,11,12
# eventlog= 'hd'
# attribute = [3,4,5,6,7,8,9,10,11,12,13,14]
# 属性转换
EL = ['Production_Data','BPIC2012','hd','BPIC2015_1','BPIC2015_2','BPIC2015_3','BPIC2015_4','BPIC2015_5']#
Att = [[3,4,5,6,7,8,9,10,11,12],[3,5],[3,4,5,6,7,8,9,10,11,12,13,14],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],#
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
for eventlog,attribute in zip(EL,Att):
    print(eventlog,'-------------------------------------------------------------------')
    DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)#需转换属性值的下标
    # 特征类别编号
    DR.State = []
    DR.State.append(0)
    for i in range(1, len(DR.Convert[0])-12):
        if i+3 in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for i in range(6):
        DR.State.append(3)
    # 数据集划分
    DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
    # 验证
    # dataNameFR = './data/dataFR'+eventlog+'S.mat'
    # dataFR = scio.loadmat(dataNameFR)
    # DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    # # DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, dataFR['index'][0], -1, 1, 10)
    # inn = M.INN(len(dataFR['index'][0]), 32, 1, dataFR,2)
    # print(sum(param.numel() for param in inn.parameters()))
    # inn.load_state_dict(torch.load('model/MR_'+eventlog+'S.pkl'))
    # MAE = M.TestMetric(DR.Test_X,DR.Test_Y,inn,2, dataFR, 1,len(dataFR['index'][0]))
    # # print(MAE)
    # M.viewResult(DR.Test_X,DR.Test_Y,inn,2, dataFR, caseid)#DR.ConvertReflact,attribute,
    #------------------------------------------------------------------------------------------

    #特征选取
    FR = FS.FinalFLightboost(DR.Train, DR.Test, DR.header)
    # FR = FS.LightGBM(DR.Train, DR.Test, DR.header,[attribute[i]-3 for i in range(len(attribute))])
    # FS.PrefixLightGBM(DR.Train, DR.header, DR.State, [], [], FR)
    # 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    # 特征编码,默认基于下标的

    # 前缀分组
    epoch = 300
    batchSize = 100
    LEN = 100

    # 活动编码 训练
    DR.Train_XA, DR.Train_YA = P.cutPrefixBy(DR.AllData, [0],label=-3,batchSize=20,LEN=3)#[FR[2][0]]
    EmbA, ACCE = w2v.word2vec(DR.Train_XA, DR.Train_YA, DR.ConvertReflact)
    dataFR = {'0':EmbA.detach().numpy(),'name':FR[1],'index':FR[2],'state':[DR.State[i] for i in FR[2]],'result':FR[0]}

    # 其他分类特征编码 随机初始化Embding
    for i in range(1,len(DR.Train[0][0])-3):
        if i+3 in attribute:
            if len(DR.ConvertReflact[attribute.index(i+3)])>5:
                eim = 5
                olen = len(DR.ConvertReflact[attribute.index(i+3)])
                while olen > 20:
                    olen /= 4
                    eim += 5
                EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i+3)]), eim)
                X_tsne = TSNE(n_components=2, learning_rate=0.1).fit_transform(EmbS.weight.detach().numpy())
                # for j in range(len(X_tsne)):
                #     plt.scatter(X_tsne[j, 0], X_tsne[j, 1])
                # plt.xlabel('x')
                # plt.ylabel('y')
                # plt.title(DR.header[i])
                # plt.show()
                if i in dataFR['index']:#FR[2]
                    dataFR[str(i)] = EmbS.weight.detach().numpy()

    # 保存编码
    dataNameFR = './data/FR_'+eventlog+'.mat'
    scio.savemat(dataNameFR, dataFR)

    # 读取编码
    dataFR = scio.loadmat(dataNameFR)
    FR = [0,dataFR['name'],dataFR['index']]
    # 基线方法 Event:[0] Select:dataFR/E/D['index'][0]
    # 剩余时间
    timeS = time.time()
    DR.Train_X, DR.Train_Y = P.changeLen(DR.Train, [0], -1, batchSize)#dataFR['index']
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, [0], -1, 1)
    MR = M.baseLine(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, epoch, 2, len([0]), 32, 1, 'rnn')
    timeE = time.time()
    print('InAct.MAE',MR[0], timeE - timeS,'s')

    timeS = time.time()
    DR.Train_X, DR.Train_Y = P.changeLen(DR.Train, dataFR['index'][0], -1, batchSize)#dataFR['index']
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    MR = M.baseLine(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]), 128, 2, 'rnn')
    timeE = time.time()
    print('InExplainFHM.MAE',MR[0], timeE - timeS,'s')

    timeS = time.time()
    # DR.Train_X, DR.Train_Y = P.diffWindow(DR.Train, [0], -1, batchSize, LEN)
    # DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, [0], -1, 1, LEN)
    # MR = M.LSTMDiff(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, epoch, 2, len([0]), 16, 1, 'inn', dataFR)
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, [0], -1, 1)
    DR.Train_batch = P.NoFill(DR.Train, [0], -1, batchSize)
    MR = M.LSTMNew(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len([0]), 32, 1, 'inn', dataFR)
    timeE = time.time()
    print('EembAct.MAE',MR[0], timeE - timeS,'s')

    # 可解释模型整条轨迹前缀
    timeS = time.time()
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    DR.Train_batch = P.NoFill(DR.Train, dataFR['index'][0], -1, batchSize)
    MR = M.LSTMNew(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]), 64, 2, 'inn', dataFR)
    timeE = time.time()
    print('ExplainFHM1.MAE', MR[0], MR[2], timeE - timeS, 's')
    torch.save(MR[1].state_dict(), 'model/MR_'+eventlog+'SN.pkl')
    torch.save(MR[3].state_dict(), 'model/MR_'+eventlog+'SNA.pkl')

    # 可解释模型轨迹切分前缀
    timeS = time.time()
    DR.Train_X, DR.Train_Y = P.diffWindow(DR.Train, dataFR['index'][0], -1, batchSize, LEN)
    DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, dataFR['index'][0], -1, 1, LEN)
    MR = M.LSTMDiff(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]), 32, 1, 'inn', dataFR)
    timeE = time.time()
    print('ExplainFHM2.MAE', MR[0], MR[2], timeE - timeS, 's')
    torch.save(MR[1].state_dict(), 'model/MR_'+eventlog+'S.pkl')
    torch.save(MR[3].state_dict(), 'model/MR_'+eventlog+'SA.pkl')
    print('----------------------------------------------------------------------------------')