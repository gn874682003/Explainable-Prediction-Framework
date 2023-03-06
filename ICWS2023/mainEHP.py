import time
import warnings
warnings.filterwarnings('ignore')
from ICWS2023.DataRecord import DataRecord as DR
import ICWS2023.LogConvert as LC
import ICWS2023.FeatureSel as FS
import ICWS2023.DivideData as DD
import ICWS2023.Prefix as P
import ICWS2023.Model as M
import ICWS2023.word2vec as w2v
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

eventlog= 'hd'
attribute = [3,4,5,6,7,8,9,10,11,12,13,14]

# Convert attribute
DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)
# [0 activity；1 Classification static features；2 Classification dynamic features；
# 3 Numerical static characteristics；4 Numerical dynamic characteristics]
DR.State = []
DR.State.append(0)
for i in range(1, len(DR.Convert[0])-12):
    if i+3 in attribute:
        DR.State.append(1)
    else:
        DR.State.append(3)
for i in range(6):
    DR.State.append(4)

# Data set partition
DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
DR.Train, DR.Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
# Verify--------------------------------------------------------------------------------------
# caseid = []
# for line1 in DR.Test:
#     caseid.append(line1[0][0])
# dataNameFR = './data/'+eventlog+'.mat'
# dataFR = scio.loadmat(dataNameFR)
# # DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)#1
# DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, dataFR['index'][0], -1, 1, 10)#2
# inn = M.INN(len(dataFR['index'][0]), 32, 1, dataFR, 2)
# print(sum(param.numel() for param in inn.parameters()))
# inn.load_state_dict(torch.load('model/MR_'+eventlog+'S.pkl'))
# MAE = M.TestMetricD(DR.Test_X, DR.Test_Y, inn, 2, dataFR, 1, len(dataFR['index'][0]))
# print(MAE)
# M.viewResultD(DR.Test_X,DR.Test_Y,inn,2, dataFR, DR.ConvertReflact, attribute)# caseid
#------------------------------------------------------------------------------------------

# Feature selection strategy
FR = FS.LightGBM(DR.Train,  DR.Val, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])

# Depth-first Traverse Results of All Feature Combination
# FR = FS.AllFTree(DR.Train, DR.Test, DR.header,[attribute[i]-3 for i in range(len(attribute))])

# User-guided Feature Selection
# selFea = [0,13,7] #, 2,16,5,9,12,10 hd all4.324
# FR = [4.692, [DR.header[i] for i in selFea], selFea]

# Activity encoding(CBow)
DR.Train_XA, DR.Train_YA = P.cutPrefixBy(DR.AllData, [0], label=-3, batchSize=20, LEN=3)#[FR[2][0]]
EmbA, ACCE = w2v.word2vec(DR.Train_XA, DR.Train_YA, DR.ConvertReflact)
dataFR = {'0':EmbA.detach().numpy(),'name':FR[1],'index':FR[2],'state':[DR.State[i] for i in FR[2]],'result':FR[0]}

# Other classification feature encoding, random initialization Embedding
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
            for j in range(len(X_tsne)):
                plt.scatter(X_tsne[j, 0], X_tsne[j, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(DR.header[i])
            plt.show()
            if i in dataFR['index']:#FR[2]
                dataFR[str(i)] = EmbS.weight.detach().numpy()

# save information
dataNameFR = './data/'+eventlog+'.mat'
scio.savemat(dataNameFR, dataFR)

# read information
dataFR = scio.loadmat(dataNameFR)

# Hyperparameter setting
epoch = 300
batchSize = 100

hiddenSize = 32
numLayer = 1
feature = dataFR['index'][-1]
preType = 2  # 1 classical 2 regress
task = -1  # prediction task -1 remainingTime -2 nextEventDuration -3 nextActivity

DR.Train_batch = P.NoFill(DR.Train, feature, task, batchSize)
DR.Val_X, DR.Val_Y = P.changeLen(DR.Test, feature, task, 1)
DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, feature, task, 1)

# Activity,Index
MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 1, hiddenSize, 1, 'rnn', dataFR)
Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
print('Activity,Index:', Metric)
# Activity,One-hot
act = list(DR.ConvertReflact[0].keys())
MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, len(act), hiddenSize, 1, 'rnn', dataFR)
Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
print('Activity,One-hot:', Metric)
# Activity,CBOW
MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 0, hiddenSize, 1, 'rnn', dataFR)
Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
print('Activity,CBOW:', Metric)
# non-hierarchical, Word Vector
timeS = time.time()
MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, hiddenSize, numLayer, 'rnn', dataFR)
Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
timeE = time.time()
print('non-hierarchical, Word Vector:', Metric, timeE - timeS, 's')


timeS = time.time()
MR = M.LSTMEHP(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, 2, len(dataFR['index'][0]), 32, 1, 'inn', dataFR)  # MR[2] MR[3]
timeE = time.time()
print('EHPTest:', MR[2], timeE - timeS, 's')
torch.save(MR[1].state_dict(), 'model/' + eventlog + 'A.pkl')
torch.save(MR[4].state_dict(), 'model/'+eventlog+'L.pkl')