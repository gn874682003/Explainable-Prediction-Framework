import copy
import random

import ICPM2022.Prefix as P
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Hyper Parameters
INPUT_SIZE = 5          # rnn input size / image width
LR = 0.001               # learning rate

# Calculate cosine similarity
def get_att_dis(target, behaviored):
    attention_distribution = []
    result = []
    for j in range(target.shape[1]):
        for i in range(behaviored.shape[0]):
            attention_score = torch.cosine_similarity(target[0,j,:].view(1, -1), torch.FloatTensor(behaviored[i]).view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
            attention_distribution.append(attention_score)
        result.append(int(torch.argmax(torch.Tensor(attention_distribution))))
        attention_distribution = []
    return  result

def viewResult(X_Test, Y_Test, rnn, type, data,caseid):
    for x, y,cid in zip(X_Test, Y_Test,caseid):
        output, prediction = rnn(x, data)
        print(cid)
        print(x)
        print(output)
        for j in range(prediction.size(1)):
            yi = 0
            for line, label in zip(output, data['name']):
                plt.scatter(line.view(line.size(1))[j].detach().numpy(), yi)
                plt.annotate(label, xy=(line.view(line.size(1))[j].detach().numpy(), yi), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom')
                yi += 1
            # plt.vlines(y[0][j],0,yi-1)
            y_major_locator = MultipleLocator(1)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.xlabel('Time(day)')
            plt.ylabel('Feature Number')
            plt.show()

def viewResultD(X_Test, Y_Test, rnn, type, data, ConvertReflact,attribute):
    for x, y in zip(X_Test, Y_Test):
        output, prediction = rnn(x, data)
        print(x)
        for j in range(prediction.size(1)):
            yi = 0
            for line, i in zip(output,range(len(output))):
                if data['state'][0][i]<3:
                    label = ConvertReflact[attribute.index(data['index'][0][i]+3)][int(x[0][j][i])]
                else:
                    label = x[0][j][i]
                print(label)
                plt.scatter(line.view(line.size(1))[j].detach().numpy(), yi)
                #plt.annotate(label, xy=(line.view(line.size(1))[j].detach().numpy(), yi), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
                yi += 1
            plt.vlines(y[0], 0, yi-1)
            y_major_locator = MultipleLocator(1)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.yticks(range(len(output)), data['name'])
            plt.xlabel('Time(day)')
            plt.ylabel('Feature Number')
            plt.show()

def TestMetric(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                    true_y.append(line1)
                    pred_y.append(line2)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
                    true_y.append(line1)
                    pred_y.append(line2)#[-1]
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    # if type == 1:
    #     for x, y in zip(X_Test, Y_Test):
    #         prediction = rnn(x)  # rnn output
    #         if prediction.size()[-1] == 1:
    #             true_y.append(np.round(prediction.detach().numpy().tolist()[0]))
    #             pred_y.append(y.numpy().tolist()[0])
    #         else:
    #             for line1, line2 in zip(y.numpy().tolist()[0], np.round(prediction.detach().numpy()).tolist()[0]):
    #                 true_y.append(line1)
    #                 pred_y.append(line2)
    #     Metric = accuracy_score(true_y, pred_y)
    # else:
    #     for x, y in zip(X_Test, Y_Test):
    #         prediction = rnn(x)  # rnn output
    #         for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
    #             true_y.append(line1)
    #             pred_y.append(line2)
    #     Metric = mean_absolute_error(true_y, pred_y)
    return Metric

def TestMetricB(X_Test, Y_Test, rnn, type):
    pred_y = []
    true_y = []
    if type == 1:
        for x, y in zip(X_Test, Y_Test):
            prediction = rnn(x)  # rnn output
            for line1, line2 in zip(y.numpy().tolist()[0], np.round(prediction.detach().numpy()).tolist()[0]):#[0]
                true_y.append(line1)
                pred_y.append(line2)
        Metric = accuracy_score(true_y, pred_y)
    else:
        for x, y in zip(X_Test, Y_Test):
            prediction = rnn(x)
            for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):#[0]
                true_y.append(line1)
                pred_y.append(line2)#[-1]
        Metric = mean_absolute_error(true_y, pred_y)
    return Metric

def TestMetricD(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction.detach().numpy().tolist()[0][-1])#
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def TestMetricN(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output = rnn(x, data)
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(output.detach().numpy().tolist()[0][-1])#
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def LSTMNest(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    n = NNN(input_size, hidden_size, num_layers, data, type)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for x, y in zip(Train_X, Train_Y):
            output = n(x, data)
            optimizer.zero_grad()
            if type == 1:
                temp = []
                for line in y:
                    temp.append(data['0'][int(line)])
                ty = torch.FloatTensor(temp)
                loss = loss_func(output, ty)
            else:
                loss = loss_func(output[:,0], y)
            loss.backward()
            optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetricN(Test_X, Test_Y, n, type, data,1,input_size)
        else:
            Metric = TestMetricN(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1:
            if Metric[-1] > BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if max(Metric) > BestAll:
                BestAll = max(Metric)
                BestModelAll = copy.deepcopy(n)
        elif type == 2:
            if Metric[-1] < BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if min(Metric) < BestAll:
                BestAll = min(Metric)
                BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTMDiff(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if method == 'inn':
        n = INN(input_size, hidden_size, num_layers, data, type)
    elif method == 'inn2':
        n = INN2(input_size, hidden_size, num_layers, data, type)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y in zip(Train_X, Train_Y):
                output, prediction = n(x, data)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in y:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(prediction, ty)
                else:
                    loss = loss_func(output[k][:,-1,0], y)
                loss.backward()
                optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetricD(Test_X, Test_Y, n, type, data,1,input_size)
        else:
            Metric = TestMetricD(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1:
            if Metric[-1] > BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if max(Metric) > BestAll:
                BestAll = max(Metric)
                BestModelAll = copy.deepcopy(n)
        elif type == 2:
            if Metric[-1] < BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if min(Metric) < BestAll:
                BestAll = min(Metric)
                BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTMNew(Train,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):#6
            for j, (x, y, l) in enumerate(Train):
                ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
                output, prediction = n(x, data)
                py = nn.utils.rnn.pack_padded_sequence(output[k], l, batch_first=True)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in ty.data:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(py.data, ty)
                else:
                    loss = loss_func(py.data.view(py.data.size(0)), ty.data)
                loss.backward()
                optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetric(Test_X, Test_Y, n, type, data,1,input_size)
            # print(Metric)
        else:
            Metric = TestMetric(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1 and Metric[-1] > BestResult:
            BestResult = Metric[-1]
            BestModel = copy.deepcopy(n)
        elif type == 2 and Metric[-1] < BestResult:
            BestResult = Metric[-1]
            BestModel = copy.deepcopy(n)
        elif type == 1 and max(Metric) > BestAll:
            BestAll = max(Metric)
            BestModelAll = copy.deepcopy(n)
        elif type == 2 and min(Metric) < BestAll:
            BestAll = min(Metric)
            BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTM(X,Y,X_Test,Y_Test, Train_L, Test_L,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if method == 'inn':
        n = INN(input_size, hidden_size, num_layers, data, type)
    elif method == 'inn2':
        n = INN2(input_size, hidden_size, num_layers, data, type)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MSELoss()  # the target label is not one-hotted.L1Loss()
    loss_func = nn.L1Loss()  # the target label is not one-hotted.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y, l in zip(X, Y, Train_L):
                if type == 1:
                    ty, temp = [], []
                    for line1 in y:
                        for line2 in line1:
                            temp.append(data['0'][int(line2)])
                        ty.append(temp)
                        temp = []
                    y = torch.FloatTensor(ty)
                else:
                    y = y.view(y.size(0), y.size(1), 1)
                # for k in range(start,input_size):
                output, prediction = n(x, data, l)#
                optimizer.zero_grad()  # clear gradients for this training step
                # for preI in output:#[0:j]
                    # preI = output[j]
                loss = loss_func(output[k], y)  # cross entropy loss[:,-1]
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
        if (i+1) % 10 == 0:
            Metric = TestMetric(X_Test, Y_Test, n, type, data,1)
        else:
            Metric = TestMetric(X_Test, Y_Test, n, type, data,0)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestModel = n
        elif type == 1 and Metric[-1] > BestResult:
            BestResult = Metric[-1]
            BestModel = n
        elif type == 2 and Metric[-1] < BestResult:
            BestResult = Metric[-1]
            BestModel = n
    return BestResult, BestModel

def baseLine(X,Y,X_Test,Y_Test,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None):
    if method == 'fnn':
        n = FNN(input_size,hidden_size)
    elif method == 'rnn':
        n = RNN(input_size, hidden_size, num_layers)
    elif method == 'OLSTM':
        embedding = nn.Embedding(15, 4, padding_idx=14)
        n = OLSTM(4, 5, 1, embedding=embedding)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MSELoss()  # the target label is not one-hotted.L1Loss()
    loss_func = nn.L1Loss()  # the target label is not one-hotted.L1Loss()
    for i in range(epoch):
        for x, y in zip(X, Y):
            prediction = n(x)#
            optimizer.zero_grad()  # clear gradients for this training step
            loss = loss_func(prediction, y)  # cross entropy loss[:,-1]
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        # if i % 49 == 0:
        Metric = TestMetricB(X_Test, Y_Test, n, type)
        print(Metric)#
        if i == 0:
            BestResult = Metric
            BestModel = n
        elif type == 1 and Metric > BestResult:
            BestResult = Metric
            BestModel = n
        elif type == 2 and Metric < BestResult:
            BestResult = Metric
            BestModel = n
    return BestResult, BestModel

def trian(Train,Test_X, Test_Y ,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for j, (x, y, l) in enumerate(Train):
                ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
                output, prediction = n(x, data)
                py = nn.utils.rnn.pack_padded_sequence(output[k], l, batch_first=True)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in ty.data:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(py.data, ty)
                else:
                    loss = loss_func(py.data.view(py.data.size(0)), ty.data)
                loss.backward()
                optimizer.step()
    Metric, abs_error = test(Test_X, Test_Y, n, type, data, 1, input_size)
    return n, Metric, abs_error

def trianD(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y in zip(Train_X, Train_Y):
                output, prediction = n(x, data)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in y:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(prediction, ty)
                else:
                    loss = loss_func(output[k][:,-1,0], y)
                loss.backward()
                optimizer.step()
    Metric, abs_error = test(Test_X, Test_Y, n, type, data, 1, input_size)
    return n, Metric, abs_error

def test(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                    true_y.append(line1)
                    pred_y.append(line2)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        abs_error = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
                    true_y.append(line1)
                    pred_y.append(line2)#[-1]
            Metric.append(mean_absolute_error(true_y, pred_y))
            abs_error.append(abs(np.array(true_y) - np.array(pred_y)).tolist())
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric, abs_error

class NNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(NNN, self).__init__()
        self.rnn1 = nn.LSTM(1, 8, num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(8, 128, num_layers, batch_first=True)
        self.fnn1 = nn.Linear(128, 64)
        self.fnn2 = nn.Linear(64, 1)

    def forward(self, x, data):
        nn_out1 = np.zeros([x.size(0),x.size(1),8])
        for i in range(x.size(0)):
            xi = self.rnn1(x[i,:,:].view(x.size(1),x.size(2),1))
            nn_out1[i] = xi[1][0].detach().numpy()
        nn_out2 = self.rnn2(torch.FloatTensor(nn_out1))
        nn_out3 = self.fnn1(nn_out2[0][:,-1,:])
        output = self.fnn2(nn_out3)
        return output

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=num_layers,  # number of rnn layer
            batch_first=True,
        )
        self.out1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out = self.rnn(x)
        # r_out = r_out[0].view(-1, 32)
        outs1 = self.out1(F.relu(r_out[0]))#
        outs = self.out2(F.relu(outs1))
        return outs.view(-1, x.size(1))#[:,-1]

class INN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(INN, self).__init__()
        self.input_size = input_size
        if type == 2:
            output_size = 1
        for i in range(input_size):#6
            if data['state'][0][i] == 0: # activity
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fnn1 = nn.Linear(hidden_size, int(hidden_size / 2))
                self.out1 = nn.Linear(int(hidden_size / 2), output_size)
                in_size = output_size
            # elif data['state'][0][i] == 1: # Static classification feature
            #     if str(i) in data.keys():
            #         setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(i)])))
            #         in_size += torch.tensor(data[str(i)]).size(1)
            #     else:
            #         in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*2))
            #     # setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size*2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: # classification feature
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                    in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size*2, in_size))
                setattr(self, 'out' + str(i + 1), nn.Linear(in_size, output_size))
                in_size = output_size
            # elif data['state'][0][i] == 3: # Static numerical features
            #     in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*4))
            #     setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size * 2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: # numerical features
                in_size += 1
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size*2, in_size))
                setattr(self, 'out' + str(i + 1), nn.Linear(in_size, output_size))
                in_size = output_size

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):#6:#len(data['state'][0])
            if data['state'][0][i] == 0: # activity
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 1:  # Static classification feature
            #     xi = x[:, :, i].view(x.size(0), x.size(1))
            #     if str(i) in data.keys():
            #         xi = torch.LongTensor(np.array(xi))
            #         xi = self.__getattr__('embed' + str(i + 1))(xi)
            #         x2 = torch.cat([nn_out.detach(), xi], dim=2)
            #     else:
            #         x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
            #     nn_out1 = self.__getattr__('fnn' + str(i + 1))(x2)
            #     # nn_out2 = self.__getattr__('fnnt' + str(i + 1))(nn_out1)
            #     nn_out = self.__getattr__('out' + str(i + 1))(nn_out1)
            #     output.append(nn_out)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # classification feature
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                    x2 = self.__getattr__('fnnf' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 3:  # Static numerical features
            #     xi = x[:, :, i].view(x.size(0), x.size(1))
            #     x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
            #     nn_out1 = self.__getattr__('fnn' + str(i + 1))(x2)
            #     nn_out2 = self.__getattr__('fnnt' + str(i + 1))(nn_out1)
            #     nn_out = self.__getattr__('out' + str(i + 1))(nn_out1)
            #     output.append(nn_out)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # numerical features
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                x2 = self.__getattr__('fnnf' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
        return output, nn_out#.view(x.size(0), x.size(1))

class INN2(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(INN2, self).__init__()
        if type == 2:
            output_size = 1
        for i in range(input_size):
            if data['state'][0][i] == 0: # activity
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fnn1 = nn.Linear(hidden_size, int(hidden_size / 2))
                self.out1 = nn.Linear(int(hidden_size / 2), output_size)
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 1: # Static classification feature
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, hidden_size))
                setattr(self, 'fnns' + str(i + 1), nn.Linear(hidden_size, int(hidden_size / 2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 2: # Dynamic classification feature
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, int(hidden_size/2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size/2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 3: # Static numerical features
                in_size += 1
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, hidden_size))
                setattr(self, 'fnns' + str(i + 1), nn.Linear(hidden_size, int(hidden_size / 2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 4: # Dynamic numerical feature
                in_size += 1
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, int(hidden_size/2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size/2), output_size))
                in_size = int(hidden_size / 2)

    def forward(self, x, data):
        output = []
        for i in range(len(data['state'][0])):
            if data['state'][0][i] == 0: # activity
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out = self.__getattr__('rnn' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 1:  # Static classification feature
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out1.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnns' + str(i + 1))(nn_out)
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 2:  # Dynamic classification feature
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out1.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 3:  # Static numerical features
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnns' + str(i + 1))(nn_out)
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 4:  # Dynamic numerical feature
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
        return output, nn_out#.view(x.size(0), x.size(1))

class FNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(FNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.out2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        h_out = self.hidden(x)
        outs1 = self.out1(torch.sigmoid(h_out))
        outs = self.out2(torch.sigmoid(outs1))
        return outs.view(-1, x.size(1))[:,-1]

class OLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_size, batch_size=20, n_layer=1, dropout=0, embedding = None):
        super(OLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                           num_layers=self.n_layer, bidirectional=False)
        self.out = nn.Linear(hidden_dim, out_size)
    def forward(self, X):
        X = X.view(X.size(0),X.size(1)).long()
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.rnn(input)
        hn = output[-1]
        output = self.out(hn)
        return output