import copy

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from torch.optim import lr_scheduler
import numpy as np
import torch
import os
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap

LR = 0.001

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=20, verbose=False, delta=0.1):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def get_att_dis(target, behaviored):#计算余弦相似度
    attention_distribution = []
    result = []
    for j in range(target.shape[1]):
        for i in range(behaviored.shape[0]):
            attention_score = torch.cosine_similarity(target[0,j,:].view(1, -1), torch.FloatTensor(behaviored[i]).view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
            attention_distribution.append(attention_score)
        result.append(int(torch.argmax(torch.Tensor(attention_distribution))))
        attention_distribution = []
    return  result

def viewResult(X_Test, Y_Test, rnn, type, data,caseid, ConvertReflact,attribute):
    for x, y,cid in zip(X_Test, Y_Test,caseid):
        if cid == 'Case 889' or cid == 'Case 4130' or cid == 'Case 2336' or cid == 'Case 2155':
            output, prediction = rnn(x, data)
            print(cid)
            # print(x)
            print(output)
            for j in range(2):#prediction.size(1)
                yi = 0
                labels = ['Assign seriousness', 0.0, 'Value 2', 'Value 3']#10.0387,11.,,7.3245
                lablen = []
                lablea = []
                # for line, label in zip(output, data['name']):
                for line, i in zip(output, range(len(output))):
                    if data['state'][0][i] < 3:
                        label = ConvertReflact[attribute.index(data['index'][0][i]+3)][int(x[0][j][i])]
                        lablen.append(label)
                    else:
                        label = float(x[0][j][i])
                    lablea.append(label)
                    plt.scatter(line.view(line.size(1))[j].detach().numpy(), yi)
                    # plt.annotate(label, xy=(line.view(line.size(1))[j].detach().numpy(), yi), xytext=(5, 2),
                    #              textcoords='offset points', ha='right', va='bottom')
                    yi += 1
                print(lablea)
                # if lablea == labels:
                #     print('end')
                plt.vlines(y[0][j],0,yi-1)#
                # y_major_locator = MultipleLocator(1)
                # ax = plt.gca()
                # ax.yaxis.set_major_locator(y_major_locator)
                # plt.xlabel('Time(day)')
                # plt.ylabel('Feature Number')
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
                true_y.extend(y.numpy().tolist()[0])
                pred_y.extend(prediction.detach().numpy().tolist()[0])#[-1]
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def prefixPlot(X_Test, Y_Test, rnn, data, input_size, X_Train, Y_Train):
    MetricAll = []
    for j in range(input_size):
        Metric = []
        tPrefix = {}
        pPrefix = {}
        for x, y in zip(X_Test, Y_Test):
            output, prediction = rnn(x, data)
            prediction = output[j]
            prediction = prediction.view(prediction.size(0),prediction.size(1))
            true_y = y.numpy().tolist()[0]
            pred_y = prediction.detach().numpy().tolist()[0]
            for i in range(1,len(true_y)+1):
                if i in tPrefix.keys():
                    tPrefix[i].append(true_y[i-1])
                    pPrefix[i].append(pred_y[i-1])
                else:
                    tPrefix[i] = [true_y[i-1]]
                    pPrefix[i] = [pred_y[i-1]]
        for i in tPrefix.keys():
            Metric.append(mean_absolute_error(tPrefix[i], pPrefix[i]))
        print(Metric)
        MetricAll.append(Metric)

    numPrefix = {}
    for x, y in zip(X_Train, Y_Train):
        true_y = y.numpy().tolist()[0]
        for i in range(1,len(true_y)+1):
            if i in numPrefix.keys():
                numPrefix[i].append(true_y[i-1])
            else:
                numPrefix[i] = [true_y[i-1]]
    # # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    # x = np.arange(len(Metric))+1
    # # 有a/b/c三种类型的数据，n设置为3
    # total_width, n = 0.8, len(MetricAll)
    # # 每种类型的柱状图宽度
    # width = total_width / n
    # # 重新设置x轴的坐标
    # x = x - (total_width - width) / 2
    # print(x)
    # # 画柱状图
    # plt.bar(x, MetricAll[0], width=width, label="Layer 1: Activity", color='mistyrose')
    # plt.bar(x + width, MetricAll[1], width=width, label="Layer 2: allDuration", color='lightsalmon')
    # plt.bar(x + 2 * width, MetricAll[2], width=width, label="Layer 3: seriousness 2", color='salmon')
    # plt.bar(x + 3 * width, MetricAll[3], width=width, label="Layer 4: service_level", color='tomato')
    # # 显示图例
    # plt.legend()
    # plt.xlabel("Prefix Length")
    # plt.ylabel("MAE")
    # # 显示柱状图
    # plt.show()


    y2 = [len(tPrefix[i]) for i in tPrefix.keys()]
    # y2 = [len(numPrefix[i]) for i in numPrefix.keys()]
    # x2 = [i+1 for i in range(len(y2))]
    x = [i+1 for i in range(len(Metric))]
    # 创建一个图形和两个y轴
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # 绘制图形
    bar = ax1.bar(x, Metric, color='tomato', width=0.4)#, label='MAE'
    line = ax2.plot(x, y2, color='royalblue', marker='o', ls='-.')#, label='Number of trace'
    # 设置x轴和y轴的标签，指明坐标含义
    # ax1.set_xlabel('Prefix Length', fontdict={'size': 12})
    # ax1.set_ylabel('MAE', fontdict={'size': 12})
    # ax2.set_ylabel('Number of trace', fontdict={'size': 12})
    # 添加图表题
    # plt.title('Prefix')
    # 添加图例
    lines = [line[0], bar]
    labels = [h.get_label() for h in lines]
    plt.legend(lines, labels, loc='upper right')
    # 设置轴标签颜色
    ax1.tick_params('x', labelsize=16)
    ax1.tick_params('y', colors='tomato', labelsize=16)
    ax2.tick_params('y', colors='royalblue', labelsize=16)
    # 设置轴颜色
    ax1.spines['left'].set_color('tomato')
    ax2.spines['left'].set_color('tomato')
    ax1.spines['right'].set_color('royalblue')
    ax2.spines['right'].set_color('royalblue')
    # 去掉上轴线
    # ax1.spines['top'].set_visible(False)#bottom
    # ax2.spines['top'].set_visible(False)
    # 设置中文显示
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 展示图片
    plt.show()

    # for i in range(len(Metric)):
    #     plt.bar(i+1, Metric[i])
    # plt.title('Prefix')
    # plt.xlabel("Prefix Length")
    # plt.ylabel("MAE")
    # plt.show()
    return Metric

def test(X_Test, Y_Test, rnn, type, data):
    eval_loss = 0
    loss_func = nn.L1Loss()
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for x, y in zip(X_Test, Y_Test):
            output, prediction = rnn(x, data)
            prediction = output
            # prediction = get_att_dis(prediction, data['0'])
            for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(accuracy_score(true_y, pred_y))
    else:
        Metric = []
        represents = []
        for x, y in zip(X_Test, Y_Test):
            output, represent = rnn(x, data)#
            output = output.view(output.size(0),output.size(1))
            # represents.append(represent)
            # 记录误差
            loss = loss_func(output, y)
            eval_loss += loss.item()
            for line1, line2 in zip(y.numpy().tolist()[0], output.detach().numpy().tolist()[0]):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(mean_absolute_error(true_y, pred_y))
    return Metric, eval_loss#len(true_y), represents

def trian(Train,Test_X, Test_Y ,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None,isEarly=0):
    # LR = 0.001
    save_path = '../Save/multiFea/'  # 当前目录下
    early_stopping = EarlyStopping(save_path)
    train_loss = 0
    if method == 'rnn':
        method = RNN(input_size, hidden_size, num_layers, data)
    optimizer = torch.optim.Adam(method.parameters(), lr=LR)
    # 1.学习率下降：每过n个epoch，学习率乘以0.1
    # scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for j, (x, y, l) in enumerate(Train):
            ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
            output, represents = method(x, data)#
            py = nn.utils.rnn.pack_padded_sequence(output, l, batch_first=True)
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
            # 2.梯度剪裁
            nn.utils.clip_grad_norm_(method.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
        # scheduler.step()
        if isEarly == 1:
            Metric, eval_loss = test(Test_X, Test_Y, method, type, data, input_size)
            # 早停止
            early_stopping(eval_loss, method)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
    if isEarly == 0:
        Metric, count = test(Test_X, Test_Y, method, type, data)
    print(i, Metric)
    return method, Metric#, count

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data):
        super(RNN, self).__init__()
        self.input_size = input_size
        if input_size == -1:  # 全部拼接，索引编码
            input_size = len(data['index'][0])
        elif input_size == -2:  # 全部拼接，CBOW和向量编码
            input_size = 0
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim
                        if type == 1:
                            output_size = input_size
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        setattr(self, 'embed' + str(i + 1),
                                nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                        input_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                    else:
                        input_size += 1
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    input_size += 1
        elif input_size == -3:  # 全部拼接，活动CBOW和其他索引编码
            input_size = 0
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim+len(data['index'][0])-1
        elif input_size == 0:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(data['0']))
            input_size = self.embed.embedding_dim
        hidden_size = input_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size*2,
            num_layers=num_layers,
            dropout=0.2,  # 3.丢弃层
            batch_first=True
        )
        # self.out = nn.Linear(hidden_size, 1)
        self.out1 = nn.Linear(hidden_size*2, hidden_size)
        self.out2 = nn.Linear(hidden_size, 1)#int(hidden_size/2)
        # 正则初始化
        nn.init.orthogonal(self.rnn.weight_ih_l0)
        nn.init.orthogonal(self.rnn.weight_hh_l0)
        # nn.init.zeros_(self.rnn.bias_ih_l0)
        # nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, x, data):
        if self.input_size == -1:  # 全部拼接，索引编码
            r_out = self.rnn(x)
        elif self.input_size == -2:  # 全部拼接，CBOW和向量编码
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        xi = x[:, :, i].view(x.size(0), x.size(1))
                        xi = torch.LongTensor(np.array(xi))
                        xi = self.__getattr__('embed' + str(i + 1))(xi)
                    else:
                        xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
        elif self.input_size == -3:  # 全部拼接，活动CBOW和其他索引编码
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                else:  # 其他
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
        elif self.input_size == 1:  # 仅活动，索引编码
            r_out = self.rnn(x[:, :, 0:1])
        elif self.input_size == 0:  # 仅活动，CBOW编码
            xi = torch.tensor(x[:, :, 0], dtype=torch.int64)
            xi = self.embed(xi)
            r_out = self.rnn(xi)
        else:  # 仅活动，One-hot编码
            ohx = np.eye(self.input_size)[torch.tensor(x[:, :, 0], dtype=torch.int64)]
            ohx = torch.tensor(ohx, dtype=torch.float32)
            if ohx.shape.__len__() == 1:
                ohx = ohx.view(1, 1, -1)
            r_out = self.rnn(ohx)
        # outs = self.out(r_out[0])  # F.relu()
        outs1 = self.out1(r_out[0])
        outs = self.out2(outs1)
        return outs.view(-1, x.size(1)), r_out[1]

def TestMetric(X_Test, Y_Test, rnn, type, data, flag, input_size):
    eval_loss = 0
    loss_func = nn.L1Loss()
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
                # 记录误差
                loss = loss_func(prediction, y)
                eval_loss += loss.item()
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
    return Metric, eval_loss

def LSTMNewTwo(Train,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None,display=0):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    print(n)
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
        if (i+1) % 20 == 0:  # display == 1 and
            Metric, _ = TestMetric(Test_X, Test_Y, n, type, data,1,input_size)
            print(i+1, Metric)
            if i == 19:
                BestAll = Metric[-1]
                BestModelAll = copy.deepcopy(n)
                bestEpoch = 300
                bestLayer = input_size
            elif type == 1 and max(Metric) > BestAll:
                BestAll = max(Metric)
                BestModelAll = copy.deepcopy(n)
                bestEpoch = i
                bestLayer = Metric.index(min(Metric)) + 1
            elif type == 2 and min(Metric) < BestAll:
                BestAll = min(Metric)
                BestModelAll = copy.deepcopy(n)
                bestEpoch = i
                bestLayer = Metric.index(min(Metric)) + 1
    return BestAll, BestModelAll, bestEpoch+1, bestLayer, n, loss

def LSTMNewTwoEarly(Train,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None, LR=0.001,opt='Adam'):
    save_path = '../Save/multiFea/'  # 当前目录下
    early_stopping = EarlyStopping(save_path)
    train_loss = 0
    isEarly = 0
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    print(n)
    if opt == 'Adam':
        optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    else:
        optimizer = torch.optim.SGD(n.parameters(), lr=LR)
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
                # 记录误差
                train_loss += loss.item()
        if isEarly == 1:
            Metric, eval_loss = TestMetric(Test_X, Test_Y, method, type, data, input_size)
            # 早停止
            early_stopping(eval_loss, method)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
        if isEarly == 0:
            if (i+1) % 20 == 0:  # display == 1 and
                Metric, eval_loss = TestMetric(Test_X, Test_Y, n, type, data,1,input_size)
                print(i+1, Metric)
                if i == 19:
                    BestAll = Metric[-1]
                    BestModelAll = copy.deepcopy(n)
                    bestEpoch = 300
                    bestLayer = input_size
                elif type == 1 and max(Metric) > BestAll:
                    BestAll = max(Metric)
                    BestModelAll = copy.deepcopy(n)
                    bestEpoch = i
                    bestLayer = Metric.index(min(Metric)) + 1
                elif type == 2 and min(Metric) < BestAll:
                    BestAll = min(Metric)
                    BestModelAll = copy.deepcopy(n)
                    bestEpoch = i
                    bestLayer = Metric.index(min(Metric)) + 1
    return BestAll, BestModelAll, bestEpoch+1, bestLayer, n, loss

class INN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(INN, self).__init__()
        self.input_size = input_size
        if type == 2:
            output_size = 1
        for i in range(input_size):#6
            if data['state'][0][i] == 0: # 活动
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                else:
                    in_size = 1
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fnn1 = nn.Linear(hidden_size, output_size)
                # self.out1 = nn.Linear(int(hidden_size / 2), output_size)
                in_size = output_size
            # elif data['state'][0][i] == 1: #静态分类
            #     if str(i) in data.keys():
            #         setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(i)])))
            #         in_size += torch.tensor(data[str(i)]).size(1)
            #     else:
            #         in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*2))
            #     # setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size*2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #动态分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                    # in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,#in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, output_size))
                # setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = output_size
            # elif data['state'][0][i] == 3: #静态数值
            #     in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*4))
            #     setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size * 2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #动态数值
                in_size += 1
                # setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                # in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,#in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, output_size))
                # setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = output_size

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):#6:#len(data['state'][0])
            if data['state'][0][i] == 0: # 活动
                xi = x[:, :, i].view(x.size(0), x.size(1),1)
                if str(i) in data.keys():
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)
                nn_out = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                # nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 1:  # 静态分类
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
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 动态分类
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                # nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 3:  # 静态数值
            #     xi = x[:, :, i].view(x.size(0), x.size(1))
            #     x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
            #     nn_out1 = self.__getattr__('fnn' + str(i + 1))(x2)
            #     nn_out2 = self.__getattr__('fnnt' + str(i + 1))(nn_out1)
            #     nn_out = self.__getattr__('out' + str(i + 1))(nn_out1)
            #     output.append(nn_out)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 动态数值
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                # x2 = self.__getattr__('fnnf' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                # nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
        return output, nn_out#.view(x.size(0), x.size(1))

def testT(X_Test, Y_Test, model, type, data):
    eval_loss = 0
    loss_func = nn.L1Loss()
    pred_y = []
    true_y = []
    Metric = []
    abs_error = []
    represents = []
    for x, y in zip(X_Test, Y_Test):
        batch = x.shape[0]
        length = x.shape[1]
        PosEncode = torch.zeros(batch, length)
        for ii in range(batch):
            for jj in range(length):
                PosEncode[ii][jj] = jj
        prediction, represent = model(x, PosEncode, data)
        represents.append(represent)
        prediction = prediction.view(prediction.shape[0], prediction.shape[1])
        # 记录误差
        loss = loss_func(prediction, y)
        eval_loss += loss.item()
        for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
            true_y.append(line1)
            pred_y.append(line2)
    Metric.append(mean_absolute_error(true_y, pred_y))
    return Metric, len(pred_y), represents

def trianT(Train, Test_X, Test_Y,epoch,type,mcl,method=None,data=None):
    save_path = '../Save/multiFea/'  # 当前目录下
    early_stopping = EarlyStopping(save_path)
    train_loss = 0
    if isinstance(method, str):
        n = Transformer(mcl, data)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    # 学习率下降：每过n个epoch，学习率乘以0.1
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        scheduler.step()
        for j, (x, y, l) in enumerate(Train):
            batch = x.shape[0]
            length = x.shape[1]
            PosEncode = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(length):
                    PosEncode[ii][jj] = jj
            padding = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(l[ii]):
                    padding[ii][jj] = 1
            output, _ = n(x, PosEncode, data)
            py = output.view(output.shape[0], output.shape[1])
            optimizer.zero_grad()
            loss = loss_func(py, y)
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
    Metric, count, _ = testT(Test_X, Test_Y, n, type, data)
        # 早停止
        # early_stopping(eval_loss, n)
        # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练
    # print(i, Metric, count)
    return n, Metric, count

class Transformer(nn.Module):
    def __init__(self, pos_dim, data, hidden=128, nhead=8, num_layers=3, dim_feedforward=64, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(Transformer, self).__init__()
        self.dim = 0
        for i in range(len(data['index'][0])):
            if data['state'][0][i] < 3:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    self.dim += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    self.dim += 1
            else:  # 数值
                self.dim += 1
        m = self.dim % nhead
        if m > 0:
            self.dim = self.dim + nhead - m
        self.pos = nn.Embedding(pos_dim, self.dim)
        encoder_layer = nn.TransformerEncoderLayer(self.dim, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(self.dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pooling = nn.AdaptiveAvgPool1d(self.dim)
        self.fnn1 = nn.Linear(self.dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(hidden, 1)

    def forward(self, x, PosEncode, data):
        for i in range(len(data['index'][0])):
            if data['state'][0][i] == 0:  # 活动
                xc = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xc = torch.LongTensor(np.array(xc))
                    inputc = self.__getattr__('embed' + str(i + 1))(xc)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    xc = x[:, :, i].view(x.size(0), x.size(1))
                    xc = torch.LongTensor(np.array(xc))
                    xc = self.__getattr__('embed' + str(i + 1))(xc)
                else:
                    xc = x[:, :, i].view(x.size(0), x.size(1), 1)
                inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xc = x[:, :, i].view(x.size(0), x.size(1), 1)
                inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
        m = inputc.size(2)
        if m != self.dim:
            xc = torch.zeros(inputc.size(0), inputc.size(1), self.dim - m)
            inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
        pos = self.pos(PosEncode.long())
        memory = self.encoder(inputc + pos)
        out1 = self.pooling(memory)
        out2 = self.fnn1(out1)
        output = self.dropout(out2)
        output = self.fnn2(output)  # F.relu()
        return output, memory[:, -1, :]

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
