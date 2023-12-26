import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.utils.data as data_
from torch import nn

#一整条轨迹
def changeLen(data,feature,label,batchSize):
    # 轨迹长度升序排序
    data.sort(key=lambda i: len(i), reverse=False)
    # 取出重要特征值
    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    X = []
    Y = []
    Y1 = []
    Y2 = []
    Y3 = []
    num = 0
    if batchSize - 1 > len(data):
        length = len(data[-1])
    else:
        length = len(data[batchSize - 1])
    fn = [0 for i in range(len(feature))]
    for line1 in data:
        if num % batchSize == 0 and num != 0:
            X.append(torch.Tensor(x[num - batchSize:num]))
            if label == 0:
                Y1.append(torch.Tensor(y1[num - batchSize:num]))
                Y2.append(torch.Tensor(y2[num - batchSize:num]))
                Y3.append(torch.Tensor(y3[num - batchSize:num]))
            else:
                Y.append(torch.Tensor(y[num - batchSize:num]))
            if num + batchSize - 1 > len(data):#
                length = len(data[-1])
            else:
                length = len(data[num + batchSize - 1])
        num += 1
        tempx = []
        if label == 0:
            tempy1 = []
            tempy2 = []
            tempy3 = []
        else:
            tempy = []
        for line2 in line1:
            temp = []
            for i in feature:
                temp.append(line2[i])
            tempx.append(temp)
            if label == 0:
                tempy1.append(line2[-1])
                tempy2.append(line2[-3])
                tempy3.append(line2[-2])
            else:
                tempy.append(line2[label])
        while len(tempx) != length:
            tempx.append(fn)
            tempy.append(0)
        x.append(tempx)
        if label == 0:
            y1.append(tempy1)
            y2.append(tempy2)
            y3.append(tempy3)
        else:
            y.append(tempy)
    if batchSize == 1:
        X.append(torch.Tensor(x[num - batchSize:num]))
        if label == 0:
            Y1.append(torch.Tensor(y1[num - batchSize:num]))
            Y2.append(torch.Tensor(y2[num - batchSize:num]))
            Y3.append(torch.Tensor(y3[num - batchSize:num]))
        else:
            Y.append(torch.Tensor(y[num - batchSize:num]))
    if num % batchSize != 0:
        while num % batchSize != 0:
            rand = random.randint(0, num - batchSize)
            tempx = x[rand]
            tempy = y[rand]
            while len(tempx) != length:
                tempx.append(fn)
                tempy.append(0)
            x.append(tempx)
            y.append(tempy)
            num += 1
        X.append(torch.Tensor(x[num - batchSize:num]))
        Y.append(torch.Tensor(y[num - batchSize:num]))
    if label == 0:
        return X, [Y1, Y2, Y3]
    else:
        return X, Y

#按给定前缀长度截取
def cutPrefixBy(data,feature,label,batchSize,LEN):
    # 取出重要特征值
    x = []
    y = []
    X = []
    Y = []
    fn = [0 for i in range(len(feature))]
    for line1 in data:
        prex = []
        prey = []
        # for i in range(LEN - 1):
        #     prex.append(fn)
        #     prey.append(0)
        for line2 in line1:
            pre = []
            for i in feature:
                pre.append(line2[i])
            prex.append(pre)
            prey.append(line2[label])
        for i in range(len(prex)-LEN+1):
            tempx = prex[i:i+LEN]
            tempy = prey[i+LEN-1]
            x.append(tempx)
            y.append(tempy)
            if len(x) == batchSize:
                X.append(torch.Tensor(x))
                Y.append(torch.Tensor(y))
                x = []
                y = []
    if len(x) != 0:
        while len(x) != batchSize:
            rand = random.randint(0, len(x)-1)
            x.append(x[rand])
            y.append(y[rand])
        X.append(torch.Tensor(x))
        Y.append(torch.Tensor(y))
    return X, Y

#上文不同窗口的前缀，最大窗口长度设为LEN=10
def diffWindow(data,feature,label,batchSize,LEN=10):
    x = []
    y = []
    X = []
    Y = []
    a = []
    fn = [0 for i in range(len(feature))]#0#
    for line1 in data:
        prex = []
        prey = []
        for line2 in line1:
            pre = []
            for i in feature:
                if i >= len(line2):
                    print(line2)
                pre.append(line2[i])
            prex.append(pre)
            prey.append(line2[label])
        for j in range(0, min(len(prex), LEN)):
            a.append([prex[0:j + 1], prey[j]])
        for i in range(1, len(prex) - LEN + 1):
            a.append([prex[i:i + LEN], prey[i + LEN - 1]])
        # for i in range(len(prex)-LEN+1):
        #     if i == 0:
        #         for j in range(0, min(len(prex), LEN)):
        #             a.append([prex[i:j+1], prey[j]])
        #     else:
        #         a.append([prex[i:i + LEN], prey[i + LEN-1]])

        # for i in range(len(prex)-1):
        #     for j in range(i, min(len(prex), LEN)):
        #         a.append([prex[i:j+1], prey[j]])

    # 轨迹长度升序排序
    # a.sort(key=lambda i: len(i[0]), reverse=False)
    if batchSize - 1 > len(a):
        length = len(a[-1][0])
    else:
        length = len(a[batchSize - 1][0])

    for line in a:
        tempx = line[0]
        if len(x) == batchSize:
            X.append(torch.Tensor(x))
            Y.append(torch.Tensor(y))
            x = []
            y = []
            if (len(X)+1)*batchSize - 1 > len(a):
                length = len(a[-1][0])
            else:
                length = len(a[(len(X)+1)*batchSize - 1][0])

        while len(tempx) != length:
            tempx.append(fn)
        x.append(tempx)
        y.append(line[1])

    while len(x) != batchSize:
        rand = random.randint(0, len(x) - 1)
        x.append(x[rand])
        y.append(y[rand])
    X.append(torch.Tensor(x))
    Y.append(torch.Tensor(y))
    return X, Y

#word2vector语料准备
def sentence(data):
    sentence = []
    for line1 in data:
        trace = []
        for line2 in line1:
            trace.append(line2[0])
        sentence.append(trace)
    return sentence

def NoFill(data,feature,label,batchSize):
    # 轨迹长度升序排序
    data.sort(key=lambda i: len(i), reverse=False)
    # 取出重要特征值
    x = []
    if label == 0:
        y1 = []
        y2 = []
        y3 = []
    else:
        y = []
    for line1 in data:
        tempx = []
        if label == 0:
            tempy1 = []
            tempy2 = []
            tempy3 = []
        else:
            tempy = []
        for line2 in line1:
            temp = []
            for i in feature:
                temp.append(line2[i])
            tempx.append(temp)
            if label == 0:
                tempy1.append(line2[-1])
                tempy2.append(line2[-3])
                tempy3.append(line2[-2])
            else:
                tempy.append(line2[label])
        x.append(torch.Tensor(tempx))
        if label == 0:
            y1.append(torch.Tensor(tempy1))
            y2.append(torch.Tensor(tempy2))
            y3.append(torch.Tensor(tempy3))
        else:
            y.append(torch.Tensor(tempy))
    if label == 0:
        data_ = MyDataMulti(x, y1, y2, y3)
        batch = DataLoader(data_, batch_size=batchSize, shuffle=False, drop_last=True, collate_fn=collate_fnMulti)
    else:
        data_ = MyData(x, y)
        batch = DataLoader(data_, batch_size=batchSize, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return batch

class MyData(data_.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx])
        return tuple_

class MyDataMulti(data_.Dataset):
    def __init__(self, data, label1, label2, label3):
        self.data = data
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label1[idx], self.label2[idx], self.label3[idx])
        return tuple_

def collate_fnMulti(data_tuple):  # data_tuple是一个列表，列表中包含batchsize个元组，每个元组中包含数据和标签
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data0 = [sq[0] for sq in data_tuple]
    label1 = [sq[1] for sq in data_tuple]
    label2 = [sq[2] for sq in data_tuple]
    label3 = [sq[3] for sq in data_tuple]
    data_length = [len(sq) for sq in data0]
    data = nn.utils.rnn.pad_sequence(data0, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐，变为tensor
    label1 = nn.utils.rnn.pad_sequence(label1, batch_first=True, padding_value=0.0)
    label2 = nn.utils.rnn.pad_sequence(label2, batch_first=True, padding_value=0.0)
    label3 = nn.utils.rnn.pad_sequence(label3, batch_first=True, padding_value=0.0)
    return data, label1, label2, label3, data_length

def collate_fn(data_tuple):  # data_tuple是一个列表，列表中包含batchsize个元组，每个元组中包含数据和标签
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data0 = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    data_length = [len(sq) for sq in data0]
    data = nn.utils.rnn.pad_sequence(data0, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐，变为tensor
    label = nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0.0)
    return data, label, data_length