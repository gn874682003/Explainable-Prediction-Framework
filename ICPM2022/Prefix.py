import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.utils.data as data_
from torch import nn

#Whole trace as prefix
def changeLen(data,feature,label,batchSize):
    # Trace length ascending sort
    # data.sort(key=lambda i: len(i), reverse=False)
    # get important features
    x = []
    y = []
    X = []
    Y = []
    num = 0
    if batchSize - 1 > len(data):
        length = len(data[-1])
    else:
        length = len(data[batchSize - 1])
    fn = [0 for i in range(len(feature))]
    for line1 in data:
        if num % batchSize == 0 and num != 0:
            X.append(torch.Tensor(x[num - batchSize:num]))
            Y.append(torch.Tensor(y[num - batchSize:num]))
            if num + batchSize - 1 > len(data):#
                length = len(data[-1])
            else:
                length = len(data[num + batchSize - 1])
        num += 1
        tempx = []
        tempy = []
        for line2 in line1:
            temp = []
            for i in feature:
                temp.append(line2[i])
            tempx.append(temp)
            tempy.append(line2[label])
        while len(tempx) != length:
            tempx.append(fn)
            tempy.append(0)
        x.append(tempx)
        y.append(tempy)
    if batchSize == 1:
        X.append(torch.Tensor(x[num - batchSize:num]))
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
    return X,Y

# Intercept by fixed prefix length
def cutPrefixBy(data,feature,label,batchSize,LEN):
    # get important features
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

# Trace segmentation prefix, LEN=10
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

        for i in range(len(prex)-1):
            for j in range(i, min(len(prex), LEN)):
                a.append([prex[i:j+1], prey[j]])

        # for j in range(0, min(len(prex), LEN)):
        #     a.append([prex[0:j + 1], prey[j]])
        # for i in range(1, len(prex) - LEN + 1):
        #     a.append([prex[i:i + LEN], prey[i + LEN - 1]])

    # Trace length ascending sort
    a.sort(key=lambda i: len(i[0]), reverse=False)
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

# word2vector corpus preparation
def sentence(data):
    sentence = []
    for line1 in data:
        trace = []
        for line2 in line1:
            trace.append(line2[0])
        sentence.append(trace)
    return sentence

def NoFill(data,feature,label,batchSize):
    # Trace length ascending sort
    data.sort(key=lambda i: len(i), reverse=False)
    # get important features
    x = []
    y = []
    for line1 in data:
        tempx = []
        tempy = []
        for line2 in line1:
            temp = []
            for i in feature:
                temp.append(line2[i])
            tempx.append(temp)
            tempy.append(line2[label])
        x.append(torch.Tensor(tempx))
        y.append(torch.Tensor(tempy))
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

# data_tuple is a list, including batchsize tuple, each tuple contains data and labels
def collate_fn(data_tuple):
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data0 = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    data_length = [len(sq) for sq in data0]
    # Zero complement to align the length, change to tensor
    data = nn.utils.rnn.pad_sequence(data0, batch_first=True, padding_value=0.0)
    label = nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0.0)
    return data, label, data_length