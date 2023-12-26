import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from BPP_Frame.Method.AETS.utils.construct_transition_system import *
from BPP_Frame.Method.AETS.utils.metrics import *

def getTrainAndTestData(Train, Test, event_list, data):
    initTrans = []
    for i in range(0, len(event_list)):
        initTrans.append(0)
    train_vec = []
    train_sel = []
    train_label = []
    train_event = []
    # train dataset construct
    if Train != None:
        for trace in Train:
            for event in trace[:-1]:
                remainTime = float(event[-1])
                currEventIndex = trace.index(event)
                transVec = initTrans.copy()
                if currEventIndex >= 2:
                    tempEvent = []
                    for i in range(0, currEventIndex + 1):
                        event_ = trace[i][0]
                        currEventTempIndex = event_list.index(event_)
                        transVec[currEventTempIndex] += 1
                        tempEvent.append(event_)
                    train_vec.append(transVec)
                    train_sel.append([event[i] for i in data])
                    train_label.append(remainTime)
                    tempEvent.sort()
                    train_event.append(tempEvent)
    test_vec = []
    test_sel = []
    test_label = []
    test_event = []
    # test dataset construct
    if Test != None:
        for trace in Test:
            for event in trace[:-1]:
                remainTime = float(event[-1])
                currEventIndex = trace.index(event)
                transVec = initTrans.copy()
                if currEventIndex >= 2:
                    tempEvent = []
                    for i in range(0, currEventIndex + 1):
                        event_ = trace[i][0]
                        currEventTempIndex = event_list.index(event_)
                        transVec[currEventTempIndex] += 1
                        tempEvent.append(event_)
                    test_vec.append(transVec)
                    test_sel.append([event[i] for i in data])
                    test_label.append(remainTime)
                    tempEvent.sort()
                    test_event.append(tempEvent)
    return train_sel, test_sel, train_vec, test_vec, train_label, test_label, train_event, test_event

def train(AllData,Train,Test,method, data, modelDict=None, isMul=0):
    state_dict, edge_list, edgeCount_list, event_list = readFile(AllData, method)
    stateIndex = list(state_dict.keys())
    stateValue = []
    for state in state_dict.values():
        tem1 = []
        for key in state.keys():
            for num in range(0, state[key]):
                tem1.append(key)
        tem1.sort()
        stateValue.append(tem1)
    train_sel, test_sel, train_vec, test_vec, train_label, test_label, train_event, test_event = \
        getTrainAndTestData(Train, Test, event_list, data)

    tensor_x = torch.from_numpy(np.array(train_vec).astype(np.float32))#.cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32))#.cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
    model = autoencoder(tensor_x.shape[1])
    criterion = nn.MSELoss()#.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            optimizer.zero_grad()
            _, pred = model(Variable(x))#.cuda()).cuda())
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            total_loss += loss
        # if epoch % 100 == 0:
            # print(str(total_loss.data))
    trainLowDim = []
    testLowDim = []
    for x,y in zip(train_vec,train_sel):
        x_ = Variable(torch.Tensor(np.array(x)))#.cuda()).cuda()
        _, pred = model(Variable(x_))#.cuda()).cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            trainLowDim.append(input)
        else:
            trainLowDim.append(_.detach().cpu().numpy().tolist())
    for x,y in zip(test_vec,test_sel):
        x_ = Variable(torch.Tensor(np.array(x)))  # .cuda()).cuda()
        _, pred = model(Variable(x_))  # .cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            testLowDim.append(input)
        else:
            testLowDim.append(_.detach().cpu().numpy().tolist())

    singleTrainVec = {}
    singleTrainLabel = {}
    modelDict = {}
    for index in stateIndex:
        singleTrainVec[index] = []
        singleTrainLabel[index] = []
    for eventIndex in range(0, len(train_event)):
        currStateIndex = stateValue.index(train_event[eventIndex])
        singleTrainVec[currStateIndex].append(trainLowDim[eventIndex])
        singleTrainLabel[currStateIndex].append(train_label[eventIndex])
    singleTestVec = {}
    singleTestLabel = {}
    for index in stateIndex:
        singleTestVec[index] = []
        singleTestLabel[index] = []
    for eventIndex in range(0, len(test_event)):
        currStateIndex = stateValue.index(test_event[eventIndex])
        singleTestVec[currStateIndex].append(testLowDim[eventIndex])
        singleTestLabel[currStateIndex].append(test_label[eventIndex])
    tensor_x = torch.from_numpy(np.array(trainLowDim).astype(np.float32))  # .cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32))  # .cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
    modelTotal = MLP(tensor_x.shape[1])
    criterionLinear = nn.L1Loss()  # .cuda()
    optimizerLinear = optim.Adam(modelTotal.parameters(), lr=0.0001)
    for epoch in range(100):
        # print("total train epoch:" + str(epoch))
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            pred = modelTotal(Variable(x))#.cuda()).cuda())
            pred = pred.squeeze(-1)
            loss = criterionLinear(pred, y)
            optimizerLinear.zero_grad()
            loss.backward()
            optimizerLinear.step()
            total_loss += loss
    # transfer learning train
    for index in singleTrainVec.keys():
        if len(singleTrainVec[index]) < 10:
            continue
        tensor_x = torch.from_numpy(np.array(singleTrainVec[index]).astype(np.float32))#.cuda()
        tensor_y = torch.from_numpy(np.array(singleTrainLabel[index]).astype(np.float32))#.cuda()
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataset_loader = DataLoader(my_dataset, batch_size=8, shuffle=False)
        modelLinear = copy.deepcopy(modelTotal)
        criterionLinear = nn.L1Loss()#.cuda()
        optimizerLinear = optim.Adam(modelLinear.parameters(), lr=0.00001)
        for epoch in range(100):
            total_loss = 0
            for i, (x, y) in enumerate(my_dataset_loader):
                pred = modelLinear(Variable(x))#.cuda()).cuda())
                pred = pred.squeeze(-1)
                loss = criterionLinear(pred, y)
                optimizerLinear.zero_grad()
                loss.backward()
                optimizerLinear.step()
                total_loss += loss
            # print("total_loss = " + str(total_loss))
        modelDict[index] = modelLinear
    MAE, MSE, _ = predict(singleTestVec, singleTestLabel, modelDict)
    return modelDict, MAE, model, modelTotal, event_list

def update(AllData, Train, modelDict, model ,method, data, isMul=0):
    state_dict, edge_list, edgeCount_list, event_list = readFile(AllData, method)
    stateIndex = list(state_dict.keys())
    stateValue = []
    for state in state_dict.values():
        tem1 = []
        for key in state.keys():
            for num in range(0, state[key]):
                tem1.append(key)
        tem1.sort()
        stateValue.append(tem1)
    train_sel, _, train_vec, _, train_label, _, train_event, _ = getTrainAndTestData(Train, None, event_list, data)

    trainLowDim = []
    for x, y in zip(train_vec, train_sel):
        x_ = Variable(torch.Tensor(np.array(x)))#.cuda()).cuda()
        _, pred = model(Variable(x_))#.cuda()).cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            trainLowDim.append(input)
        else:
            trainLowDim.append(_.detach().cpu().numpy().tolist())

    singleTrainVec = {}
    singleTrainLabel = {}
    for index in stateIndex:
        singleTrainVec[index] = []
        singleTrainLabel[index] = []
    for eventIndex in range(0, len(train_event)):
        currStateIndex = stateValue.index(train_event[eventIndex])
        singleTrainVec[currStateIndex].append(trainLowDim[eventIndex])
        singleTrainLabel[currStateIndex].append(train_label[eventIndex])

    tensor_x = torch.from_numpy(np.array(trainLowDim).astype(np.float32))#.cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32))#.cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
    modelTotal = MLP(tensor_x.shape[1])
    criterionLinear = nn.L1Loss()#.cuda()
    optimizerLinear = optim.Adam(modelTotal.parameters(), lr=0.0001)
    for epoch in range(100):
        # print("total train epoch:" + str(epoch))
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            pred = modelTotal(Variable(x))#.cuda()).cuda())
            pred = pred.squeeze(-1)
            loss = criterionLinear(pred, y)
            optimizerLinear.zero_grad()
            loss.backward()
            optimizerLinear.step()
            total_loss += loss
    # transfer learning train
    for index in singleTrainVec.keys():
        if len(singleTrainVec[index]) < 10:
            continue
        tensor_x = torch.from_numpy(np.array(singleTrainVec[index]).astype(np.float32))#.cuda()
        tensor_y = torch.from_numpy(np.array(singleTrainLabel[index]).astype(np.float32))#.cuda()
        my_dataset = TensorDataset(tensor_x, tensor_y)
        my_dataset_loader = DataLoader(my_dataset, batch_size=8, shuffle=False)
        if index in modelDict.keys():
            modelLinear = modelDict[index]
        else:
            modelLinear = copy.deepcopy(modelTotal)
        criterionLinear = nn.L1Loss()#.cuda()
        optimizerLinear = optim.Adam(modelLinear.parameters(), lr=0.00001)
        for epoch in range(100):
            total_loss = 0
            for i, (x, y) in enumerate(my_dataset_loader):
                pred = modelLinear(Variable(x))#.cuda()).cuda())
                pred = pred.squeeze(-1)
                loss = criterionLinear(pred, y)
                optimizerLinear.zero_grad()
                loss.backward()
                optimizerLinear.step()
                total_loss += loss
            # print("total_loss = " + str(total_loss))
        modelDict[index] = modelLinear

    return modelDict, None, model

def test(AllData, Test, modelDict, model, method, data, isMul=0):#
    state_dict, edge_list, edgeCount_list, event_list = readFile(AllData, method)
    stateIndex = list(state_dict.keys())
    stateValue = []
    for state in state_dict.values():
        tem1 = []
        for key in state.keys():
            for num in range(0, state[key]):
                tem1.append(key)
        tem1.sort()
        stateValue.append(tem1)
    _, test_sel, _, test_vec, _, test_label, _, test_event = getTrainAndTestData(None, Test, event_list, data)

    testLowDim = []
    for x,y in zip(test_vec,test_sel):
        x_ = Variable(torch.Tensor(np.array(x)))#.cuda()).cuda()
        _, pred = model(Variable(x_))#.cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            testLowDim.append(input)
        else:
            testLowDim.append(_.detach().cpu().numpy().tolist())

    singleTestVec = {}
    singleTestLabel = {}
    for index in stateIndex:
        singleTestVec[index] = []
        singleTestLabel[index] = []
    for eventIndex in range(0, len(test_event)):
        currStateIndex = stateValue.index(test_event[eventIndex])
        singleTestVec[currStateIndex].append(testLowDim[eventIndex])
        singleTestLabel[currStateIndex].append(test_label[eventIndex])
    MAE, MSE, count = predict(singleTestVec, singleTestLabel, modelDict)
    return MAE, count

def predict(singleTestVec, singleTestLabel, modelDict):
    predList = []
    realList = []
    for index in singleTestVec.keys():
        if index in modelDict.keys():
            for vec in range(0, len(singleTestVec[index])):
                input = Variable(torch.Tensor(np.array(singleTestVec[index][vec])))#.cuda()).cuda()
                pred = modelDict[index](input)
                predList.append(pred)
                realList.append(singleTestLabel[index][vec])
    if len(realList) == 0:
        return 0, 0, 0
    MSE = computeMSE(realList, predList)
    MAE = computeMAE(realList, predList)
    return MAE, MSE, len(realList)

def train_multiset(file, ts_type, time_unit):
    # train
    singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict = train(file, ts_type, time_unit)
    # predict
    MAE, MSE = predict(file, singleTrainVec, singleTrainLabel, singleTestVec, singleTestLabel, modelDict, ts_type)
    return MAE, MSE

class autoencoder(nn.Module):
    def __init__(self, dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128),#.cuda(),
            nn.Tanh(),#.cuda(),
            nn.Linear(128, 32),#.cuda(),
        )#.cuda()
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),#.cuda(),
            nn.Tanh(),#.cuda(),
            nn.Linear(128, dim),#.cuda(),
            nn.Sigmoid()#.cuda()
        )#.cuda()

    def forward(self, x):
        encoder = self.encoder(x)#.cuda()
        decoder = self.decoder(encoder)#.cuda()
        return encoder, decoder

class MLP(nn.Module):
    def __init__(self, x):
        super(MLP, self).__init__()
        self.firstLayer = nn.Sequential(nn.Linear(x, x*2),#.cuda(),
                                        nn.ReLU(),#.cuda(),
                                        nn.Linear(x*2, x),#.cuda(),
                                        nn.ReLU(),#.cuda(),
                                        nn.Linear(x, 1),#.cuda(),
                                        )#.cuda()

    def forward(self, x):
        first = self.firstLayer(x)
        return first

def represent(AllData, Test, method, data, model, event_list):
    initTrans = []
    for i in range(0, len(event_list)):
        initTrans.append(0)
    test_vec = []
    test_sel = []
    test_label = []
    test_event = []
    if Test != None:
        for trace in Test:
            event = trace[-1]
            remainTime = float(event[-1])
            currEventIndex = trace.index(event)
            transVec = initTrans.copy()
            tempEvent = []
            for i in range(0, currEventIndex + 1):
                event_ = trace[i][0]
                currEventTempIndex = event_list.index(event_)
                transVec[currEventTempIndex] += 1
                tempEvent.append(event_)
            test_vec.append(transVec)
            test_sel.append([event[i] for i in data])
            test_label.append(remainTime)
            tempEvent.sort()
            test_event.append(tempEvent)

    testLowDim = []
    for x, y in zip(test_vec, test_sel):
        x_ = Variable(torch.Tensor(np.array(x)))  # .cuda()).cuda()
        _, pred = model(Variable(x_))  # .cuda())
        input = _.detach().cpu().numpy().tolist()
        input.extend(y)
        testLowDim.append(input)

    singleTestVec = []
    for eventIndex in range(0, len(test_event)):
        singleTestVec.append(testLowDim[eventIndex])
    singleTestVec = np.array(singleTestVec)
    return singleTestVec

def trainBucket(AllData, Train, Test, method, data, modelEncoder, modelTotal, isMul=0):
    state_dict, edge_list, edgeCount_list, event_list = readFile(AllData, method)
    train_sel, test_sel, train_vec, test_vec, train_label, test_label, train_event, test_event = \
        getTrainAndTestData(Train, Test, event_list, data)
    trainLowDim = []
    for x,y in zip(train_vec,train_sel):
        x_ = Variable(torch.Tensor(np.array(x)))#.cuda()).cuda()
        _, pred = modelEncoder(Variable(x_))#.cuda()).cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            trainLowDim.append(input)
        else:
            trainLowDim.append(_.detach().cpu().numpy().tolist())

    tensor_x = torch.from_numpy(np.array(trainLowDim).astype(np.float32))#.cuda()
    tensor_y = torch.from_numpy(np.array(train_label).astype(np.float32))#.cuda()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size=8, shuffle=False)
    modelLinear = copy.deepcopy(modelTotal)
    criterionLinear = nn.L1Loss()#.cuda()
    optimizerLinear = optim.Adam(modelLinear.parameters(), lr=0.00001)
    for epoch in range(50):#100
        total_loss = 0
        for i, (x, y) in enumerate(my_dataset_loader):
            pred = modelLinear(Variable(x))#.cuda()).cuda())
            pred = pred.squeeze(-1)
            loss = criterionLinear(pred, y)
            optimizerLinear.zero_grad()
            loss.backward()
            optimizerLinear.step()
            total_loss += loss
    MAE = testBucket(AllData, Test, modelEncoder, modelLinear, method, data, isMul=1)
    return modelLinear, MAE

def testBucket(AllData, Test, modelEncoder, modelDict, method, data, isMul=0):
    state_dict, edge_list, edgeCount_list, event_list = readFile(AllData, method)
    _, test_sel, _, test_vec, _, test_label, _, test_event = getTrainAndTestData(None, Test, event_list, data)

    testLowDim = []
    for x,y in zip(test_vec,test_sel):
        x_ = Variable(torch.Tensor(np.array(x)))#.cuda()).cuda()
        _, pred = modelEncoder(Variable(x_))#.cuda())
        if isMul == 1:
            input = _.detach().cpu().numpy().tolist()
            input.extend(y)
            testLowDim.append(input)
        else:
            testLowDim.append(_.detach().cpu().numpy().tolist())

    predList = []
    for vec in range(0, len(testLowDim)):
        input = Variable(torch.Tensor(np.array(testLowDim[vec])))#.cuda()).cuda()
        pred = modelDict(input)
        predList.append(float(pred))

    # input = Variable(torch.Tensor(np.array(testLowDim)))  # .cuda()).cuda()
    # pred = modelDict(input)
    MAE = computeMAE(test_label, predList)
    return MAE, len(predList)