# 划分数据集，将原始数据按照时间顺序均分为5部分
# 每一部分选1/5作为测试集，剩下的数据作为训练集
import datetime
import math

dt = -1 # 0 以轨迹开始时间划分，-1 以轨迹结束时间划分

def sortByTime(elem):
    return elem[dt][1]

def sortByD(elem):
    return elem[dt][-7]

def sortByM(elem):
    return elem[dt][-8]

def sortByY(elem):
    return elem[dt][-4]

def DiviData(dataset, State):
    # 按轨迹分开
    orginal_trace = list()
    trace_temp = list()
    flag = dataset[0][0]
    for line in dataset:
        if flag == line[0]:
            trace_temp.append(line)
        else:
            orginal_trace.append(trace_temp)
            trace_temp = list()
            trace_temp.append(line)
        flag = line[0]
    # 轨迹按时间排序,去掉日期属性列
    orginal_trace.sort(key=sortByTime)
    for line1 in orginal_trace:
        for line2 in line1:
            line2.remove(line2[0])
            line2.remove(line2[0])
            line2.remove(line2[0])
    # 输入属性类别编号
    j = 0
    for line1 in orginal_trace:
        j += 1
        if j == len(State):
            break
        for i in range(1,len(line1)):
            if line1[0][j] != line1[i][j]:
                if State[j] == 1 or State[j] == 3:
                    State[j] += 1
                    break
    # 划分为5部分,每部分抽取1/5作为测试集，其他为训练集
    traceNum = len(orginal_trace)
    Train = []
    Test = []
    for i in range(5):
        secNum = int(traceNum / 25)
        for j in range(i * secNum * 5, i * secNum * 5 + secNum):
            Test.append(orginal_trace[j])
        for j in range(i * secNum * 5 + secNum, (i + 1) * secNum * 5):
            Train.append(orginal_trace[j])
    return Train,Test,orginal_trace

def DiviDataByTime(AllData, proportion, time):
    # 轨迹按时间排序,去掉日期属性列
    AllData.sort(key=sortByD)
    AllData.sort(key=sortByM)
    AllData.sort(key=sortByY)
    # 确定周期数
    if time == 'year':
        year = str(AllData[0][dt][-4])
        Datas = {year: []}
        for line in AllData:
            if year == str(line[dt][-4]):
                Datas[year].append(line)
            else:
                year = str(line[dt][-4])
                Datas[year] = []
                Datas[year].append(line)
    elif time == 'month':
        month = str(AllData[0][dt][-4]) + '/' + str(AllData[0][dt][-8])
        Datas = {month:[]}
        for line in AllData:
            if month == str(line[dt][-4]) + '/' + str(line[dt][-8]):
                Datas[month].append(line)
            else:
                month = str(line[dt][-4]) + '/' + str(line[dt][-8])
                Datas[month] = []
                Datas[month].append(line)
    elif time == 'week':
        week = str(AllData[0][dt][-4]) + '/' + \
               str(datetime.date(AllData[0][dt][-4]+2000, AllData[0][dt][-8], AllData[0][dt][-7]).isocalendar()[1])
        Datas = {week: []}
        for line in AllData:
            if week == str(line[dt][-4]) + '/' + \
                    str(datetime.date(line[dt][-4]+2000, line[dt][-8], line[dt][-7]).isocalendar()[1]):
                Datas[week].append(line)
            else:
                week = str(line[dt][-4]) + '/' + \
                    str(datetime.date(line[dt][-4]+2000, line[dt][-8], line[dt][-7]).isocalendar()[1])
                Datas[week] = []
                Datas[week].append(line)
    elif time == 'day':
        day = str(AllData[0][dt][-4]) + '/' + str(AllData[0][dt][-8]) + '/' + str(AllData[0][dt][-7])
        Datas = {day: []}
        for line in AllData:
            if day == str(line[dt][-4]) + '/' + str(line[dt][-8]) + '/' + str(line[dt][-7]):
                Datas[day].append(line)
            else:
                day = str(line[dt][-4]) + '/' + str(line[dt][-8]) + '/' + str(line[dt][-7])
                Datas[day] = []
                Datas[day].append(line)
    # 按比例划分训练集和测试集
    timeNum = len(Datas)
    trainNum = math.floor(timeNum*proportion)
    Train = []
    Test = []
    Tests = {}
    for line in Datas:
        if trainNum == 0:
            for line2 in Datas[line]:
                Test.append(line2)
            Tests[line] = Datas[line]
        else:
            for line2 in Datas[line]:
                Train.append(line2)
            trainNum -= 1
    return Train, Test, Tests
