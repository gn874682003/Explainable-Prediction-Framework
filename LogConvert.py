##原始日志属性处理，输入原始日志csv，输出属性标签csv作为特征选择的输入文件
#caseID、数值属性不动，字符属性转为序号，时间属性进行转换

from numpy import loadtxt
import copy
import time
from datetime import datetime
from scipy.io import savemat
import Frame.Repeat as FR

#对字符属性进行标号
def makeVocabulary(data, index, co=None):
    temp = list()
    for line in data:
        temp.append(line[index])#原去掉int()
    temp_temp = set(temp)#删除重复数据
    if 'null' in temp_temp:
        temp_temp.remove('null')
    if co == None:
        vocabulary = {str(sorted(list(temp_temp))[i]): i + 1 for i in range(len(temp_temp))}
        vocabulary['null'] = 0
    else:
        vocabulary = {co[i]: i for i in range(len(co))}
        for line in temp_temp:
            if line not in vocabulary.keys():
                vocabulary[line] = len(vocabulary)
    for i in range(len(data)):
        data[i][index] = vocabulary[data[i][index]]
    voc = {vocabulary[i]: i for i in vocabulary.keys()}
    return voc


#计算时间特征
def makeTime(data, index):
    dayS = 60 * 60 * 24
    ind = 0
    front = data[ind]
    # t2 = time.strptime(front[index-1], "%Y/%m/%d %H:%M:%S")
    t = time.strptime(front[index], "%Y/%m/%d %H:%M:%S")
    temp = 0. #(datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))).seconds / dayS#
    count = temp
    data[ind].append(temp) #当前事件的执行时间
    data[ind].append(count) #总执行时间
    maxR = temp
    maxA = count
    data[ind].append(t.tm_mon) #月/12
    data[ind].append(t.tm_mday) #日/30
    data[ind].append(t.tm_wday) #周/7
    data[ind].append(t.tm_hour) #时/24
    data[ind].append(t.tm_year-2000) #年-2000 增量
    for line in data[1:]:
        ind += 1
        t = time.strptime(line[index], "%Y/%m/%d %H:%M:%S")
        if line[0] == front[0]:
            t2 = time.strptime(front[index], "%Y/%m/%d %H:%M:%S")
            temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).total_seconds() / dayS
            count += temp
        else:
            # t2 = time.strptime(line[index - 1], "%Y/%m/%d %H:%M:%S")
            temp = 0.  # (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).seconds / dayS#
            count = temp
        data[ind].append(temp)
        data[ind].append(count)
        # if maxR < temp:
        #     maxR = temp
        # if maxA < count:
        #     maxA = count
        data[ind].append(t.tm_mon)
        data[ind].append(t.tm_mday)
        data[ind].append(t.tm_wday)
        data[ind].append(t.tm_hour)
        data[ind].append(t.tm_year - 2000)
        front = line
    # for j in range(len(data)):
    #     data[j][-5] = data[j][-5] / maxA
    #     data[j][-6] = data[j][-6] / maxR
    return maxA, maxR

#计算标签
def makeLable(data, ti, ei, maxA, maxR):
    ind = 0
    front = data[ind]
    for line in data[1:]:
        if line[0] == front[0]:
            data[ind].append(line[ei]) #下一事件
            data[ind].append(line[ti]) #下一事件持续时间*maxR
        else:
            data[ind].append(0)
            data[ind].append(0.)
        front = line
        ind += 1
    data[ind].append(0)
    data[ind].append(0.)
    data[ind].append(0.) #剩余时间
    count = 0.
    for line in reversed(data[0:-1]):
        ind -= 1
        if line[0] == front[0]:
            count += front[ti]#*maxR
        else:
            count = 0.
        front = line
        data[ind].append(count)  # 剩余时间

def LogC(eventlog, convertI, convertO=None):
    data, header = FR.readcsv('./Dataset/'+eventlog+'.csv')
    attribNum = len(header)
    header.remove(header[0])
    header.remove(header[0])
    header.remove(header[0])
    #数值转换
    convertReflact = list()
    if convertO==None:
        for i in range(3,len(data[0])):
            if i in convertI:
                 convertReflact.append(makeVocabulary(data, i))
            else:
                if data[0][i] == '':
                    data[0][i] = 0
                min = float(data[0][i])
                max = float(data[0][i])
                for j in range(len(data)):
                    if data[j][i] == '':
                        data[j][i] = 0
                    data[j][i] = float(data[j][i])
                    if min > data[j][i]:
                        min = data[j][i]
                    if max < data[j][i]:
                        max = data[j][i]
                for j in range(len(data)):
                    data[j][i] = data[j][i]/max
    else:
        for i, co in zip(convertI, convertO):
             convertReflact.append(makeVocabulary(data, i, co))
    #时间特征，总执行时间，当前事件的执行时间，月日周时
    maxA, maxR = makeTime(data, 2) #时间下标
    header.append('duration')
    header.append('allDuration')
    header.append('month')
    header.append('day')
    header.append('week')
    header.append('hour')
    header.append('year') # 增量更新
    #标签：下一事件，下一事件持续时间，剩余时间
    makeLable(data, attribNum, 3, maxA, maxR) #时间、事件下标
    header.append('nextEvent')
    header.append('nextDuration')
    header.append('remaining')
    # data2 = copy.deepcopy(data)
    # for line in data2:
    #     line.remove(line[0])
    #     line.remove(line[0])
    #     line.remove(line[0])
    return data,header,convertReflact, maxA, maxR

    #文件保存
    # f = open(eventlog+'F.csv', 'w', newline='', encoding='utf-8')
    # f_csv = csv.writer(f)
    # # f_csv.writerow(header)
    # f_csv.writerows(data)
    # f.close()
    #
    # mdict = {'header': header, 'vocab_3': vocab_3, 'vocab_5': vocab_5}#
    # savemat('data/'+eventlog+'F.mat', mdict)

    # f1 = open('hdf.fmap', "w")
    # for i, feat in enumerate(header):
    #     f1.write('{0}\t{1}\tq\n'.format(i, feat))
    # f1.close()