# Original log attribute processing, input the original log CSV,
# normalize the numeric attribute, convert the character attribute to the index
# expand the timestamp

from numpy import loadtxt
import copy
import time
from datetime import datetime
from scipy.io import savemat
import ICPM2022.Repeat as FR

# Convert classification attribute value to index
def makeVocabulary(data, index, co=None):
    temp = list()
    for line in data:
        temp.append(line[index])
    temp_temp = set(temp)
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

# Extended time features
def makeTime(data, index):
    dayS = 60 * 60 * 24
    ind = 0
    front = data[ind]
    # t2 = time.strptime(front[index-1], "%Y/%m/%d %H:%M:%S")
    t = time.strptime(front[index], "%Y/%m/%d %H:%M:%S")
    temp = 0. #(datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))).seconds / dayS#
    count = temp
    data[ind].append(temp) # current event duration
    data[ind].append(count) # allDuration
    maxR = temp
    maxA = count
    data[ind].append(t.tm_mon) # month/12
    data[ind].append(t.tm_mday) # day/30
    data[ind].append(t.tm_wday) # week/7
    data[ind].append(t.tm_hour) # hour/24
    data[ind].append(t.tm_year - 2000)  # 年-2000 增量
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

# Calculate label
def makeLable(data, ti, ei, maxA, maxR):
    ind = 0
    front = data[ind]
    for line in data[1:]:
        if line[0] == front[0]:
            data[ind].append(line[ei]) # next event
            data[ind].append(line[ti]) # next event duration *maxR
        else:
            data[ind].append(0)
            data[ind].append(0.)
        front = line
        ind += 1
    data[ind].append(0)
    data[ind].append(0.)
    data[ind].append(0.) # remaining time
    count = 0.
    for line in reversed(data[0:-1]):
        ind -= 1
        if line[0] == front[0]:
            count += front[ti]#*maxR
        else:
            count = 0.
        front = line
        data[ind].append(count)  # remaining time

def LogC(eventlog, convertI, convertO=None):
    data, header = FR.readcsv('./dataset/'+eventlog+'.csv')
    attribNum = len(header)
    header.remove(header[0])
    header.remove(header[0])
    header.remove(header[0])
    # Numerical attribute normalization
    convertReflact = list()
    if convertO==None:
        for i in range(3,len(data[0])):
            if i in convertI:
                 convertReflact.append(makeVocabulary(data, i))
            else:
                min = float(data[0][i])
                max = float(data[0][i])
                for j in range(len(data)):
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
    # Timestampe: allDuration, duration, month, day, week, hour
    maxA, maxR = makeTime(data, 2) #Timestampe's index
    header.append('duration')
    header.append('allDuration')
    header.append('month')
    header.append('day')
    header.append('week')
    header.append('hour')
    header.append('year')  # 增量更新
    # label：next event，next event duration，remaining time
    makeLable(data, attribNum, 3, maxA, maxR)
    header.append('nextEvent')
    header.append('nextDuration')
    header.append('remaining')
    return data,header,convertReflact, maxA, maxR