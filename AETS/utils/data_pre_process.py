import pandas as pd
from datetime import datetime
"""
不同数据集的数据预处理
"""

def train_helpdesk(file, time_unit):
    if time_unit == 'second':
        time_unit = 1
    elif time_unit == 'minute':
        time_unit = 60
    elif time_unit == 'hour':
        time_unit = 60 * 60
    elif time_unit == 'day':
        time_unit = 24 * 60 * 60
    elif time_unit == 'month':
        time_unit = 30 * 24 * 60 * 60
    # 读取整个csv文件
    csv_data = pd.read_csv(file)
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    trace_log = fp.readlines()
    trace_temp = []
    data_list = []
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    traceTransList = []
    timeTransList = []
    for trace in data_list:
        traceTrans = []
        timeTrans = []
        endTime = trace[-1].split(",")[2].replace('\n', '')
        for event in trace:
            traceId, eventId, time = event.split(",")[0], event.split(",")[1], event.split(",")[2]
            time = time.replace('\n', '')
            target_time = abs((datetime.strptime(str(endTime),
                                                 '%Y/%m/%d %H:%M:%S') - datetime.strptime(str(time),
                                                                                          '%Y/%m/%d %H:%M:%S')).total_seconds() / time_unit)
            traceTrans.append(eventId)
            timeTrans.append(target_time)
        traceTransList.append(traceTrans)
        timeTransList = timeTransList + timeTrans
    csv_data = pd.read_csv(file)
    # print(len(timeTransList))
    remainTime = pd.DataFrame({'remainTime': timeTransList})
    # print(remainTime.shape)
    # print(csv_data.shape)
    print(csv_data)
    act = pd.get_dummies(csv_data.ActivityID)
    csv_data = csv_data.join(act).join(remainTime['remainTime'])
    del csv_data['Complete Time']
    csv_data.to_csv('../data/helpdesk.csv', index=False)
    print(data_list)


def train_SepsisCases(file, time_unit):
    if time_unit == 'second':
        time_unit = 1
    elif time_unit == 'minute':
        time_unit = 60
    elif time_unit == 'hour':
        time_unit = 60 * 60
    elif time_unit == 'day':
        time_unit = 24 * 60 * 60
    elif time_unit == 'month':
        time_unit = 30 * 24 * 60 * 60
    # 读取整个csv文件
    csv_data = pd.read_csv(file)
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    trace_log = fp.readlines()
    trace_temp = []
    data_list = []
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    traceTransList = []
    timeTransList = []
    for trace in data_list:
        traceTrans = []
        timeTrans = []
        endTime = trace[-1].split(",")[2].replace('\n', '')
        for event in trace:
            traceId, eventId, time = event.split(",")[0], event.split(",")[1], event.split(",")[2]
            time = time.replace('\n', '')
            target_time = abs((datetime.strptime(str(endTime) + ":00",
                                                 '%Y/%m/%d %H:%M:%S') - datetime.strptime(str(time) + ":00",
                                                                                          '%Y/%m/%d %H:%M:%S')).total_seconds() / time_unit)
            traceTrans.append(eventId)
            timeTrans.append(target_time)
        traceTransList.append(traceTrans)
        timeTransList = timeTransList + timeTrans
    csv_data = pd.read_csv(file)
    # print(len(timeTransList))
    remainTime = pd.DataFrame({'remainTime': timeTransList})
    # print(remainTime.shape)
    # print(csv_data.shape)
    csv_data = csv_data.join(remainTime['remainTime'])
    # print(csv_data)
    del csv_data['startTime']
    del csv_data['completeTime']
    act = pd.get_dummies(csv_data.event)
    InfectionSuspected = pd.get_dummies(csv_data.InfectionSuspected)
    DiagnosticUrinarySediment = pd.get_dummies(csv_data.DiagnosticUrinarySediment, columns=None)
    SIRSCritTemperature = pd.get_dummies(csv_data.SIRSCritTemperature)
    DiagnosticLiquor = pd.get_dummies(csv_data.DiagnosticLiquor)
    DiagnosticIC = pd.get_dummies(csv_data.DiagnosticIC)
    DiagnosticLacticAcid = pd.get_dummies(csv_data.DiagnosticLacticAcid)
    DiagnosticBlood = pd.get_dummies(csv_data.DiagnosticBlood)
    SIRSCritHeartRate = pd.get_dummies(csv_data.SIRSCritHeartRate)
    DiagnosticArtAstrup = pd.get_dummies(csv_data.DiagnosticArtAstrup)
    DiagnosticOther = pd.get_dummies(csv_data.DiagnosticOther)
    DiagnosticXthorax = pd.get_dummies(csv_data.DiagnosticXthorax)
    Oligurie = pd.get_dummies(csv_data.Oligurie)
    Hypotensie = pd.get_dummies(csv_data.Hypotensie)
    SIRSCriteria2OrMore = pd.get_dummies(csv_data.SIRSCriteria2OrMore)
    DiagnosticUrinaryCulture = pd.get_dummies(csv_data.DiagnosticUrinaryCulture)
    Infusion = pd.get_dummies(csv_data.Infusion)
    DiagnosticSputum = pd.get_dummies(csv_data.DiagnosticSputum)
    DiagnosticECG = pd.get_dummies(csv_data.DiagnosticECG)
    DisfuncOrg = pd.get_dummies(csv_data.DisfuncOrg)
    Diagnose = pd.get_dummies(csv_data.Diagnose)
    Hypoxie = pd.get_dummies(csv_data.Hypoxie)
    SIRSCritLeucos = pd.get_dummies(csv_data.SIRSCritLeucos)
    SIRSCritTachypnea = pd.get_dummies(csv_data.SIRSCritTachypnea)
    # print(DiagnosticUrinarySediment[:10])
    # act.to_csv("1111.csv")
    del csv_data['InfectionSuspected']
    del csv_data['DiagnosticUrinarySediment']
    del csv_data['SIRSCritTemperature']
    del csv_data['DiagnosticLiquor']
    del csv_data['DiagnosticIC']
    del csv_data['DiagnosticLacticAcid']
    del csv_data['DiagnosticBlood']
    del csv_data['SIRSCritHeartRate']
    del csv_data['DiagnosticArtAstrup']
    del csv_data['DiagnosticOther']
    del csv_data['DiagnosticXthorax']
    del csv_data['Oligurie']
    del csv_data['Hypotensie']
    del csv_data['SIRSCriteria2OrMore']
    del csv_data['DiagnosticUrinaryCulture']
    del csv_data['Infusion']
    del csv_data['DiagnosticSputum']
    del csv_data['DiagnosticECG']
    del csv_data['DisfuncOrg']
    del csv_data['Diagnose']
    del csv_data['Hypoxie']
    del csv_data['SIRSCritLeucos']
    del csv_data['SIRSCritTachypnea']
    csv_data = pd.concat(
        [csv_data, act, InfectionSuspected, DiagnosticUrinarySediment, SIRSCritTemperature, DiagnosticLiquor,
         DiagnosticIC, DiagnosticLacticAcid
            , DiagnosticBlood, SIRSCritHeartRate, DiagnosticArtAstrup, DiagnosticOther, DiagnosticXthorax, Oligurie,
         Hypotensie, SIRSCriteria2OrMore,
         DiagnosticUrinaryCulture, Infusion, DiagnosticSputum, DiagnosticECG, SIRSCritTachypnea,
         remainTime['remainTime']], axis=1)
    print(csv_data)
    csv_data.to_csv('./vecData/SepsisCases.csv', index=False)
    print(data_list)


def train_HospitalBilling(file, time_unit):
    if time_unit == 'second':
        time_unit = 1
    elif time_unit == 'minute':
        time_unit = 60
    elif time_unit == 'hour':
        time_unit = 60 * 60
    elif time_unit == 'day':
        time_unit = 24 * 60 * 60
    elif time_unit == 'month':
        time_unit = 30 * 24 * 60 * 60
    # 读取整个csv文件
    csv_data = pd.read_csv(file)
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    trace_log = fp.readlines()
    trace_temp = []
    data_list = []
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    traceTransList = []
    timeTransList = []
    for trace in data_list:
        traceTrans = []
        timeTrans = []
        endTime = trace[-1].split(",")[2].replace('\n', '')
        for event in trace:
            traceId, eventId, time = event.split(",")[0], event.split(",")[1], event.split(",")[2]
            time = time.replace('\n', '')
            target_time = abs((datetime.strptime(str(endTime) + ":00",
                                                 '%Y/%m/%d %H:%M:%S') - datetime.strptime(str(time) + ":00",
                                                                                          '%Y/%m/%d %H:%M:%S')).total_seconds() / time_unit)
            traceTrans.append(eventId)
            timeTrans.append(target_time)
        traceTransList.append(traceTrans)
        timeTransList = timeTransList + timeTrans
    csv_data = pd.read_csv(file)
    # print(len(timeTransList))
    remainTime = pd.DataFrame({'remainTime': timeTransList})
    del csv_data['startTime']
    del csv_data['completeTime']
    act = pd.get_dummies(csv_data.event)
    speciality = pd.get_dummies(csv_data.speciality)
    isCancelled = pd.get_dummies(csv_data.isCancelled)
    state = pd.get_dummies(csv_data.state)
    diagnosis = pd.get_dummies(csv_data.diagnosis)
    actRed = pd.get_dummies(csv_data.actRed)
    actOrange = pd.get_dummies(csv_data.actOrange)
    version = pd.get_dummies(csv_data.version)
    flagD = pd.get_dummies(csv_data.flagD)
    closeCode = pd.get_dummies(csv_data.closeCode)
    msgCode = pd.get_dummies(csv_data.msgCode)
    msgType = pd.get_dummies(csv_data.msgType)
    blocked = pd.get_dummies(csv_data.blocked)
    msgCount = pd.get_dummies(csv_data.msgCount)
    flagA = pd.get_dummies(csv_data.flagA)
    isClosed = pd.get_dummies(csv_data.isClosed)
    flagB = pd.get_dummies(csv_data.flagB)
    flagC = pd.get_dummies(csv_data.flagC)
    del csv_data['speciality']
    del csv_data['isCancelled']
    del csv_data['state']
    del csv_data['diagnosis']
    del csv_data['actRed']
    del csv_data['actOrange']
    del csv_data['version']
    del csv_data['flagD']
    del csv_data['closeCode']
    del csv_data['msgCode']
    del csv_data['msgType']
    del csv_data['blocked']
    del csv_data['msgCount']
    del csv_data['flagA']
    del csv_data['isClosed']
    del csv_data['flagB']
    del csv_data['flagC']
    csv_data = pd.concat(
        [csv_data, act, speciality, isCancelled, state, diagnosis,
         actRed, actOrange
            , version, flagD, closeCode, msgCode, msgType, blocked,
         msgCount, flagA,
         isClosed, flagB, flagC,
         remainTime['remainTime']], axis=1)
    print(csv_data)
    csv_data.to_csv('./vecData/HospitalBilling.csv', index=False)
    print(data_list)


def train_traffic_fines(file, time_unit):
    if time_unit == 'second':
        time_unit = 1
    elif time_unit == 'minute':
        time_unit = 60
    elif time_unit == 'hour':
        time_unit = 60 * 60
    elif time_unit == 'day':
        time_unit = 24 * 60 * 60
    elif time_unit == 'month':
        time_unit = 30 * 24 * 60 * 60
    # 读取整个csv文件
    csv_data = pd.read_csv(file)
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    trace_log = fp.readlines()
    trace_temp = []
    data_list = []
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    traceTransList = []
    timeTransList = []
    for trace in data_list:
        traceTrans = []
        timeTrans = []
        endTime = trace[-1].split(",")[2].replace('\n', '')
        for event in trace:
            traceId, eventId, time = event.split(",")[0], event.split(",")[1], event.split(",")[2]
            time = time.replace('\n', '')
            target_time = abs((datetime.strptime(str(endTime) + ":00",
                                                 '%Y/%m/%d %H:%M:%S') - datetime.strptime(str(time) + ":00",
                                                                                          '%Y/%m/%d %H:%M:%S')).total_seconds() / time_unit)
            traceTrans.append(eventId)
            timeTrans.append(target_time)
        traceTransList.append(traceTrans)
        timeTransList = timeTransList + timeTrans
    csv_data = pd.read_csv(file)
    # print(len(timeTransList))
    remainTime = pd.DataFrame({'remainTime': timeTransList})
    del csv_data['startTime']
    del csv_data['completeTime']
    act = pd.get_dummies(csv_data.event)
    notificationType = pd.get_dummies(csv_data.notificationType)
    vehicleClass = pd.get_dummies(csv_data.vehicleClass)
    lastSent = pd.get_dummies(csv_data.lastSent)
    dismissal = pd.get_dummies(csv_data.dismissal)
    del csv_data['notificationType']
    del csv_data['vehicleClass']
    del csv_data['lastSent']
    del csv_data['dismissal']
    csv_data = pd.concat([csv_data, act, notificationType, vehicleClass, lastSent, dismissal, remainTime['remainTime']],
                         axis=1)
    print(csv_data)
    csv_data.to_csv('./vecData/traffic_fines.csv', index=False)
    print(data_list)

def train(file, time_unit):
    if time_unit == 'second':
        time_unit = 1
    elif time_unit == 'minute':
        time_unit = 60
    elif time_unit == 'hour':
        time_unit = 60 * 60
    elif time_unit == 'day':
        time_unit = 24 * 60 * 60
    elif time_unit == 'month':
        time_unit = 30 * 24 * 60 * 60
    # 读取整个csv文件
    csv_data = pd.read_csv(file)
    fp = open(file, "r", encoding='utf-8')
    next(fp)
    trace_log = fp.readlines()
    trace_temp = []
    data_list = []
    current_traceId = trace_log[0].split(",")[0]
    for line in trace_log:
        traceId = line.split(",")[0]
        if current_traceId == traceId:
            trace_temp.append(line)
        else:
            current_traceId = traceId
            data_list.append(trace_temp)
            trace_temp = []
            trace_temp.append(line)
    data_list.append(trace_temp)
    traceTransList = []
    timeTransList = []
    for trace in data_list:
        traceTrans = []
        timeTrans = []
        endTime = trace[-1].split(",")[2].replace('\n', '')
        for event in trace:
            traceId, eventId, time = event.split(",")[0], event.split(",")[1], event.split(",")[2]
            time = time.replace('\n', '')
            target_time = abs((datetime.strptime(str(endTime),
                                                 '%Y/%m/%d %H:%M:%S') - datetime.strptime(str(time),
                                                                                          '%Y/%m/%d %H:%M:%S')).total_seconds() / time_unit)
            traceTrans.append(eventId)
            timeTrans.append(target_time)
        traceTransList.append(traceTrans)
        timeTransList = timeTransList + timeTrans
    csv_data = pd.read_csv(file)
    # print(len(timeTransList))
    remainTime = pd.DataFrame({'remainTime': timeTransList})
    # print(remainTime.shape)
    # print(csv_data.shape)
    print(csv_data)
    act = pd.get_dummies(csv_data.ActivityID)
    csv_data = csv_data.join(act).join(remainTime['remainTime'])
    del csv_data['Complete Time']
    csv_data.to_csv(file, index=False)
    print(data_list)

if __name__ == '__main__':
    train('../data/CCF.csv', 'day')
    # path = "../data/"
    # fileList = ['helpdesk.csv']#,'BPI2012.csv','BPI2015_1.csv','BPI2015_4.csv','BPI2015_5.csv','traffic_fines.csv']
    # # fileList = ['traffic_fines.csv']
    # for i in fileList:
    #     file = path + i
    #     if i == 'helpdesk.csv':
    #         train_helpdesk(file, 'day')
    #     if i == 'SepsisCases.csv':
    #         train_SepsisCases(file, 'day')
    #     if i == 'HospitalBilling.csv':
    #         train_HospitalBilling(file, 'day')
    #     if i == 'traffic_fines.csv':
    #         train_traffic_fines(file, 'day')
