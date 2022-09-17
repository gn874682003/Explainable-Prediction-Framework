import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import lightgbm as lgb
import time
import EHP.tree as tp

def plotFeature(X,name):
    aki = sorted(range(len(X)), key=lambda k: X[k])
    X.sort()
    name = [name[aki[i]] for i in range(len(aki))]
    # 图像绘制
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), X, color='k')
    # 添加数据标签
    for rect in b:
        w = rect.get_width()
        if w < 0:
            ax.text(0, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
        else:
            ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
    # 设置Y轴刻度线标签
    ax.set_yticks(range(len(name)))
    ax.set_yticklabels(name)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('重要性值')#Importance value
    plt.ylabel('特征')#Feature Name
    plt.show()

# 两阶段选取
def FinalFLightboost(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    # ai = [0]
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:, 0:1], y_train[:, 2])
    y_pre = modelR.predict(X_test[:, 0:1])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('Activity', MAE)

    modelR.fit(X_train[:, ai], y_train[:, 2])
    y_pre = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('All', MAE)
    # X3 = modelR.feature_importances_
    # plotFeature(X3, ak)

    # 剩余时间特征选取
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 10
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:# + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    #重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    print('剩余时间：', temp3[-1])
    ai = []
    ai.append(0)
    tempFR = []
    iFR = 0
    for i in range(1, len(temp3[-1][2])+1):#min(len(temp3[-1][2]), 6)):#
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        FR = [MAE, [ak[j] for j in ai], ai.copy()]
        tempFR.append(FR)
        if i != 1:
            if MAE < tempFR[iFR][0]:
                iFR = i-1
            else:
                ai.pop(-1)
        if i == len(temp3[-1][2]):
            break
        else:
            ai.append(d_value[i - 1][0])
    timeE = time.time()
    print('特征选取时间：',timeE-timeS)
    print('剩余时间：', tempFR[iFR])
    return tempFR[iFR]

#前缀选取
def PrefixLightGBM(Train, header, DRstate, FE, FD, FR, PF=3):
    state = [j for i, j in zip(DRstate, range(len(DRstate))) if i == 2 or i == 4]
    staId = [j for i, j in zip(DRstate, range(len(DRstate))) if i < 3]
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float1.append(IP)
    dataTra = np.array(list_to_float1)
    y_train = dataTra[:, (len(header)-3)*PF:(len(header)-3)*PF + 3]
    # 剩余时间
    ak = [header[i] + str(j) for j in range(PF) for i in FR[2]]
    ai = [i + (len(header) - 3) * j for j in range(PF) for i in FR[2]]
    aid = [i + (len(header) - 3) * j for j in range(PF) for i in staId]
    modelR = lgb.LGBMRegressor()  #max_depth=5
    sk = []
    si = []
    for i in FR[2][1:]:  #
        if i not in state:
            sk.extend([header[i] + str(j) for j in range(PF - 1)])
            si.extend([i + (len(header) - 3) * j for j in range(PF - 1)])
    for j, k in zip(si, sk):
        ak.remove(k)
        ai.remove(j)
    modelR.fit(dataTra[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if DRstate[i] < 3])
    y_pre = modelR.predict(dataTra[:, ai])
    MAE = mean_absolute_error(y_train[:, 2], y_pre)
    print(MAE)
    X3 = modelR.feature_importances_
    plotFeature(X3, ak)
    # 迭代策略
    lkk = []
    lii = []
    for i in FR[2]:#[1:]
        aii = ai.copy()
        akk = ak.copy()
        lk = [header[i] + str(j) for j in range(PF - 1)]
        li = [i + (len(header) - 3) * j for j in range(PF - 1)]
        if i in state:
            for j, k in zip(li, lk):
                akk.remove(k)
                aii.remove(j)
        else:
            continue
        modelR.fit(dataTra[:, aii], y_train[:, 2], feature_name=[str(aii[i]) for i in range(len(aii))],
               categorical_feature=[str(aii[i]) for i in range(len(aii)) if aii[i] in aid])
        y_pre = modelR.predict(dataTra[:, aii])
        MAEi = mean_absolute_error(y_train[:, 2], y_pre)
        print(i, ':', MAEi)
        if MAEi <= MAE:
            lkk.extend(lk)
            lii.extend(li)
    for j, k in zip(lii, lkk):
        ak.remove(k)
        ai.remove(j)
    modelR.fit(dataTra[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if ai[i] in aid])
    y_pre = modelR.predict(dataTra[:, ai])
    MAE = mean_absolute_error(y_train[:, 2], y_pre)
    print(MAE,ak)
    X = modelR.feature_importances_
    plotFeature(X, ak)
    return

#前向后向特征选择策略
def LightGBM(Train, Test, header, catId):
    cid = catId.copy()
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    global ai, ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = X_train
    y_test = y_train
    X_testE = dataTes[:, ai]
    y_testE = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    modelR = lgb.LGBMRegressor()  # max_depth=5
    modelR.fit(X_train[:, 0:1], y_train[:, 2], feature_name='0', categorical_feature='0')
    y_pre = modelR.predict(X_test[:, 0:1])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('Activity', MAE)

    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    y_pre = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('All',MAE)

    # 后向删除消极特征
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                if ti in catId:
                    cid.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
                    categorical_feature=[str(cid[i]) for i in range(len(cid))])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                # d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
        if ti in catId:
            cid.remove(ti)
    timeE = time.time()
    print('Step1特征选取时间：', timeE - timeS, len(ai))
    print('MAE：', temp3[-1])
    # modelR.fit(X_trainE[:, ai], y_trainE[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
    #            categorical_feature=[str(cid[i]) for i in range(len(cid))])
    # y_pre = modelR.predict(X_testE[:, temp3[-1][2]])
    # TMAE = mean_absolute_error(y_testE[:, 2], y_pre)
    # print('测试集', TMAE)
    # 重要性值画图
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    ai.sort()
    print(ai)
    ai, aiMAE = showLocalTree(Train, Train, header, ai, cid)
    print(ai,aiMAE)
    #自动选取
    aii = len(ai) - 1
    for i in range(len(ai) - 1):
        if aiMAE[i] - aiMAE[i + 1] > 0:#.1
            aii = i + 1
    FR = [aiMAE[aii], [header[i] for i in ai[0:aii]], ai[0:aii]]
    print(FR)
    ncid = []
    for i in FR[2]:
        if i in cid:
            ncid.append(i)
    modelR.fit(X_train[:, FR[2]], y_train[:, 2], feature_name=[str(FR[2][i]) for i in range(len(FR[2]))],
               categorical_feature=[str(ncid[i]) for i in range(len(ncid))])
    y_pre = modelR.predict(X_testE[:, FR[2]])
    TMAE = mean_absolute_error(y_testE[:, 2], y_pre)
    print('测试集', TMAE, FR[2])
    return FR

#特征全遍历树
def FTTree(Train, Test, header, catId):
    cid = catId.copy()
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    global ai,ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    modelR = lgb.LGBMRegressor()

    # 剩余时间特征选取全遍历
    aai = []
    aai.append(ai[0])
    cid = []
    cid.append(0)
    global TR,minIn
    TR = []
    minIn = []
    global tree
    tree = tp.Tree('0')
    timeS = time.time()
    fnAllTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)
    # fnTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)#加入特征类别
    # fnTree2(modelR, aai, 1, X_train, y_train, X_test, y_test)
    minVI = np.argmin(np.array(TR)[:, 0])
    timeE = time.time()
    print('特征选取时间：', timeE-timeS)
    print('剩余时间：', TR[minVI])
    # 画局部树图
    minV = min(np.array(TR)[:, 0])
    maxV = max(np.array(TR)[:, 0])
    tree.show(20,minV,(maxV-minV)/19)
    # if len(minIn)>1:
    #     myTree = {'0'+str(minIn[0][1]):plotLocalTree(1)}
    #     tp.createPlot(myTree)
    return TR[minVI]

def fnAllTree(modelR,aai,n,X_train,y_train,X_test,y_test,catId,cid):
    modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    else:
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2], TR)).index(line)][0]
                q = tp.Node(data=str(i), tag=str(line), value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
                tree.insert(p, q)
            p = q
    if aai[-1] == ai[-1]:
        return tree
    else:
        for i in range(n, len(ai)):
            if aai[-1] >= ai[i]:
                continue
            if ai[i] in aai:
                break
            aai.append(ai[i])
            if ai[i] in catId:
                cid.append(ai[i])
            fnAllTree(modelR, aai, n+1, X_train, y_train, X_test, y_test,catId,cid)
            if aai[-1] in catId:
                cid.pop(-1)
            aai.pop(-1)

#特征选取参考树
def showLocalTree(Train, Test, header, ai, cid):
    attribNum = len(header) - 3
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    X_train = dataTra[:, 0:attribNum]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, 0:attribNum]
    y_test = dataTes[:, attribNum:attribNum + 3]

    modelR = lgb.LGBMRegressor()
    timeS = time.time()
    aai = []
    aaiMAE = []
    cci = []
    aai.append(ai[0])
    cci.append(ai[0])
    ai.remove(ai[0])
    tree = tp.Tree('0')
    modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
                    categorical_feature=[str(cci[i]) for i in range(len(cci))])
    y_pre = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    aaiMAE.append(MAE)
    tree.root.data = '0'
    tree.root.tag = '0'
    tree.root.value = '0 : ' + str(round(MAE, 3))
    p = tree.root
    minMAE = MAE
    maxMAE = MAE
    while len(ai) != 0:
        for line in ai:
            aai.append(line)
            if line in cid:
                cci.append(line)
            modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
                    categorical_feature=[str(cci[i]) for i in range(len(cci))])
            y_pre = modelR.predict(X_test[:, aai])
            MAE = mean_absolute_error(y_test[:, 2], y_pre)
            q = tp.Node(data=str(line), tag=str(aai), value=str(line) + ' : ' + str('%.3f'%MAE))
            tree.insert(p, q)
            aai.remove(line)
            if line in cci:
                cci.remove(line)
            if line == ai[0] or MAE < MAEO:
                MAEO = MAE
                t = q
                linet = line
            if MAE < minMAE:
                minMAE = MAE
            elif MAE > maxMAE:
                maxMAE = MAE
        p = t
        aai.append(linet)
        aaiMAE.append(MAEO)
        if linet in cid:
            cci.append(linet)
        ai.remove(linet)
    timeE = time.time()
    print('特征选取时间：', timeE - timeS)
    # 画局部树图
    # tree.show(20, round(minMAE, 3), (maxMAE - minMAE) / 19)
    return aai, aaiMAE