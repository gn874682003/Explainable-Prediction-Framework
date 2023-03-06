import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import sklearn.feature_selection as feaSel
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from minepy import MINE
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import lightgbm as lgb
# import shap
import time
import ICWS2023.tree as tp

def plotFeature(X,name):
    aki = sorted(range(len(X)), key=lambda k: X[k])
    X.sort()
    name = [name[aki[i]] for i in range(len(aki))]
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), X, color='k')
    for rect in b:
        w = rect.get_width()
        if w < 0:
            ax.text(0, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
        else:
            ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
    ax.set_yticks(range(len(name)))
    ax.set_yticklabels(name)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('Importance value')
    plt.ylabel('#Feature Name')
    plt.show()

# feature selection strategy
def LightGBM(Train, Test, Test1, header, catId):
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
    y_train = dataTra[:, attribNum:attribNum + 3]#-1
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    list_to_float2 = []
    for line in Test1:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_test1 = dataTes[:, ai]
    y_test1 = dataTes[:, attribNum:attribNum + 3]
    modelR = lgb.LGBMRegressor()  # max_depth=5
    modelR.fit(X_train[:, 0:1], y_train[:, 2], feature_name='0', categorical_feature='0')
    y_pre = modelR.predict(X_test1[:, 0:1])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print('Activity', MAE)
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pre = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print('All', MAE)

    # Priority-based Feature Selection
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 30
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # Training model, calculation MAE
        modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # Judge whether the accuracy rate drops. If so, change the priority
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                # d_value.pop(ti)
        # Delete the attribute with the lowest importance value among the attributes with the lowest priority
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
    timeE = time.time()
    print('Val:Step1', temp3[-1],',1.1 execution time：', timeE - timeS, len(ai))
    # 重要性值画图
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pred)
    print("Test:Step1", MAE)
    timeS = time.time()
    ai.sort()
    # Incremental Feature Tree Construction
    ai, aiMAE = showLocalTree(Train, Test, header, ai, catId, 2)
    aii = aiMAE.index(min(aiMAE))
    for i in range(aii):
        if aiMAE[i] - aiMAE[aii] < 0.2:  # .1
            aii = i
            break
    FR = [aiMAE[aii], [header[i] for i in ai[0:aii+1]], ai[0:aii+1]]
    timeE = time.time()
    print('Val:Step2', FR,',1.2 execution time：', timeE - timeS, len(ai))
    ai = FR[2]
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pred)
    print("Test:Step2",MAE)
    return FR

# Depth-first Traverse
def AllFTree(Train, Test, header, catId):
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
    # Record local tree
    fnTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)
    # Record global tree
    # fnAllTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)
    minVI = np.argmin(np.array(TR)[:, 0])
    timeE = time.time()
    print('Depth-first Traverse execution time：', timeE-timeS)
    print('MAE：', TR[minVI])
    # local tree
    minV = min(np.array(TR)[:, 0])
    maxV = max(np.array(TR)[:, 0])
    tree.show(20,minV,(maxV-minV)/19)
    #  full tree
    # if len(minIn)>1:
    #     myTree = {'0'+str(minIn[0][1]):plotLocalTree(1)}
    #     tp.createPlot(myTree)
    return TR[minVI]

def fnTree(modelR,aai,n,X_train,y_train,X_test,y_test,catId,cid):
    modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    # print(MAE, aai)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:# or MAE < minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    elif MAE <= minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2] ,TR)).index(line)][0]
                q = tp.Node(data=str(i),tag=str(line),value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
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
            fnTree(modelR, aai, n+1, X_train, y_train, X_test, y_test,catId,cid)
            if aai[-1] in catId:
                cid.pop(-1)
            aai.pop(-1)

def fnTree2(modelR,aai,n,X_train,y_train,X_test,y_test):
    modelR.fit(X_train[:, aai], y_train[:, 2])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    # print(MAE, aai)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:# or MAE < minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    elif MAE <= minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2] ,TR)).index(line)][0]
                q = tp.Node(data=str(i),tag=str(line),value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
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
            fnTree2(modelR, aai, n+1, X_train, y_train, X_test, y_test)
            aai.pop(-1)

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

# Incremental Feature Tree Construction
def showLocalTree(Train, Test, header, ai, cid, task):
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

    if task == 0:
        modelR = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    else:
        modelR = lgb.LGBMRegressor()
    aai = []
    aaiMAE = []
    cci = []
    aai.append(ai[0])
    cci.append(ai[0])
    ai.remove(ai[0])
    tree = tp.Tree('0')
    modelR.fit(X_train[:, aai], y_train[:, task], feature_name=[str(i) for i in aai],
               categorical_feature=[str(i) for i in aai if i in cid])
    y_pre = modelR.predict(X_test[:, aai])
    if task == 0:
        predictions = [round(value) for value in y_pre]
        MAE = accuracy_score(y_test[:, task], predictions)
    else:
        MAE = mean_absolute_error(y_test[:, task], y_pre)
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
            modelR.fit(X_train[:, aai], y_train[:, task], feature_name=[str(i) for i in aai],
               categorical_feature=[str(i) for i in aai if i in cid])
            y_pre = modelR.predict(X_test[:, aai])
            if task == 0:
                predictions = [round(value) for value in y_pre]
                MAE = accuracy_score(y_test[:, task], predictions)
            else:
                MAE = mean_absolute_error(y_test[:, task], y_pre)
            q = tp.Node(data=str(line), tag=str(aai), value=str(line) + ' : ' + str('%.3f'%MAE))
            tree.insert(p, q)
            aai.remove(line)
            if line in cci:
                cci.remove(line)
            if line == ai[0] or (MAE < MAEO and task !=0) or (MAE > MAEO and task ==0):
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
    # show incramental tree
    tree.show(20, round(minMAE, 3), (maxMAE - minMAE) / 19)
    return aai, aaiMAE