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
import Frame.tree as tp

# 返回模型在Train训练后，Test的测试结果
def TestK(Train, Test, header, catId, aii):
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
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]
    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:, aii], y_train[:, 2], feature_name=[str(i) for i in aii],
               categorical_feature=[str(i) for i in aii if i in catId])
    y_pred = modelR.predict(X_test[:, aii])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    FR = [MAE, [header[i] for i in aii], aii]
    print(FR)
    return FR

def TestLGBM(Train, Test, dataFE, dataFD, dataFR, PF=5):
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            list_to_float1.append(line1)
    for line in Test:
        for line1 in line:
            list_to_float2.append(line1)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = dataFE['index'][-1].tolist()
    # print([str(i) for i in ai])
    # print([str(ai[i]) for i in range(len(ai)) if dataFE['state'][-1][i] < 3])
    modelE = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    modelE.fit(dataTra[:, ai], dataTra[:, -3], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFE['state'][-1][i] < 3])
    y_pred = modelE.predict(dataTes[:, ai])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(dataTes[:, -3], predictions)
    print('下一事件：', accuracy)
    # 持续时间
    ai = dataFD['index'][-1].tolist()
    modelD = lgb.LGBMRegressor()
    modelD.fit(dataTra[:, ai], dataTra[:, -2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFD['state'][-1][i] < 3])
    y_pre = modelD.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -2], y_pre)
    print('持续时间：', MAE)
    # 剩余时间
    ai = dataFR['index'][-1].tolist()
    modelR = lgb.LGBMRegressor()
    modelR.fit(dataTra[:, ai], dataTra[:, -1], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFR['state'][-1][i] < 3])
    y_pre = modelR.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -1], y_pre)
    print('剩余时间：', MAE)

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
    for line in Test:
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
            list_to_float2.append(IP)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFE['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF-1) for i in dataFE['index'][-1] if i not in dataFE['prefix'][-1]])
    cf = [np.where(dataFE['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelE = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    modelE.fit(dataTra[:, ai], dataTra[:, -3], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pred = modelE.predict(dataTes[:, ai])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(dataTes[:, -3], predictions)
    print('下一事件：', accuracy)
    # 持续时间
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFD['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF - 1) for i in dataFD['index'][-1] if i not in dataFD['prefix'][-1]])
    cf = [np.where(dataFD['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelD = lgb.LGBMRegressor()
    modelD.fit(dataTra[:, ai], dataTra[:, -2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pre = modelD.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -2], y_pre)
    print('持续时间：', MAE)
    # 剩余时间
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFR['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF - 1) for i in dataFR['index'][-1] if i not in dataFR['prefix'][-1]])
    cf = [np.where(dataFR['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelR = lgb.LGBMRegressor()
    modelR.fit(dataTra[:, ai], dataTra[:, -1], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pre = modelR.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -1], y_pre)
    print('剩余时间：', MAE)

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
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
    # 设置Y轴刻度线标签
    ax.set_yticks(range(len(name)))
    ax.set_yticklabels(name)
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('Importance Value')
    plt.ylabel('Feature Name')
    plt.show()

def LightGBMNew(Train, Val, TrainAll, Test, header, catId):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    global ai, ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    list_to_float = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in Val:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_val = dataTra[:, ai]
    y_val = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in TrainAll:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_trainAll = dataTra[:, ai]
    y_trainAll = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_test = dataTra[:, ai]
    y_test = dataTra[:, attribNum:attribNum + 3]

    modelR = lgb.LGBMRegressor()  # max_depth=5
    # modelR.fit(X_trainAll[:, 0:1], y_trainAll[:, 2], feature_name='0', categorical_feature='0')
    # y_pre = modelR.predict(X_test[:, 0:1])
    # MAE = mean_absolute_error(y_test[:, 2], y_pre)
    # print('NR:Activity', MAE)
    modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pre = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('NR:All', MAE)

    # 后向删除消极特征
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
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
                   categorical_feature=[str(i) for i in ai if i in catId])
        y_pred = modelR.predict(X_val[:, ai])
        MAE = mean_absolute_error(y_val[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
                           categorical_feature=[str(i) for i in ai if i in catId])
                y_pred = modelR.predict(X_val[:, ai])
                MAE = mean_absolute_error(y_val[:, 2], y_pred)
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
    timeE = time.time()
    print('NR:Step1', temp3[-1], ',特征选取时间：', timeE - timeS, len(ai))
    # 重要性值画图
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    print("New1_MAE:", MAE)
    timeS = time.time()
    ai.sort()
    ai, aiMAE = showLocalTree(Train, Val, header, ai, catId, 2)
    # 自动选取
    aii = aiMAE.index(min(aiMAE))
    for i in range(aii):
        if aiMAE[i] - aiMAE[aii] < 0.2:  # 0.2 BPIC2012,hd; 0 BPIC2015, production
            aii = i - 1
            break
    FR = [aiMAE[aii], [header[i] for i in ai[0:aii+1]], ai[0:aii+1]]
    timeE = time.time()
    print('NR:Step2', FR, ',特征选取时间：', timeE - timeS, len(ai))
    ai = FR[2]
    modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    print("New2_MAE:", MAE)
    return FR  # FE, FD,

#特征选取参考树
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

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    # modelR = xgb.XGBRegressor()
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
    # 画局部树图
    # tree.show(20, round(minMAE, 3), (maxMAE - minMAE) / 19)
    return aai, aaiMAE

# Top信息增益的特征组合结果
def Top(Train, Test, header, catId):
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

    modelR = lgb.LGBMRegressor()  #max_depth=5
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    # y_pre = modelR.predict(X_test[:, ai])
    # MAE = mean_absolute_error(y_test[:, 2], y_pre)
    # print('All',MAE)
    X3 = modelR.feature_importances_
    # plotFeature(X3, ak)
    aki = sorted(range(len(X3)), reverse=True, key=lambda k: X3[k])
    X3 = X3.tolist()
    X3.sort(reverse=True)
    name = [ak[aki[i]] for i in range(len(aki))]
    sumImp = sum(X3)
    count = 0
    for i in range(len(X3)):
        count += X3[i]
        if count/sumImp > 0.9:
            ai = aki[:i]
            name = name[:i]
            break
    print(ai, name)
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(i) for i in ai if i in cid])
    y_pre = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print('TopMAE', MAE)
    print('end')

# tree = tp.Tree('1')
# tree.root.data = '1'
# tree.root.tag = '1'
# tree.root.value = '1 : 3.3'
# p = tree.root
# q1 = tp.Node(data='4', tag='2', value='4 : 3.0')
# tree.insert(p, q1)
# q2 = tp.Node(data='6', tag='3', value='6 : 3.2')
# tree.insert(p, q2)
# q3 = tp.Node(data='3', tag='4', value='3 : 3.1')
# tree.insert(p, q3)
# q4 = tp.Node(data='6', tag='5', value='6 : 3.1')
# tree.insert(q1, q4)
# q5 = tp.Node(data='3', tag='6', value='3 : 3.05')
# tree.insert(q1, q5)
# q6 = tp.Node(data='6', tag='7', value='6 : 2.9')
# tree.insert(q5, q6)
# # 画局部树图
# tree.show(10, 2.9, 0.4 / 9)