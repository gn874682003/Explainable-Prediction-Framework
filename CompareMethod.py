from sklearn.feature_selection import RFECV
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error#roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from EHM.DataRecord import DataRecord as DR
import warnings
warnings.simplefilter('ignore', UserWarning)
import gc
gc.enable()

def RFECV_Method(Train, Test, header, catId):
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
    y_train = dataTra[:, -1]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, -1]

    clf = lgb.LGBMRegressor()

    rfecv = RFECV(estimator=clf,  # 学习器
                  min_features_to_select=2,  # 最小选择的特征数量
                  step=1,  # 移除特征个数
                  cv=KFold(5),  # 交叉验证次数
                  scoring='neg_mean_absolute_error',  # 学习器的评价标准
                  verbose=0,
                  n_jobs=1
                  ).fit(X_train, y_train)  # (xtr, ytr)#(X,y)#
    # X_RFECV = rfecv.transform(X)
    print("RFECV特征选择结果——————————————————————————————————————————————————")
    print("有效特征个数 : %d" % rfecv.n_features_)
    print("全部特征等级 : %s" % list(rfecv.ranking_))
    allFea = rfecv.ranking_
    aii = [i for i in range(allFea.size) if allFea[i] == 1]
    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:, aii], y_train, feature_name=[str(i) for i in aii],
               categorical_feature=[str(i) for i in aii if i in catId])
    y_pred = modelR.predict(X_test[:, aii])
    MAE = mean_absolute_error(y_test, y_pred)
    FR = [MAE, [header[i] for i in aii], aii]
    print(FR)



def get_feature_importances(data, categorical_feats, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['TARGET', 'SK_ID_CURR']]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['TARGET'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    # lgb_params = {
    #     'objective': 'binary',
    #     'boosting_type': 'rf',
    #     'subsample': 0.623,
    #     'colsample_bytree': 0.7,
    #     'num_leaves': 127,
    #     'max_depth': 8,
    #     'seed': seed,
    #     'bagging_freq': 1,
    #     'n_jobs': 4
    # }
    lgb_params = {'objective':'regression','boosting_type': 'gbdt'}#
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_absolute_error(y, clf.predict(data[train_features]))

    return imp_df

def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    # plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    # plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())

def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
    # Fit LightGBM
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    # lgb_params = {
    #     'objective': 'binary',
    #     'boosting_type': 'gbdt',
    #     'learning_rate': .1,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'num_leaves': 31,
    #     'max_depth': -1,
    #     'seed': 13,
    #     'n_jobs': 4,
    #     'min_split_gain': .00001,
    #     'reg_alpha': .00001,
    #     'reg_lambda': .00001,
    #     'metric': 'auc'
    # }
    lgb_params = {'objective': 'regression', 'boosting_type': 'gbdt'}
    # Fit the model
    hist = lgb.cv(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        categorical_feature=cat_feats,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]

def NullImportance():
    Train = copy.deepcopy(DR.Train)
    header = DR.header.copy()
    data = []
    for line in Train:
        for line2 in line:
            line2.pop(-2)
            line2.pop(-2)
            data.append(line2)
    data = pd.DataFrame(data)
    header.pop(-2)
    header.pop(-2)
    header[-1] = 'TARGET'
    data.columns = header

    # data = pd.read_csv('../Dataset/hd.csv')

    categorical_feats = [f for f,i in zip(data.columns,DR.State) if i < 3]

    for f_ in categorical_feats:
        data[f_], _ = pd.factorize(data[f_])
        # Set feature type as categorical
        data[f_] = data[f_].astype('category')

    # Seed the unexpected randomness of this world
    np.random.seed(123)
    # Get the actual importance, i.e. without shuffling
    actual_imp_df = get_feature_importances(data, categorical_feats,shuffle=False)
    actual_imp_df.head()
    null_imp_df = pd.DataFrame()
    nb_runs = 80
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(data,categorical_feats, shuffle=True)
        imp_df['run'] = i + 1
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)
    null_imp_df.head()
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='LIVINGAPARTMENTS_AVG')
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='CODE_GENDER')
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='EXT_SOURCE_1')
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='EXT_SOURCE_2')
    display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='EXT_SOURCE_3')
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

    plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    plt.tight_layout()

    a = scores_df.sort_values('split_score', ascending=False).iloc[0:70]
    b = []
    c = []
    for i in a.index:
        if a['split_score'][i] > 0:
            b.append(a['feature'][i])
            c.append(i)
    return b, c

    # null_imp_df.to_csv('null_importances_distribution_rf.csv')
    # actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')

#     correlation_scores = []
#     for _f in actual_imp_df['feature'].unique():
#         f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
#         f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
#         gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
#         f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
#         f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
#         split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
#         correlation_scores.append((_f, split_score, gain_score))
#
#     corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
#
#     fig = plt.figure(figsize=(16, 16))
#     gs = gridspec.GridSpec(1, 2)
#     # Plot Split importances
#     ax = plt.subplot(gs[0, 0])
#     sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
#     ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
#     # Plot Gain importances
#     ax = plt.subplot(gs[0, 1])
#     sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
#     ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
#     plt.tight_layout()
#     plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
#     fig.subplots_adjust(top=0.93)
#     plt.show()
#
#     features = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
#     print(features)
# # score_feature_selection(df=data[features], train_features=features, target=data['TARGET'])
#
#
#
# for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
#     split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
#     split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
#     gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
#     gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
#
#     print('Results for threshold %3d' % threshold)
#     split_results = score_feature_selection(df=data, train_features=split_feats, cat_feats=split_cat_feats,
#                                             target=data['TARGET'])
#     print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
#     gain_results = score_feature_selection(df=data, train_features=gain_feats, cat_feats=gain_cat_feats,
#                                            target=data['TARGET'])
#     print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
#
#
