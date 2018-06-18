import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb

def split_train_valid(df_train ,test_size=0.2):
    '''
    k-fold交叉验证,默认k=10
    df_train:训练数据
    '''
    X_train, X_vali, y_train, y_vali = train_test_split(df_train[features], df_train[label], test_size=test_size, random_state=40000)
    # added some parameters

    #     dtrain = df_train.iloc[train_list]
    #     dvali =  df_train.iloc[vali_list]

    dtrain = xgb.DMatrix(X_train ,label=y_train)
    dvalid = xgb.DMatrix(X_vali ,label=y_vali)
    watchlist = [(dtrain, 'train') ,(dvalid, 'valid')]

    return dtrain, dvalid, watchlist


def valid_score(pre_result, real_result):
    '''
    score = 0.6 * acu + 0.4 * F1
    :param pre_result:
    :param real_result:
    :return:
    '''
    auc = roc_auc_score(real_result['label'], pre_result['score'])
    f1 = f1_score(real_result['label'], pre_result['label'])
    score = 0.6 * auc + 04 * f1
    print "auc = %f, f1 = %f, score = %f" % (auc, f1, score)
    return score