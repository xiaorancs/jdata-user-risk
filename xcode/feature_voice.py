# _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

## voice features
#### 先对所有的时间进行特征提取
# 1. 用户电话的总的通话次数 opp_num
# 2. 通话的人数，voice_all_unique_cnt
# 3. 通话次数 / 人数的比例， voice_all_cnt_all_unique_cnt_rate
# 4. 对端电话的前n位的个数，所有的不同号码的个数。 opp_head
# 5. 对端号码长度的分布个数   opp_len
# 6. 通话最大时长，平均时长，最小时长，极差时长等统计的信息  start_time, end_time
# 7. 通话类型的分布个数或者比例 call_type
# 8. 通话类型的分布个数和比例   in_out

def get_voice_feature(df_train_voice, target='train'):
    if target == 'train':
        # 复制lable的数据，作为所有的特征的标示
        df_train_label = pd.read_csv('../data/train/uid_train.txt',sep='\t',low_memory=False)
        df_train = df_train_label.copy()
    else:
        tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['uid'].count())
        df_train = pd.DataFrame(data={'uid': tmp.index})

    tmp = df_train_voice.groupby('uid', as_index=True)['opp_num'].unique()
    uids = tmp.index
    opp_nums = []
    for opp_num in tmp:
        opp_nums.append(len(opp_num))

    df_tmp = pd.DataFrame(data={'uid': uids, 'voice_all_opp_num_unique_cnt': opp_nums})
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['opp_num'].count())
    df_tmp.columns = ['voice_all_opp_num_cnt']
    df_tmp['uid'] = df_tmp.index

    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # 通话次数 / 通话的人数
    df_train['voice_opp_num_all_cnt_unique_cnt_rate'] = df_train['voice_all_opp_num_cnt'] / df_train[
        'voice_all_opp_num_unique_cnt']

    # 全部的不同开头的次数,唯一的标示
    tmp = df_train_voice.groupby('uid', as_index=True)['opp_head'].unique()
    uids = tmp.index
    opp_nums = []
    for opp_num in tmp:
        opp_nums.append(len(opp_num))

    df_tmp = pd.DataFrame(data={'uid': uids, 'voice_all_opp_head_unique_cnt': opp_nums})
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # 通话次数 / opp_len的次数
    df_train['voice_opp_head_all_cnt_unique_cnt_rate'] = df_train['voice_all_opp_num_cnt'] / df_train[
        'voice_all_opp_head_unique_cnt']

    # 通话最多的head的个数，
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['opp_head'].value_counts().unstack().max(axis=1))
    df_tmp.columns = ['voice_all_opp_head_max']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # 通话最小的head的个数，
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['opp_head'].value_counts().unstack().min(axis=1))
    df_tmp.columns = ['voice_all_opp_head_min']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # 极差，和占总个比例
    df_train['voice_all_opp_head_jc'] = df_train['voice_all_opp_head_max'] - df_train['voice_all_opp_head_min']
    df_train['voice_opp_head_all_max_rate'] = df_train['voice_all_opp_head_max'] / df_train['voice_all_opp_num_cnt']

    # call_type 分布
    df_tmp = df_train_voice.groupby('uid', as_index=True)['call_type'].value_counts().unstack()
    df_tmp.columns = ['voice_all_call_type_' + str(i) for i in range(1, 6)]
    df_tmp['uid'] = df_tmp.index
    df_tmp.fillna(0, inplace=True)

    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # call_type 分布
    df_tmp = df_train_voice.groupby('uid', as_index=True)['in_out'].value_counts().unstack()

    df_tmp.columns = ['voice_all_in_out_' + str(i) for i in range(2)]
    df_tmp['uid'] = df_tmp.index
    df_tmp.fillna(0, inplace=True)

    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # diff_time
    df_train_voice['diff_time'] = df_train_voice['end_time'] - df_train_voice['start_time']

    # sum
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].sum())
    df_tmp.columns = ['voice_all_diff_time_sum']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # meam
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].mean())
    df_tmp.columns = ['voice_all_diff_time_avg']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    # max
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].max())
    df_tmp.columns = ['voice_all_diff_time_max']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')
    # min
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].min())
    df_tmp.columns = ['voice_all_diff_time_min']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')
    # std
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].std())
    df_tmp.columns = ['voice_all_diff_time_std']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')
    # skew
    df_tmp = pd.DataFrame(df_train_voice.groupby('uid', as_index=True)['diff_time'].skew())
    df_tmp.columns = ['voice_all_diff_time_skew']
    df_tmp['uid'] = df_tmp.index
    df_train = pd.merge(df_train, df_tmp, on='uid', how='left')

    df_train['voice_all_diff_time_jc'] = df_train['voice_all_diff_time_max'] - df_train['voice_all_diff_time_min']
    df_train['voice_all_diff_time_fd'] = df_train['voice_all_diff_time_std'] / df_train['voice_all_diff_time_avg']

    return df_train




