# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import datetime

def normalize_map(data):
    for x in data.columns:
        if x not in ['row_id', 'user_id', 'label', 'shop_id', 
                     'time_stamp', 'mall_id', 'wifi_infos',
                     'longitude', 'latitude', 'weekend', 'night']:
            col_max = data[x].max()
            col_min = data[x].min()
            if col_max == col_min:
                data[x] = -data[x]
            else:
                data[x] = \
                (data[x] - col_min) / (col_max - col_min) * 100
            data[x] = data[x].fillna(0)
    print "Finish normalization!"
    
def add_wk_feature(data):
    combine['time_stamp'] = pd.to_datetime(combine['time_stamp'])
    combine['weekend'] = 0 # 工作日
    combine.loc[combine['time_stamp'].dt.dayofweek > 4, 'weekend'] = 1 # 周末    

def add_night_feature(data):
    combine['night'] = 0 # 白天
    combine.loc[combine['time_stamp'].dt.hour == 12, 'night'] = 1 # 晚上
    combine.loc[combine['time_stamp'].dt.hour == 20, 'night'] = 1 # 晚上
    combine.loc[combine['time_stamp'].dt.hour == 18, 'night'] = 1 # 晚上
    combine.loc[combine['time_stamp'].dt.hour == 19, 'night'] = 1 # 晚上

# main
path = './'
train = pd.read_csv(path + 'train.csv')
train.head(5)
test = pd.read_csv(path + 'X.csv')
test.head(5)
combine = pd.concat([train, test])
combine.info()
# weekend feature
add_wk_feature(combine)
# night or not
add_night_feature(combine)

mall_list = list(set(list(train.mall_id)))
len(mall_list)
# 记录预测结果
result = pd.DataFrame()
start = datetime.datetime.now()
# 对于每一个mall，训练一个多分类器
for mall in mall_list:
    # 提取与这个商场相关的所有数据
    combine1 = combine[combine.mall_id == mall].reset_index(drop=True)
    # wifi_dict统计每个wifi出现的次数
    wifi_dict = {}
    # bhv_row的每一项是一个r(表示每一行行为记录的wifi字典)
    bhv_row = []
    for index, row in combine1.iterrows():
        r = {}
        # b_34366982|-82|false;b_37756289|-53|false;...
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            # r = {b_34366982: -82}
            r[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        bhv_row.append(r)
    
    # 出现次数在mall的所有记录里小于20次
    mobile_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            mobile_wifi.append(i)
    
    m = []
    for row in bhv_row:
        new = {}
        for n in row.keys():
            if n not in mobile_wifi:
                new[n] = row[n]
        m.append(new)
        
    combine1 = pd.concat([combine1, pd.DataFrame(m)], axis=1)
    
    # 归一化映射到0-100
    # normalize_map(combine1)
    
    df_train = combine1[combine1.shop_id.notnull()]
    df_test = combine1[combine1.shop_id.isnull()]

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    num_class = df_train['label'].max() + 1
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class':num_class,
        'silent' : 1
        }
    feature = [x for x in combine1.columns if x not in [
        'row_id', 'user_id', 'label', 'shop_id', 'time_stamp', 
        'mall_id', 'wifi_infos']]
    
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    
    watchlist = [(xgbtrain,'train'), (xgbtrain, 'test')]
    num_rounds = 60
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    df_test['label'] = model.predict(xgbtest)
    df_test['shop_id'] = df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')

result.to_csv(path + 'xgboost-w-n.csv', index=False)
end = datetime.datetime.now()
print "Running time: ", (end - start).seconds, "seconds"