import datetime
import numpy as np, pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def wifi_preprocess(df):
    # wifi_dict统计每个wifi出现的次数
    wifi_dict = {}
    # bhv_row的每一项是一个r(表示每一行行为记录的wifi字典)
    bhv_row = []
    for index, row in df.iterrows():
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
    df = pd.concat([df, pd.DataFrame(m)], axis=1)
    return df

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
    return data

# start = datetime.datetime.now()
print('start read data')
shop_info = pd.read_csv('ccf_first_round_shop_info.csv')
user_shop_behavior = pd.read_csv('ccf_first_round_user_shop_behavior.csv')
# 合并的数据集
df = pd.merge(user_shop_behavior ,shop_info[['shop_id','mall_id']],how='left',on='shop_id')

# 店铺id列表
mall_id_list=list(set(list(shop_info.mall_id)))

# 选定test_mall_id的商场数据
test_mall_id = mall_id_list[0]
print('current mall id:', test_mall_id)
sub_df_mall = df[df.mall_id == test_mall_id].reset_index(drop=True)
# test_mall_id商场的店铺id列表
shop_id_list = list(set(list(sub_df_mall.shop_id)))
print('mall', test_mall_id, 'has', len(shop_id_list), 'shops')

print('start preprocess')

# fattern wifi_infos into columns
sub_df_mall = wifi_preprocess(sub_df_mall)
# normalize
sub_df_mall = normalize_map(sub_df_mall)

print('preprocess complete')

# feature
feature = [x for x in sub_df_mall.columns if x not in ['row_id', 'user_id', 'label', 'shop_id',
                 'time_stamp', 'mall_id', 'wifi_infos',
                 'longitude', 'latitude', 'weekend', 'night']]

# 按时间分割训练集和测试集
trainset = sub_df_mall[sub_df_mall['time_stamp'] < '2017-08-20 14:00']
testset = sub_df_mall[sub_df_mall['time_stamp'] >= '2017-08-20 14:00']
print('testset length:', len(testset))
trainset_shop_id_list = list(set(list(trainset.shop_id)))


trainXcolumns = list(feature)
trainXcolumns.append('label')

lbl = preprocessing.LabelEncoder()
lbl.fit(list(trainset['shop_id'].values))
# 最终计算的预测表，
likelihood_shopid_testid = np.zeros([len(trainset_shop_id_list), len(testset)])

# 针对训练集中的每个shop，计算一个trainX，去fit，predict，得到likelihood
for shop_id_index, shop_id in enumerate(trainset_shop_id_list):
    print('=============================================')
    print('start compute trainX and likelihoods for shop', shop_id)
    ## 筛选属于这个店铺的训练集
    sub_df_shop = trainset[trainset.shop_id == shop_id].reset_index(drop=True)
    mean_longitude = sub_df_shop.longitude.mean()
    mean_latitude = sub_df_shop.latitude.mean()
    # query集
    sub_df_shop_query = sub_df_shop[sub_df_shop.index%2 == 1]
    print('sub_df_shop_query length:', len(sub_df_shop_query))
    # sample集 1
    sub_df_shop_sample = sub_df_shop[sub_df_shop.index%2 == 0]
    # others 0
    sub_df_othershop = trainset[(trainset.shop_id != shop_id) & \
                                 (abs(trainset.longitude - mean_longitude) < 0.1) & \
                                 (abs(trainset.latitude - mean_latitude) < 0.1)] \
                                 .sort_values(by=['longitude', 'latitude']) \
                                 .head(len(sub_df_shop_query)) \
                                 .reset_index(drop=True)
    print('sub_df_othershop length:', len(sub_df_othershop))

    # 只保留wifi数值
    sub_df_shop_query = sub_df_shop_query[feature]
    sub_df_shop_sample = sub_df_shop_sample[feature]
    sub_df_othershop = sub_df_othershop[feature]
    testset = testset[feature]

    print('start compute trainX')
    start = datetime.datetime.now()
    trainX = pd.DataFrame(columns=trainXcolumns)
    for i,query_row in sub_df_shop_query.iterrows():
        # if i > 1:
        #     break
        for j,sample_row in sub_df_shop_sample.iterrows():
            diff = query_row.values - sample_row.values
            row = np.concatenate((diff,[1]))
            rowdf = pd.DataFrame([row],columns=trainXcolumns)
            trainX = trainX.append(rowdf, ignore_index=True)
        for j,other_row in sub_df_othershop.iterrows():
            diff = query_row.values - other_row.values
            row = np.concatenate((diff, [0]))
            rowdf = pd.DataFrame([row], columns=trainXcolumns)
            trainX = trainX.append(rowdf, ignore_index=True)

    end = datetime.datetime.now()
    print("compute trainX for this shop costs", (end - start).seconds, "seconds")
    # fit
    trainX_X = trainX[feature]
    trainX_label = trainX['label']
    clf = RandomForestClassifier(n_estimators=5)
    print('start fit')
    clf.fit(trainX_X, trainX_label)

    print('start compute likelihoods')
    start = datetime.datetime.now()
    # predict
    for i,test_row in testset.iterrows():
        # if i > 1:
        #     break
        diff_rows = []
        for i,query_row in sub_df_shop_query.iterrows():
            diff = query_row.values - test_row.values
            diff_rows.append(diff)
        test_rowX = pd.DataFrame(data=diff_rows,columns=list(feature))
        test_rowY = clf.predict(test_rowX)
        likelihood = test_rowY.sum()/len(test_rowY)
        # 加入likelihood表
        likelihood_shopid_testid[shop_id_index][i] = likelihood
    end = datetime.datetime.now()
    print("compute likelihoods for this shop costs", (end - start).seconds, "seconds")

print('likelihoods compute complete')
predicted_shop_id_indexes = likelihood_shopid_testid.argmax(axis=0)
predicted_shop_ids = lbl.inverse_transform(predicted_shop_id_indexes)
result = pd.DataFrame(data=predicted_shop_ids)
result.to_csv('binary_classify.csv', index=False)

# end = datetime.datetime.now()
# print("Running time: ", (end - start).seconds, "seconds")
