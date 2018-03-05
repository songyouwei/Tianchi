# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
path = './'

y = pd.read_csv(path + 'y.csv')

print y['shop_id']

predict_m = []
predict_m.append(pd.read_csv(path + 'randomforest-baseline.csv').sort_values(by=['row_id'])['shop_id'].reset_index(drop=True))
predict_m.append(pd.read_csv(path + 'randomforest-w-n.csv').sort_values(by=['row_id'])['shop_id'].reset_index(drop=True))
predict_m.append(pd.read_csv(path + 'xgboost-w-n.csv').sort_values(by=['row_id'])['shop_id'].reset_index(drop=True))
predict_m.append(pd.read_csv(path + 'xgboost-baseline.csv').sort_values(by=['row_id'])['shop_id'].reset_index(drop=True))
predict_m.append(pd.read_csv(path + 'ovr-lr-w-n.csv').sort_values(by=['row_id'])['shop_id'].reset_index(drop=True))


predict = []
for t_i in range(len(y)):
    print t_i
    class_count = {}
    for i in range(5):
        vote_label = predict_m[i][t_i]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    print class_count
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    print sorted_class_count[0][0]
    predict.append(sorted_class_count[0][0])

print predict
accuracy = accuracy_score(y['shop_id'], predict)
print "accuracy: %f\" % accuracy"