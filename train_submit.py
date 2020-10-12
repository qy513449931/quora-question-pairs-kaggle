from __future__ import division
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split

train=pd.read_csv("train_feature_clear_magic_all.csv")
test=pd.read_csv("test_feature_clear_magic_all.csv")
col = [c for c in train.columns if (c[:1] == 'z' or c[:1] == 'f')]

x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'].values, test_size=0.1, random_state=0)

params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.05
params["subsample"] = 0.8
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 8
params["silent"] = 1
params["seed"] = 1632
params["gama"] = 0.005

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 700, watchlist, early_stopping_rounds=60, verbose_eval=100)
# 打印损失函数
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))
# 生成提交文件
sub.to_csv('submit.csv', index=False)

