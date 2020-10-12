 #调参后的模型
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, precision_score, recall_score
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, svm, ensemble

data = pd.read_csv('train_feature_clear_magic_all.csv')
# 取得训练列
col = [c for c in data.columns if (c[:1] == 'z' or c[:1] == 'f')]
income_raw = data['is_duplicate']
features_raw = data.drop('is_duplicate', axis=1)

features_raw = features_raw[col]
# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_raw, income_raw, test_size=0.2, random_state=0,
                                                    stratify=income_raw)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                  stratify=y_train)
print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape)
print("y_test.shape", y_test.shape)
print("X_val.shape", X_val.shape)
print("y_val.shape", y_val.shape)
colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm', 'brown', 'r']
lw = 3

plt.figure(figsize=(8, 8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for different classifiers')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

labels = []

clf_A = tree.DecisionTreeClassifier(random_state=1, max_depth=5)
# clf_B = svm.SVC(random_state=1)

# clf_C = ensemble.AdaBoostClassifier(random_state=1)
clf_D = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)

clf_E = LogisticRegression(class_weight='balanced')
idx = 0

for learner in [clf_A, clf_D, clf_E]:
    idx = idx + 1
    clf_name = learner.__class__.__name__
    start = time.time()
    learner.fit(X_train, y_train)
    end = time.time()
    print(clf_name, "训练时间", end - start)
    start = end
    # print( "预测值",clf.predict(X_val))
    end = time.time()
    print(clf_name, "测试时间", end - start)
    y_pred = learner.predict(X_val)
    f1score = f1_score(y_val, y_pred)
    print(clf_name, "f1_score:", f1score)

    p = precision_score(y_val, y_pred)
    print(clf_name, "precision_score:", p)
    r = recall_score(y_val, y_pred)
    print(clf_name, "recall_score:", r)
    # clf.predict_proba(X_val)[:,1]取第二列数据，因为第二列概率为趋于0时分类类别为0，概率趋于1时分类类别为1
    fpr, tpr, _ = roc_curve(y_val, learner.predict_proba(X_val)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=colors[idx])
    labels.append("Model: {}, AUC = {}".format(clf_name, np.round(roc_auc, 4)))

plt.legend(['random AUC = 0.5'] + labels)
plt.show()