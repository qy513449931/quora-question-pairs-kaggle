from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_feature_clear_magic_all.csv')
# 取得训练列
col = [c for c in data.columns if (c[:1] == 'z' or c[:1] == 'f')]
income_raw = data['is_duplicate']
features_raw = data.drop('is_duplicate', axis=1)
xgb_ = xgb.XGBClassifier(max_depth=5, learning_rate=0.05)
# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_raw, income_raw, test_size=0.2, random_state=0,
                                                    stratify=income_raw)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                  stratify=y_train)
X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=0.1, random_state=0,
                                                      stratify=y_train)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                      stratify=y_train)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

X_train4, X_val4, y_train4, y_val4 = train_test_split(X_train, y_train, test_size=0.4, random_state=0, stratify=y_train)
X_train5, X_val5, y_train5, y_val5 = train_test_split(X_train, y_train, test_size=0.5, random_state=0, stratify=y_train)

X_trainList = []
X_valList = []
y_trainList = []
y_valList = []
X_trainList.append(X_train1)
X_trainList.append(X_train2)
X_trainList.append(X_train3)
X_trainList.append(X_train4)
X_trainList.append(X_train5)
X_valList.append(X_val1)
X_valList.append(X_val2)
X_valList.append(X_val3)
X_valList.append(X_val4)
X_valList.append(X_val5)
y_trainList.append(y_train1)
y_trainList.append(y_train2)
y_trainList.append(y_train3)
y_trainList.append(y_train4)
y_trainList.append(y_train5)
y_valList.append(y_val1)
y_valList.append(y_val2)
y_valList.append(y_val3)
y_valList.append(y_val4)
y_valList.append(y_val5)

x_input = []
y_train = []
y_test = []
for i in range(0, 5, 1):
    x_input.append(str(0.1+0.1*i))
    # 训练集上的准确率
    accuracy = cross_val_score(xgb_, X_trainList[i], y_trainList[i], cv=10, scoring='accuracy').mean()
    y_train.append(str(accuracy))
    # 验证集上的准确率
    accuracy = cross_val_score(xgb_, X_valList[i], y_valList[i], cv=10, scoring='accuracy').mean()
    y_test.append( str(accuracy) )

# 打印统计数值
print(x_input)
print(y_train)
print(y_test)

x_input = [0.1, 0.2, 0.3, 0.4, 0.5]
y_train = [0.8976, 0.8976, 0.8977, 0.8978, 0.8983]
y_test = [0.8975, 0.8977, 0.8976, 0.8974, 0.8972]
# plt.plot(squares)
plt.plot(x_input, y_train, linewidth=5)

plt.plot(x_input, y_test, linewidth=5)
# 设置图表标题，并给坐标轴加标签
plt.title("accuracy ", fontsize=24)
plt.xlabel("data size", fontsize=14)
plt.ylabel("accuracy", fontsize=14)

# 设置刻度标记的大小
plt.tick_params(axis='both', labelsize=14)

plt.legend(['train data', 'validation data'])
plt.ylim(0.894, 0.9)
plt.show()