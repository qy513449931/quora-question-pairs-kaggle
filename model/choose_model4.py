import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score, accuracy_score
import pickle
import xgboost as xgb
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm, ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """

    # Create figure
    fig = pl.figure(figsize=(11, 5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_val', 'f_val']):
            for i in np.arange(3):
                # Creative plot code
                ax[j // 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
              loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    pl.subplots_adjust(top=0.85, bottom=0., left=0.10, right=0.95, hspace=0.3, wspace=0.35)
    pl.show()


def feature_plot(importances, X_train, y_train):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    rects = pl.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', \
                   label="Feature Weight")

    # make bar chart higher to fit the text label
    axes = pl.gca()
    axes.set_ylim([0, np.max(values) * 1.1])

    # add text label on each bar
    delta = np.max(values) * 0.02

    for rect in rects:
        height = rect.get_height()
        pl.text(rect.get_x() + rect.get_width() / 2.,
                height + delta,
                '%.2f' % height,
                ha='center',
                va='bottom')

    # Detect if xlabels are too long
    rotation = 0
    for i in columns:
        if len(i) > 20:
            rotation = 10  # If one is longer than 20 than rotate 10 degrees
            break
    pl.xticks(np.arange(5), columns, rotation=rotation)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()


def train_predict(learner, sample_size, X_train, y_train, X_val, y_val):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''

    results = {}

    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # 获得程序开始时间
    learner = learner.fit(X_train[: sample_size], y_train[: sample_size])
    end = time()  # 获得程序结束时间

    results['train_time'] = end - start

    start = time()  # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # 获得程序结束时间

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # 计算在验证上的准确率
    results['acc_val'] = accuracy_score(predictions_val, y_val)

    # 计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[: 300], predictions_train, beta=0.5)

    # 计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, beta=0.5)

    # 成功
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # 返回结果
    return results


data = pd.read_csv('../feature/train_feature_clear_magic_all.csv')
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
display(X_train.head())
clf_A = tree.DecisionTreeClassifier(random_state=1)
clf_B = svm.SVC(random_state=1)
clf_C = ensemble.AdaBoostClassifier(random_state=1)
clf_D = xgb.XGBClassifier()
clf_E = LogisticRegression()
# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(len(X_train) * 0.01)
samples_10 = int(len(X_train) * 0.1)
samples_100 = int(len(X_train) * 1)
print(samples_1, samples_10, samples_100)
# # 收集学习器的结果
results = {}
for clf in [clf_A, clf_D, clf_E]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 结果序列化存储
with open('results.data.model', 'wb') as f:
    pickle.dump(results, f)

accuracy = 0
fscore = 0
# 对选择的三个模型得到的评价结果进行可视化
evaluate(results, accuracy, fscore)





