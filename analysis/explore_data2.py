import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import matplotlib.pyplot as plt


df_train = pd.read_csv('../data/train.csv')
df_train.head()
df_test = pd.read_csv('../data/test.csv')
df_test.head()
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    # 去除停用词
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    # 计算此问题的词语在另个问题的个数
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    # 共有的词语/所有的词语
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

plt.figure(figsize=(15, 5))
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)
plt.show()
