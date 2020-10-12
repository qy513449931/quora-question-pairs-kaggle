import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import difflib
import re
from string import punctuation
from gensim.models import Word2Vec
from nltk.corpus import brown
nltk.download('stopwords')
stops = set(stopwords.words("english"))
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from IPython.display import display  # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve,f1_score
import  datetime

# 获得词频占比
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

# 定义停用词
stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then',
              'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to',
              'What', 'Which',
              'Is', 'If', 'While', 'This']

# 加载数据
train = pd.read_csv('../data/train.csv')[:]
test = pd.read_csv('../data/test.csv')[:]
# 空值填充
train = train.fillna('empty')
test = test.fillna('empty')

# 加载TFIDF词向量模型
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
# 获得训练集和测试集组成的序列
tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test[
    'question2'].tolist()).astype(str)
# 拟合数据
tfidf.fit_transform(tfidf_txt)
words = (" ".join(tfidf_txt)).lower().split()
# 获得总单词数
counts = Counter(words)
# 获得单词权重字典
weights = {word: get_weight(count) for word, count in counts.items()}

br = Word2Vec(brown.sents())


# 获得两个文本的距离，1是完全一样
def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()


def get_similar(word):
    if word in br:
        # 获得单词相关的前5个单词
        lis = br.most_similar(word, topn=5)
        ret = []
        for one in lis:
            ret.append(one[0])
        return ret
    else:
        return [word]


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # 处理数据
    # 删除？和,
    text = text.rstrip('?')
    text = text.rstrip(',')
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        # 提取单词词干信息
        stemmed_words = [nltk.PorterStemmer().stem(word.lower()) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


# 获得句子之间公共单词占比
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


# 获得公共单词权重之和和总单词权重占比
def tfidf_word_match_share(q1words, q2words):
    word_1 = []
    for word in q1words:
        word_1.extend(get_similar(word))

    word_2 = []
    for word in q2words:
        word_2.extend(get_similar(word))

    # 获得公共单词的权重之和
    shared_weights = [0] + [weights.get(w, 0) for w in word_1 if w in word_2] + [weights.get(w, 0) for w in word_2 if
                                                                                 w in word_1]
    total_weights = [weights.get(w, 0) for w in word_1] + [weights.get(w, 0) for w in word_2]

    if (np.sum(shared_weights) == 0):
        return 0

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


# 获得特征
def get_features(df_features):
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print("stem_tfidf")
    # 句子提取词干
    df_features['q1_stem'] = df_features.question1.map(
        lambda x: [w for w in nltk.PorterStemmer().stem(str(x).lower()).split(' ')])
    df_features['q2_stem'] = df_features.question2.map(
        lambda x: [w for w in nltk.PorterStemmer().stem(str(x).lower()).split(' ')])
    # df_features['z_adj_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_adjs if w in r.question2_adjs]), axis=1)  #takes long
    # 获得句子之间公共单词权重占比
    df_features['z_stem_tfidf'] = df_features.apply(lambda r: tfidf_word_match_share(r.q1_stem, r.q2_stem), axis=1)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('nouns...')
    # 句子分词，词性标注，获得是名词的单词
    df_features['question1_nouns'] = df_features.question1.map(
        lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['question2_nouns'] = df_features.question2.map(
        lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # 获得句子1与句子2中交集名词单词个数
    df_features['z_noun_match'] = df_features.apply(
        lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  # takes long
    print('lengths...')
    # 获得句子长度
    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))
    # 获得句子中单词个数
    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('difflib...')

    # 获得文本距离
    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2),
                                                     axis=1)  # takes long
    print('word match...')
    # 两个句子中存在交集单词占总单词的比重
    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)
    print('tfidf...')
    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))
    return df_features.fillna(0.0)

#句子停用词过滤
train['question1'] =  train.question1.map(lambda x:text_to_wordlist(x))
train['question2'] = train.question2.map(lambda x:text_to_wordlist(x))
train = get_features(train)
# 获得初步特征组合并保存文件
train.to_csv('train_feature_clean.csv', index=False)
