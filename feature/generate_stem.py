##
##剔除非法单词，重新组合句子
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

seed = 1024
np.random.seed(seed)
train =  pd.read_csv('../data/train.csv', header=0)
test =  pd.read_csv('../data/test.csv', header=0)

def stem_str(x,stemmer=SnowballStemmer('english')):
    # 将不以0-9，a-z,A-Z开头的句子替换为空，也就是排除不符规则的句子
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    # 将句子分词并重新组合为句子
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

# 英文分词工具
porter = PorterStemmer()
# 语言转换库，支持15中语言
snowball = SnowballStemmer('english')


print('Generate porter')

# 将训练文档question1_porter列句子重新分词组合
train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
#
train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
#
print("开始打印")
train.to_csv('train_porter.csv')
test.to_csv('test_porter.csv')

print("end")