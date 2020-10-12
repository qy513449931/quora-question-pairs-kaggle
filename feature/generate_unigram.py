from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
from ngram import getUnigram
import string


path = ''


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

# 排除句子中的标点符号，并重新组合句子
def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def prepare_unigram(path,out):
    print (path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_unigram,question2_unigram\n')
        # DictReader 循环读取csv内容
        for t, row in enumerate(DictReader(open(path,mode='r',encoding='utf-8'), delimiter=',')):
            if c%100000==0:
                print ('finished',c)
            # 句子过滤指定标点符号，个人觉得完全用正则就可以
            q1 = remove_punctuation(str(row['question1_porter']).lower()).split(' ')
            q2 = remove_punctuation(str(row['question2_porter']).lower()).lower().split(' ')
            # 获得一元模型，也就是单次之间的出现是独立事件，不依赖其他单词，需要打断点看看，返回一个List，个人觉得这个函数getUnigram，除了判断入参是否为List
            # 类型外，貌似没干其他事
            q1_bigram = getUnigram(q1)
            q2_bigram = getUnigram(q2)
            # 重新将数组组合为句子
            q1_bigram = ' '.join(q1_bigram)
            q2_bigram = ' '.join(q2_bigram)
            outfile.write('%s,%s\n' % (q1_bigram, q2_bigram))
            c+=1


    print ('times:',datetime.now()-start)

prepare_unigram(path+'train_porter.csv',path+'train_unigram.csv')
prepare_unigram(path+'test_porter.csv',path+'test_unigram.csv')
