from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string


path=''


# 基于二元模型求得两个字符串中非重合字符串
def distinct_terms(lst1, lst2):
    lst1 = lst1.split(" ")
    lst2 = lst2.split(" ")
    # 获得两个集合的交集
    common = set(lst1).intersection(set(lst2))
    new_lst1 = ' '.join([w for w in lst1 if w not in common])
    new_lst2 = ' '.join([w for w in lst2 if w not in common])
    
    return (new_lst1,new_lst2)

def prepare_distinct(path,out):
    print (path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_distinct_bigram,question2_distinct_bigram\n')
        for t, row in enumerate(DictReader(open(path,encoding='utf-8'), delimiter=',')):
            if c%100000==0:
                print ('finished',c)
            q1 = str(row['question1_bigram'])
            q2 = str(row['question2_bigram'])
            coo_terms = distinct_terms(q1,q2)
            outfile.write('%s,%s\n' % coo_terms)
            c+=1
        end = datetime.now()
    print ('times:',end-start)

prepare_distinct(path+'train_bigram.csv',path+'train_distinct_bigram.csv')
prepare_distinct(path+'test_bigram.csv',path+'test_distinct_bigram.csv')
