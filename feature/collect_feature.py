from __future__ import division
import pandas as pd
import numpy as np

origincol = ['id','qid1','qid2','is_duplicate','z_stem_tfidf','z_noun_match','z_match_ratio','z_word_match','z_tfidf_sum1','z_tfidf_sum2','z_tfidf_mean1','z_tfidf_mean2','z_tfidf_len1','z_tfidf_len2']
origincol2= ['test_id','z_stem_tfidf','z_noun_match','z_match_ratio','z_word_match','z_tfidf_sum1','z_tfidf_sum2','z_tfidf_mean1','z_tfidf_mean2','z_tfidf_len1','z_tfidf_len2']
copycol4 = ['q1_freq','q2_freq']
copycol7 = ['z_q1_q2_intersect']
copycol15 = ['q1_q2_wm_ratio']
copycol17 = ['f_q1_pr','f_q2_pr']
copycol12 = ['z3_cosine','z3_manhatton','z3_euclidean','z3_pearson','z3_spearman','z3_kendall']

trainfeaure_old = pd.read_csv('feature/train_feature_clean.csv', usecols= origincol)[:]
testfeature_old = pd.read_csv('feature/test_feature_clean.csv', usecols = origincol2)[:]

trainfeaure_new8 = pd.read_csv('feature/train_magic2.csv',usecols =copycol7)[:]
testfeature_new8 = pd.read_csv('feature/test_magic2.csv',usecols =copycol7)[:]

trainfeaure_new5 = pd.read_csv('feature/train_freq.csv',usecols =copycol4)[:]
testfeature_new5 = pd.read_csv('feature/test_freq.csv',usecols =copycol4)[:]

trainfeaure_new16 = pd.read_csv('feature/new_magic_train.csv',usecols =copycol15)[:]
testfeature_new16 = pd.read_csv('feature/new_magic_test.csv',usecols =copycol15)[:]

trainfeaure_new13 = pd.read_csv('feature/train_doc2vec3.csv',usecols =copycol12)[:]
testfeature_new13 = pd.read_csv('feature/test_doc2vec3.csv',usecols =copycol12)[:]

trainfeaure_new17 = pd.read_csv('feature/train_ix.csv')[:]
testfeature_new17 = pd.read_csv('feature/test_ix.csv')[:]

trainfeaure_new19 = pd.read_csv('feature/pagerank_train.csv',usecols =copycol17)[:]
testfeature_new19 = pd.read_csv('feature/pagerank_test.csv',usecols =copycol17)[:]

train = trainfeaure_old
test = testfeature_old

for key in copycol7:
	train[key] = trainfeaure_new8[key]
	test[key] = testfeature_new8[key]

for key in copycol12:
	train[key] = trainfeaure_new13[key]
	test[key] = testfeature_new13[key]

for key in copycol4:
	train['f_'+key] = trainfeaure_new5[key]
	test['f_'+key] = testfeature_new5[key]

for key in copycol15:
	train['f_'+key] = trainfeaure_new16[key]
	test['f_'+key] = testfeature_new16[key]

for key in copycol17:
	train[key] = trainfeaure_new19[key]
	test[key] = testfeature_new19[key]

pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]
#p = 0.165
# p = 0.174
# scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
# while scale > 1:
#     neg_train = pd.concat([neg_train, neg_train])
#     scale -=1
scale = 0.8
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])
train = train.sample(frac=1).reset_index(drop=True)


train.to_csv('train_feature_clear_magic_all.csv', index=False)
test.to_csv('test_feature_clear_magic_all.csv', index=False)