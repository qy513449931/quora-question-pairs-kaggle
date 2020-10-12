
import pandas as pd
from collections import defaultdict

train_orig = pd.read_csv('../data/train.csv', header=0)
test_orig = pd.read_csv('../data/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], test_orig[['question1', 'question2']]], axis=0).reset_index(
    drop='index')
q_dict = defaultdict(set)

for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_orig['z_q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['z_q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

train_orig['z_q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['z_q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

col = [c for c in train_orig.columns if c[:1] == 'z']
train_orig.to_csv('train_magic2.csv', index=False, columns=col)
test_orig.to_csv('test_magic2.csv', index=False, columns=col)
