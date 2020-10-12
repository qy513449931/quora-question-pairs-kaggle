python feature/base_feature.py
python feature/freq.py
python feature/intersect.py
python feature/kcores.py
python feature/pagerank.py
# 获得语料模型
python feature/doc2vec_model/doc2vec_model.py
# 根据语料，获得基于Word2Vec向量所得到的相似值
python feature/doc2vec_infer.py

### 建立特征并存储
# python feature/generate_stem.py
# python feature/generate_unigram.py
# python feature/generate_bigram.py
# python feature/generate_distinct_bigram.py
#python feature/generate_distinct_unigram.py
# python feature/generate_cooccurence_ngram.py
# python feature/generate_cooccurence_distinct.py

# titdf 分解特征
# python feature/generate_tfidf_unigram.py
# python feature/generate_tfidf_ngram.py
# python feature/generate_tfidf_distinct_bigram.py
# python feature/generate_tfidf_coo.py
# python feature/generate_tfidf_coo_distinct.py

python feature/collect_feature.py
# 以上是提取特征，并将特征保存到文件，便于下一步模型使用
python train_submit.py

