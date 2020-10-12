# quora-question-pairs-kaggle
1、本项目基于Kaggle quora-question-pairs ，使用python3开发,使用了nltk,gensim,xgboost，difflib第三方库，需要使用pip下载。

2、数据集使用kaggele比赛下载的数据文件，存放在项目data文件夹

3、analysis文件夹存放对数据探索阶段使用的代码

4、feature文件夹存放特征提取所使用代码，具体执行流程可参见run.sh文件
这里其实比较耗时，主要在特征提取维度较多，故而好时，总体来看特征提取耗时大约在10个小时左右。

5、model文件夹存放模型选择流程所使用的代码文件，模型选择代码执行较快，耗时在20分钟左右。

6、train_submit.py 是最终模型训练文件，并产生提交文件，这里执行时间大约在1个小时。
