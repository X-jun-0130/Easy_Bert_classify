# Easy_Bert_classify
运用Bert进行文本分类


![bert_classify](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/bert_classify.png)
# 数据集：
本实验是使用THUCNews的一个子集进行训练与测试，数据集请自行到THUCTC：一个高效的中文文本分类工具包下载，请遵循数据提供方的开源协议;

文本类别涉及10个类别：categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；

cnews.train.txt: 训练集(5000*10)

cnews.val.txt: 验证集(500*10)

cnews.test.txt: 测试集(1000*10)

训练所用的数据，以及训练好的词向量可以下载：链接: https://pan.baidu.com/s/1daGvDO4UBE5NVrcLaCGeqA 提取码: 9x3i 
# 预训练的bert中文模型：
下载地址链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
![存储为如此形式：](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/bert_model.png)

# 代码分析
## parameters.py
超参数存放位置，修改模型参数，可改动内部数据
