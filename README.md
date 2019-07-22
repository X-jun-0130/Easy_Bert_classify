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

## data_process.py
数据处理代码，重点介绍，如何处理成bert模型所需样式。
```
    for i in range(len(content)):
        eachline = content[i]
        eachline = eachline[0:pm.seq_length-2]
        text = tokenizer.tokenize(eachline) #将句子变成 字序列
        text.insert(0, "[CLS]")
        text.append("[SEP]")
        text2id = tokenizer.convert_tokens_to_ids(text) #将字序列 变成 数字序列
        segment = [0] * len(text2id)
        mask_ = [1] * len(text2id)
        _label = cat_to_id[labels[i]]
        while len(text2id) < pm.seq_length:
            text2id.append(0)
            mask_.append(0)
            segment.append(0)
        assert len(text2id) == pm.seq_length
        assert len(mask_) == pm.seq_length
        assert len(segment) == pm.seq_length


        input_id.append(text2id)
        input_segment.append(segment)
        mask.append(mask_)
```
### input_id,input_segment,mask为bert模型输入数据
```
tokenizer = tokenization.FullTokenizer(vocab_file=pm.vocab_filename, do_lower_case=False) 加载预训练bert模型的中文词典
text = tokenizer.tokenize(eachline)  将句子转换成 字列表，如：输入“你好”，返回['你','好']
bert需要在字列表首位添加 "[CLS]"，尾部添加"[SEP]"字符
text.insert(0, "[CLS]")
text.append("[SEP]")
返回数据为：['CLS','你','好','SEP']
然后将列表字转换成数字，还是利用bert中文字典，
text2id = tokenizer.convert_tokens_to_ids(text) 将字列表 变成 数字列表

segemnt表示输入的句子是段落几，第一段落用0表示，第二段落用1表示，...。bert能够接受的中文句子长度为512，大于这个长度可以分段输入。

mask矩阵，句子原长度部分，权重值为1，padding得来的部分，权重值为0

在padding时，不足部分补0；text2id.append(0)，mask_.append(0)，segment.append(0)
```

## bert_classify.py
bert模型
```
with tf.variable_scope('bert'):
    bert_embedding = modeling.BertModel(config=bert_config,
                                        is_training=True,
                                        input_ids=input_x,
                                        input_mask=mask,
                                        token_type_ids=input_segment,
                                        use_one_hot_embeddings=False)

    embedding_inputs = bert_embedding.get_sequence_output()
```
模型输出：
```
is_training=True表示进行finetune,use_one_hot_embeddings=False表示不使用TPU。
bert_embedding.get_sequence_output()输出数据形式[batch_size,seq_length,hidden_dim],hidden_dim=712
```
