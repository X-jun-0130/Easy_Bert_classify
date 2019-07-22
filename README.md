# Easy_Bert_classify
运用Bert进行文本分类

使用1080ti 11G GPU, 如果GPU比较小的话，建议调小batch_size, 否则很容易OOM!

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
bert模型:
```
with tf.variable_scope('bert'):
    bert_embedding = modeling.BertModel(config=bert_config,
                                        is_training=True,
                                        input_ids=input_x,
                                        input_mask=mask,
                                        token_type_ids=input_segment,
                                        use_one_hot_embeddings=False)

    embedding_inputs = bert_embedding.get_sequence_output()
is_training=True表示进行finetune,  use_one_hot_embeddings=False表示不使用TPU。
bert_embedding.get_sequence_output()输出数据形式[batch_size,seq_length,hidden_dim],hidden_dim=712
```
分类：
```
with tf.variable_scope('fully_connected'):
    output = embedding_inputs[:, 0, :]
    output = tf.layers.dropout(output, keep_pro)
    final_out = tf.layers.dense(output, pm.num_classes)
    score = tf.nn.softmax(final_out)
    predict = tf.argmax(score, 1)
取每一句中CLS位置的值作为全连接层的输入，然后进行softmax
另外，如果做NER的话，那是输入就是bert_embedding.get_sequence_output()。
```
优化器：
```
with tf.variable_scope('optimizer'):
    num_train_steps = int((length_text) / pm.batch_size * pm.num_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)  # 总的迭代次数 * 0.1 ,这里的0.1 是官方给出的，我直接写过来了
    train_op = optimization.create_optimizer(loss, pm.lr, num_train_steps, num_warmup_steps, False)
 官方提供的 optimization 主要是学习速率可以动态调整，如下面简图，学习速率由小到大，峰值就是设置的lr,然后在慢慢变小，
 整个学习速率，呈现三角形

                 -
               -      -
             -          -
           -                 -
```

获取预训练bert模型中所有的训练参数。
```
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, pm.init_checkpoint)

tf.train.init_from_checkpoint(pm.init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
session = tf.Session()
session.run(tf.global_variables_initializer())
```
## train.py
模型训练
```
for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        num_batchs = int((len(label) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(input_id, input_segment_, mask_, label, pm.batch_size)
        for x_id, x_segment, x_mask, y_label in batch_train:
            n += 1
            feed_dict = feed_data(x_id, x_mask, x_segment, y_label, pm.keep_prob)
            _,  train_summary, train_loss, train_accuracy = session.run([train_op, merged_summary,
                                                                        loss, accuracy], feed_dict=feed_dict)
            if n % 100 == 0:
                print('步骤:', n, '损失值:', train_loss, '准确率:', train_accuracy)

        P = evaluate(session, test_id, test_segment, test_mask, test_label)
        print('测试集准确率:', P)
        if P > best:
            best = P
            print("Saving model...")
            saver.save(session, save_path, global_step=(epoch*num_batchs))
每个epoch结束，输出训练模型在测试集(1000*10)上的准确率，模型有进步时，保存当前模型。
```

模型大致流程如此，仅作为展示，我只跑了5个epoch, epoch1的准确率已到达90%以上，最优结果时第四个epoch，准确率94.37%。自己实际使用bert时，可以先设置epoch为20.


![epoch1](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/epoch1.png)
![epoch4](https://github.com/NLPxiaoxu/Easy_Bert_classify/blob/master/image/epoch4.png)


最后，我并没有写如何调用训练好的已保存的分类模型(predict.py)。这个部分很简单，直接调用最后一次保存的模型就好了。进行预测的话，train.py中的evaluate函数可以直接拿来用。

## 最后的最后，欢迎star和fork
