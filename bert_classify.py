from data_process import length_text
from bert import modeling, optimization
import tensorflow as tf
from parameters import Parameters as pm
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
input_segment = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_segment')
mask = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='mask')
input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
keep_pro = tf.placeholder(tf.float32, name='drop_out')

bert_config = modeling.BertConfig.from_json_file(pm.bert_config_file)

with tf.variable_scope('bert'):
    bert_embedding = modeling.BertModel(config=bert_config,
                                        is_training=True,
                                        input_ids=input_x,
                                        input_mask=mask,
                                        token_type_ids=input_segment,
                                        use_one_hot_embeddings=False)

    embedding_inputs = bert_embedding.get_sequence_output()

with tf.variable_scope('fully_connected'):
    output = embedding_inputs[:, 0, :]
    output = tf.layers.dropout(output, keep_pro)
    final_out = tf.layers.dense(output, pm.num_classes)
    score = tf.nn.softmax(final_out)
    predict = tf.argmax(score, 1)

with tf.variable_scope('loss'):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=final_out, labels=input_y)
    loss = tf.reduce_mean(losses)

with tf.variable_scope('optimizer'):
    num_train_steps = int((length_text) / pm.batch_size * pm.num_epochs)
    num_warmup_steps = int(num_train_steps * 0.1)  # 总的迭代次数 * 0.1 ,这里的0.1 是官方给出的，我直接写过来了
    train_op = optimization.create_optimizer(loss, pm.lr, num_train_steps, num_warmup_steps, False)

with tf.variable_scope('accuracy'):
    correct_predictions = tf.equal(predict, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')


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


def feed_data(_ids, _mask, _segment, label, keep_prob):
    feet_dict = {input_x: _ids,
                 mask: _mask,
                 input_segment: _segment,
                 input_y: label,
                 keep_pro: keep_prob
                 }
    return feet_dict

