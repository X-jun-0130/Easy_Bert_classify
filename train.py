from data_process import Token, process, batch_iter
from bert_classify import *
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

best = 0.0
n = 0

print("Loading Training data...")
input_id, input_segment_, mask_, label = Token(pm.train_filename)

test_id, test_segment, test_mask, test_label = Token(pm.test_filename)

label = process(label)

def evaluate(sess, test_id, test_segment, test_mask, test_label):
    A = 1e-10
    for i in range(len(test_label)):
        pre_lab = sess.run(predict, feed_dict={input_x: [test_id[i]],
                                               input_segment: [test_segment[i]],
                                               mask: [test_mask[i]],
                                               keep_pro: 1.0})
        result = test_label[i]
        if int(pre_lab) == int(result):
            A += 1

    P = A / float(len(test_label))

    return P

tensorboard_dir = './tensorboard/bert_classify'
save_dir = './checkpoints/bert_classify'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(tensorboard_dir)
saver = tf.train.Saver()
writer.add_graph(session.graph)

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
