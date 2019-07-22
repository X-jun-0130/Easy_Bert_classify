#encoding:utf-8

import tensorflow.contrib.keras as kr
import numpy as np
import codecs
from bert import tokenization
from parameters import Parameters as pm

def read_file(filename):
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                contents.append(content)
            except:
                pass
    return labels, contents

labels, content = read_file(pm.train_filename)
length_text = len(content)

def Token(filename):
    input_id, input_segment, mask, label = [], [], [], []
    tokenizer = tokenization.FullTokenizer(vocab_file=pm.vocab_filename, do_lower_case=False) #加载bert汉字词典

    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))

    labels, content = read_file(filename)
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
        label.append(_label)
    return input_id, input_segment, mask, label



def process(label):

    label_pad = kr.utils.to_categorical(label, num_classes=10)
    return label_pad


def batch_iter(id, segment, mask, label, batch_size = pm.batch_size):
    data_len = len(id)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    id = np.array(id)
    segment = np.array(segment)
    mask = np.array(mask)
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    id_shuff = id[indices]
    segment_shuff = segment[indices]
    mask_shuff = mask[indices]
    label_shuff = label[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield id_shuff[start_id:end_id], segment_shuff[start_id:end_id], mask_shuff[start_id:end_id], label_shuff[start_id:end_id]
