class Parameters(object):
    seq_length = 300         #max length of cnnsentence
    num_classes = 10         #number of labels

    keep_prob = 0.9          #droppout
    lr = 0.00005               #learning rate

    num_epochs = 5           #epochs
    batch_size = 16          #batch_size


    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./bert_model/chinese_L-12_H-768_A-12/vocab.txt'        #vocabulary
    bert_config_file = './bert_model/chinese_L-12_H-768_A-12/bert_config.json'
    init_checkpoint = './bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
