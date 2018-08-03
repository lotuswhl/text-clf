#! /usr/bin/python
# -*- encoding:utf-8 -*-
import os


class Config(object):
    # 词汇大小
    vocab_size = 5000
    # 语句最大长度
    sent_max_length = 600
    # 词向量的size
    embedding_size = 64
    # 训练时的batch_size
    batch_size = 100

    num_classes = 10

    num_epochs = 20

    #################
    # Text-CNN 配置
    ################
    # 卷积核的数量
    num_filters = 256
    # 卷积大小
    kernel_size = 5
    # drop out rate
    dropout_zero_prob = 0.5
    # 全连接层的单元数
    fc_hidden_node = 128

    # dirs
    data_dir = 'data/cnews'
    train_path = os.path.join(data_dir, 'cnews.train.txt')
    test_path = os.path.join(data_dir, 'cnews.test.txt')
    val_path = os.path.join(data_dir, 'cnews.val.txt')
    vocab_path = os.path.join(data_dir, 'cnews.vocab.txt')
