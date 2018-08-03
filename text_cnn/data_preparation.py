#! /usr/bin/python
# -*- encoding:utf-8 -*-
import nltk
import numpy as np
from nltk.probability import FreqDist
from config import Config
import keras
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset
import os
import torch
"""
数据准备帮助函数，主要包括：
    读取文件： 读取与处理好的文件，返回标签和内容信息
    构建词汇库： 根据训练数据集构建词汇库，并将词汇库保存到本地文件，方便后续使用
        返回词汇表，以及word_to_id
    分类帮助函数： 可以获取分类列表，以及分类到id的转换
    文件解析函数： 将文件数据转为模型可以接受的数据输入格式
    ...
"""
def open_file(filename,mode='r',encoding='utf-8',errors='ignore'):
    return open(filename,mode,encoding=encoding,errors=errors)

def read_file(filename, mode='r'):
    labels = []
    texts = []
    with open_file(filename, mode) as f:
        for line in f:
            try:
                label, text = line.strip().split("\t")
                if text is not None:
                    labels.append(label)
                    # !注意将字符串语句转换为list单词
                    texts.append(list(text))
            except:
                pass
    return labels, texts


def build_vocabulary(filename, vocab_save_path, vocab_size=Config.vocab_size):
    _, texts = read_file(filename)
    all_words=[]
    for text in texts:
        all_words.extend(text)
    freqDist = FreqDist(all_words)
    word_cnt_pairs = freqDist.most_common(vocab_size-1)
    vocab, _ = zip(*word_cnt_pairs)
    # 添加一个用于填充的单词，从而保证所有的句子的长度相同
    vocab = ["<P>"]+list(vocab)
    open_file(vocab_save_path, "w").write("\n".join(vocab)+"\n")


def get_vocabulary(vocab_save_path):
    vocab = []
    with open_file(vocab_save_path,'r') as f:
        for line in f.readlines():
            word = line.strip()
            vocab.append(word)
    word_to_id = dict([(word, id) for id, word in enumerate(vocab)])
    return vocab, word_to_id


def get_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    category_to_id = dict([(categ, id) for id, categ in enumerate(categories)])
    return categories, category_to_id


def parse_from_file(filename, word_to_id, categ_to_id, sents_max_len=Config.sent_max_length):
    labels, texts = read_file(filename)
    texts_ids = []
    labels_ids = []
    for i in range(len(labels)):
        text_ids = [word_to_id[word] for word in texts[i] if word in word_to_id]
        texts_ids.append(text_ids)
        labels_ids.append(categ_to_id[labels[i]])

    x_p = keras.preprocessing.sequence.pad_sequences(texts_ids, sents_max_len)
    # y_p = keras.utils.to_categorical(labels_ids, len(categ_to_id))
    y_p = np.array(labels_ids)
    return x_p, y_p


def batch_iterator(x, y, batch_size=Config.batch_size):

    data_len = len(x)

    num_batches = int((data_len-1)/batch_size)+1

    idx = np.random.permutation(data_len)

    x = x[idx]
    y = y[idx]

    for i in range(num_batches):
        beg = i*batch_size
        end = min(beg+batch_size, data_len)

        yield x[beg:end], y[beg:end]

def batch_iter(filename=Config.train_path):
    if not os.path.exists(Config.vocab_path):
        build_vocabulary(filename, Config.vocab_path)

    vocab, word_to_id = get_vocabulary(Config.vocab_path)
    category, categ_to_id = get_category()
    x_, y_ = parse_from_file(filename, word_to_id, categ_to_id)
    return batch_iterator(x_,y_)

class THCNewsDataSet(dataset.Dataset):
    def __init__(self, filename=Config.train_path):
        super(THCNewsDataSet, self).__init__()
        if not os.path.exists(Config.vocab_path):
            build_vocabulary(filename, Config.vocab_path)

        vocab, word_to_id = get_vocabulary(Config.vocab_path)
        category, categ_to_id = get_category()
        x_, y_ = parse_from_file(filename, word_to_id, categ_to_id)
        print(x_.shape)
        print(y_.shape)

        self.x = torch.tensor(x_, dtype=torch.long)
        self.y = torch.tensor(y_, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
