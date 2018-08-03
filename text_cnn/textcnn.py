import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config

config = Config

class TextCNNXX(nn.Module):
    def __init__(self,):
        super(TextCNN, self).__init__()
        # 词向量准备
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        # (N,C,L)
        self.conv = nn.Conv1d(in_channels=config.embedding_size, out_channels=config.num_filters,
                              kernel_size=config.kernel_size)

        self.fc = nn.Linear(config.num_filters, config.fc_hidden_node)

        self.bn = nn.BatchNorm1d(config.fc_hidden_node)

        self.drop_out = nn.Dropout(config.dropout_zero_prob)

        self.logits = nn.Linear(config.fc_hidden_node, config.num_classes)

        # self.logsoftmax = nn.LogSoftmax(-1)

    def forward(self, batch_x):
        batch_size = len(batch_x)
        batch_embeddings = self.embedding(batch_x)  # N,L,C
        batch_embeddings.transpose_(1, 2)  # N,C,L
        conv_out = self.conv(batch_embeddings)  # N,Co,L'
        pool_out = F.max_pool1d(conv_out, conv_out.size(-1))
        pool_out.squeeze_(-1)
        fc_out = self.fc(pool_out)
        bn_out = self.bn(fc_out)
        relu_out = F.relu(bn_out)
        dropout_out = self.drop_out(relu_out)
        logits = self.logits(dropout_out)
        return logits


class TextCNN(nn.Module):
    def __init__(self,):
        super(TextCNN, self).__init__()
        # 词向量准备
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        # (N,C,L)
        self.conv = nn.Conv1d(in_channels=config.embedding_size, out_channels=config.num_filters,
                              kernel_size=config.kernel_size)

        self.fc = nn.Linear(config.num_filters, config.fc_hidden_node)

        self.drop_out = nn.Dropout(config.dropout_zero_prob)

        self.logits = nn.Linear(config.fc_hidden_node, config.num_classes)

        # self.logsoftmax = nn.LogSoftmax(-1)

    def forward(self, batch_x):
        batch_size = len(batch_x)
        batch_embeddings = self.embedding(batch_x)  # N,L,C
        batch_embeddings.transpose_(1, 2)  # N,C,L
        conv_out = self.conv(batch_embeddings)  # N,Co,L'
        pool_out = F.max_pool1d(conv_out, conv_out.size(-1))
        pool_out.squeeze_(-1)
        fc_out = self.fc(pool_out)
        dropout_out = self.drop_out(fc_out)
        relu_out = F.relu(dropout_out)
        logits = self.logits(relu_out)
        return logits
