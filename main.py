#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import torch.optim as optim
import blstm
import text

## hyper parameter
EPOCH = 20
BATCH_SIZE = 3
HAS_GPU = torch.cuda.is_available()
BASE_LEARNING_RATE = 0.01
EMBEDDING_DIM = 8  # embedding
HIDDEN_DIM = 2  # hidden dim
LABEL_NUM = 2  # number of labels


def adjust_learning_rate(optimizer, epoch):
    lr = BASE_LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train():
    dataset = text.TextDataset(data_file="./data/train.txt")
    tokenizer = text.Tokenizer()
    tokenizer.build_dict(dataset)

    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=tokenizer.tokenize_labeled_batch)

    ### create model
    model = blstm.BLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                        vocab_size=len(tokenizer.dict), label_size=LABEL_NUM, use_gpu=HAS_GPU)
    if HAS_GPU:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []

    for epoch in range(EPOCH):
        optimizer = adjust_learning_rate(optimizer, epoch)

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, lengths, train_labels = traindata
            if len(train_labels.shape) > 1:
                train_labels = torch.squeeze(train_labels)

            if HAS_GPU:
                train_inputs, lengths, train_labels = train_inputs.cuda(), lengths.cuda(), train_labels.cuda()

            # 清空梯度
            model.zero_grad()
            #
            # 转置，否则需要batchfirst=True
            output = model(train_inputs, lengths)

            loss = loss_function(output, train_labels)
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            # total_loss += loss.data[0]

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f'
              % (epoch, EPOCH, train_loss_[epoch], train_acc_[epoch],))


if __name__ == '__main__':
    train()
    pass
