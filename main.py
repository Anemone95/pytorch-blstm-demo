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
from metriccalculator import MetricCalculator

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

    metric = MetricCalculator()
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
            if HAS_GPU:
                metric.update(predicted.cpu(), train_labels.cpu())
            else:
                metric.update(predicted, train_labels)
            total += len(train_labels)
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        accuracy, recall, precision = metric.compute(["accuracy", "recall", "precision"])

        print("[Epoch: {cur_epoch}/{total_epoch}] Training Loss: {loss:.3}, "
              "Training Acc: {acc:.3}, Training Precision: {precision:.3}, Training Recall: {recall:.3}, Training F1: {f1:.3}"
              .format(cur_epoch=epoch, total_epoch=EPOCH, loss=train_loss_[epoch],
                      acc=accuracy, precision=precision, recall=recall, f1=(2*precision*recall)/(precision+recall)))


if __name__ == '__main__':
    train()
    pass
