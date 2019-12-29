#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
import torch.nn as nn


class BLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.num_directions = 2

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True,
        #                     dropout=0.5, bidirectional=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, dropout=0.5,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * self.num_directions, label_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        y = self.hidden2label(lstm_out[-1])
        return y


if __name__ == '__main__':
    pass
