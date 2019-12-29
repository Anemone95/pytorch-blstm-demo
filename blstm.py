#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


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

    def forward(self, sentences, lengths):
        embeds = self.word_embeddings(sentences)
        x = embeds.view(len(sentences), self.batch_size, -1)
        x_packed = rnn_utils.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, h_c) = self.lstm(x_packed, None)
        x_unpacked, lengths= rnn_utils.pad_packed_sequence(lstm_out)
        y = self.hidden2label(x_unpacked[-1])
        return y


if __name__ == '__main__':
    pass
