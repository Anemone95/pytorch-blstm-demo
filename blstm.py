#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor


class BLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.num_directions = 2

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, dropout=0.5,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * self.num_directions, label_size)

    def get_bi_last_output(self, lstm_out: Tensor, lengths: Tensor, batch_first=False) -> Tensor:
        if batch_first:
            raise NotImplementedError
        indices = lengths - 1
        # @Anemone 因为输入向后传播，考虑batch中的padding，根据lengths从lstm_out中获取最后一个输入的输出
        indices = indices.unsqueeze(1).expand(lstm_out.shape[1:]).unsqueeze(0)
        forward_last_output = lstm_out.gather(0, indices)[0]

        backward_last_output = lstm_out[0]

        forward_last_output = forward_last_output.index_select(1, torch.arange(0, self.hidden_dim, dtype=torch.long))
        backward_last_output = backward_last_output.index_select(1,
                                                                 torch.arange(self.hidden_dim, self.hidden_dim * 2,
                                                                             dtype=torch.long))
        last_output = torch.cat([forward_last_output, backward_last_output], dim=1)
        return last_output

    def forward(self, sentences: Tensor, lengths: [int]):
        embeds = self.word_embeddings(sentences)
        x_packed = rnn_utils.pack_padded_sequence(embeds, lengths)
        lstm_out, (h_n, h_c) = self.lstm(x_packed)
        lstm_out_unpacked, lstm_out_lengths = rnn_utils.pad_packed_sequence(lstm_out)
        # @Anemone 这里不应该用x_unpacked[-1], 而因该用lengths获取最后真实的序列最后一位
        last_lstm_out = self.get_bi_last_output(lstm_out_unpacked, lstm_out_lengths)
        y = self.hidden2label(last_lstm_out)
        return y


if __name__ == '__main__':
    pass
