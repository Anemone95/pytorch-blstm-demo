#!/usr/bin/env python3
"""

:Author: Anemone Xu
:Email: anemone95@qq.com
:copyright: (c) 2019 by Anemone Xu.
:license: Apache 2.0, see LICENSE for more details.
"""
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class WordTokenDict:
    def __init__(self, unk_token: str = "<unk>"):
        self._idx2word = [""]  # 0常用做padding
        self._word2idx = {}
        self.unk_token = unk_token
        self.add_word(unk_token)

    def add_word(self, word: str):
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1

    def wtoi(self, word: str) -> int:
        return self._word2idx.get(word, self._word2idx[self.unk_token])

    def itow(self, idx: int) -> str:
        return self._idx2word[idx] if idx < len(self._idx2word) else self.unk_token

    def __str__(self):
        return self._word2idx.__str__()

    def __len__(self) -> int:
        return len(self._idx2word)


class Tokenizer:

    def __init__(self, sentence_iterator: [str] = None, freq_gt: int = 0, token_dict: WordTokenDict = None):
        if not token_dict and not sentence_iterator:
            raise AttributeError("Must specify sentence_iterator or token_dict")
        if token_dict:
            self.dict = token_dict
        else:
            self.dict = WordTokenDict()
            self._update_dict(sentence_iterator, gt=freq_gt)

    def __str__(self):
        return self.dict.__str__()

    def _update_dict(self, sentence_iterator: [str], gt: int) -> {str: int}:
        # generate_dict
        word_freq = {}
        for sample in sentence_iterator:
            for word in sample.split():
                word = word.strip()
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        for word, freq in word_freq.items():
            if freq > gt:
                self.dict.add_word(word)
        return self.dict

    def encode(self, string: str) -> [int]:
        words = string.split()
        encoded = map(lambda e: self.dict.wtoi(e.strip()), words)
        return list(encoded)

    def decode(self, int_list: [int]) -> str:
        return " ".join(map(lambda e: self.dict.itow(e), int_list))


class DataIterator:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data_generator = self._generator()

    def __iter__(self):
        self.data_generator = self._generator()
        return self.data_generator

    def _generator(self):
        with open(self.datafile, 'r') as f:
            for e in f.readlines():
                sentence, label = e.split(',')
                yield sentence, label.strip()


class TextDataset(Dataset):
    def __init__(self, data_iterator: DataIterator, tokenizer: Tokenizer):
        self.data = list(map(lambda e: (tokenizer.encode(e[0]), e[1]), data_iterator))
        self.max_len = 0
        self.tokenizer = tokenizer
        for words, _ in self.data:
            if len(words) > self.max_len:
                self.max_len = len(words)

    def __getitem__(self, item):
        # txt = torch.LongTensor(np.zeros(self.max_len, dtype=np.int64))
        WORD, LABEL = 0, 1
        txt = torch.zeros(len(self.data[item][WORD]), dtype=torch.long)
        for i, token in enumerate(self.data[item][WORD]):
            txt[i] = token
        label = torch.LongTensor([int(self.data[item][LABEL])])
        return txt, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # s = "I love you"
    # s1 = "hello world"
    # sarr = [s, s1]
    # tokenizer = Tokenizer(sentence_generator=sarr)
    # print(tokenizer.decode([2, 3, 6]))
    for i in DataIterator('./data/train.txt'):
        print(i)
    pass
