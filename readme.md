假设有如下训练数据：

```csv
Well done,1
Good work,1
Great effort,1
nice work,1
good quality,1
Weak,0
Poor effort,0
not good, 0
poor work,0
Could have done better,0
```

# 0x01 数据预处理

即读取句子和所属标签，对句子tokenize，如果不需要特殊处理可以用torchtext的，由于这里需要对出现次数少于x次的词设为`<unk>`，所以自己做了个tokenizer：

```python
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
```

简单用一下，效果如下：

```python
In[3]: import text
In[4]: s="I love you"
In[5]: s2="Hello world"
In[6]: tokenizer=text.Tokenizer([s,s2])
In[7]: tokenizer.encode(s)
Out[7]: [2, 3, 4]
In[8]: tokenizer.decode([2,3,4])
Out[8]: 'I love you'
```

# 0x02 构造神经网络

```python
class BLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.num_directions = 2

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, dropout=0.5, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * self.num_directions, label_size)

    def forward(self, sentences, lengths):
        embeds = self.word_embeddings(sentences)
        x = embeds.view(len(sentences), self.batch_size, -1)
        x_packed = rnn_utils.pack_padded_sequence(x, lengths)
        lstm_out, (h_n, h_c) = self.lstm(x_packed, None)
        x_unpacked, lengths= rnn_utils.pad_packed_sequence(lstm_out)
        y = self.hidden2label(x_unpacked[-1])
        return y
```

在`__init__()`中构造BLSTM，编码后的文本首先进行Embedding，即将句子中的单词表示成向量（那么一个句子就变成了一个二维向量）

```python
In[20]: em(torch.tensor(tokenizer.encode(s)))
Out[20]: 
tensor([[ 1.2103,  0.6848],
        [-1.0373,  0.8026],
        [ 1.3003, -0.9857]], grad_fn=<EmbeddingBackward>)
In[21]: s3="I hate you"
In[23]: em(torch.tensor(tokenizer.encode(s3))) # 可以发现相同单词embedding是相同的
Out[23]: 
tensor([[ 1.2103,  0.6848],
        [-0.6679, -0.2068],
        [ 1.3003, -0.9857]], grad_fn=<EmbeddingBackward>)
```



# Notes

## Tensor API
* torch.stack 将元素叠加产生一个新矩阵
```python
a=torch.tensor([[1,2,3],[4,5,6]])
b=torch.tensor([[11,22,33],[44,55,66]])

c=torch.stack((a,b),dim=0)
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],
#         [[11, 22, 33],
#          [44, 55, 66]]])
# c.shape=torch.Size([2, 2, 3])

torch.stack((a,b),dim=1)
# tensor([[[ 1,  2,  3],
#         [11, 22, 33]],
#        [[ 4,  5,  6],
#         [44, 55, 66]]])

torch.stack((a,b),dim=2)
# tensor([[[ 1, 11],
#          [ 2, 22],
#          [ 3, 33]],
#         [[ 4, 44],
#          [ 5, 55],
#          [ 6, 66]]])
```
* torch.gather(dim, index) 在dim维上，按index产生新矩阵
```python
In[36]: a=torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
In[37]: a
Out[37]: 
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
In[43]: b=a.gather(dim=0, torch.tensor([[1,2,3]]))
In[44]: b
Out[44]: tensor([[ 4,  8, 12]])
``` 
解释一下结果， dim=0 指针对与第0维索引（按行索引），接着index=[3,2,1]，因为按行索引，所以3,2,1指取(1,3),(2,2),(3,1)上的元素.
注意两点，否则报错：
* 非索引维度上维数不变，如例子中$a=4x3$, b=1x3, 索引维度是行，所以行可以变（原先又4行索引后只有1行），但是列不变；
* 索引结果dim数（len(a.shape)）不变，即原先a是一个2维矩阵，索引结果仍应为2维矩阵。
用数学语言描述就是，a.size=1,2,i,4,...,n, 那么index.size=1,2,j,4,...,n，dim=i，i与j无关