# Introduction

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