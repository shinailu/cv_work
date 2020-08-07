import numpy as np  # 提供矩阵运算功能的库

import torch

import torch.nn.functional as F
"""
这是使用pytorch的softmax
"""

random = np.random.rand(1,3)
print(random)
random = torch.tensor(random)
print(random)
p = F.softmax(random, dim =1)
print(p)
