# -*-coding: utf-8 -*-
"""
这是直接对softmax原始的公式的实现版本
"""

import numpy as np


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s



axis = 1  # 默认计算最后一维 random = np.array([[0.25221437,0.57489906,0.77723957]]) tensor([[0.2456, 0.3392, 0.4152]], dtype=torch.float64)
random = np.random.rand(1,3)
print(random)
# [1]使用自定义softmax
s1 = softmax(random, axis=axis)
print("s1:{}".format(s1))
