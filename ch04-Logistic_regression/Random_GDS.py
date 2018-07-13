# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:30:25 2018

@author: Administrator
"""

from numpy import *
from Grad_descent import sigmoid

def Stoch_gdescent(datamat, labels, num_iter = 150):
    '''
    基于样本集中的每个样本（随机抽取）进行迭代
    求出优化的参数，并在此基础上对alpha进行衰减
    '''
    m, n = shape(datamat)
    alpha = 0.01
    weights = ones(n)   #
    
    for j in range(num_iter):
        for i in range(m):
            #j << x时衰减效果受到影响，0.01则为了保存一定的速率
            alpha = 4 / (1.0 + j + i) + 0.01
            randidx = int(random.uniform(0,len(range(m))))
            
            z = sum(datamat[randidx] * weights)
            y_pred = sigmoid(z)
            error = float(labels[randidx]) - y_pred
            # grad(x) = (y - f(x)) * x'为迭代公式（梯度）
            weights = weights + (alpha * error) * array(datamat[randidx])
    return weights