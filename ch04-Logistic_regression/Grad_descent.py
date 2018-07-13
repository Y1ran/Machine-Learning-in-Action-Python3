# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:06:53 2018

@author: Administrator
"""

from numpy import *

def loadData(filename):
    '''
    '''
    datamat = []; labelmat = []
    with open(filename) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            datamat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            # jisuan x0, x1,x2. x0wei 1
            labelmat.append(int(line_arr[2]))
    return datamat, labelmat

def sigmoid(inp):
    return 1.0 / (1 + exp(-inp))

def Grad_descent(datamat, labels):
    '''
    
    '''
    data = mat(datamat)
    label = mat(labels).transpose()
    
    m, n = shape(datamat)
    alpha = 0.001; max_iter = 500
    weights = ones((n, 1))   # 
    for k in range(max_iter):
        # 
        z = dot(datamat, weights)
        y_pred = sigmoid(z)
        error = (label - y_pred)
        # grad(x) = (y - f(x)) * x'
        weights = weights + alpha * data.transpose() * error
    return weights
    
    
    