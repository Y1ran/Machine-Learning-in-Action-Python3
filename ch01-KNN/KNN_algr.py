# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:57:50 2018

@author: Administrator
"""

from numpy import *
import operator


def classify_KNN(test_X, train_set, labels, K):
    rows = train_set.shape[0]
    diff = tile(test_X, (rows, 1)) - train_set
    # 这一行利用tile函数将输入样本实例转化为与训练集同尺寸的矩阵
    # 便之后的矩阵减法运算
    
    sqDistance = (diff ** 2).sum(axis=1)  
    Distance = sqDistance ** 0.5
    sorted_Distance = Distance.argsort()
    # 对每个训练样本与输入的测试样本求欧几里得距离，即点之间的范数
    # 随后按距离由小到大进行排序
    
    classCount = {}
    for i in range(K):
        vote_label = labels[sorted_Distance[i]]
        classCount[vote_label] = classCount.get(vote_label, 0) + 1
    #记录距离最小的前K个类，并存放入列表。KEY对应标签，VALUE对应计数
    
    sortedClassCount = sorted(classCount.items(), 
                              key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


