# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:35:33 2018

@author: Administrator
"""
import numpy
from numpy import *


def train_bayes(train_matrix, train_class):
    '''
    利用NumPy数组计算p(wi|c1)
    词条 属于类1的概率Prob_positive = p(c1) 
    因为是二分类所以属于类0概率 =1-p(c1)
    '''
    num_docs = len(train_matrix)
    num_word = len(train_matrix[0])
    # 获取输入文档（句子）数以及向量的长度
    Prob_positive = sum(train_class)/ float(num_docs)
    Prob_num_0 = ones(num_word) # 创建一个长度为词条向量等长的列表
    Prob_num_1 = ones(num_word) 
    # 为避免0概率使得最终乘积为0，使用拉普拉斯平滑（加入常数lamda，此处为1）
    Prob_denom_0 = 2.0
    Prob_denom_1 = 2.0
    
    for i in range(num_docs):
    # 统计类别为1的词条向量中出现的所有词条的总数
            # 即统计类1所有文档中出现单词的数目
        if train_class[i] == 1:
            Prob_num_1 += train_matrix[i]
            Prob_denom_1 += sum(train_matrix[i])
        else:
            Prob_num_0 += train_matrix[i]
            Prob_denom_0 += sum(train_matrix[i])         
    p1_vec = log(Prob_num_1 / Prob_denom_1)
    p0_vec = log(Prob_num_0 / Prob_denom_0)
    # 将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    return p1_vec, p0_vec, Prob_positive
    # p1概率实际上等于[p(w1|c1),p(w2|c1), ... p(wn|c1)]组成的向量
    # 后续计算中，会基于条件独立性求出p(W|c1)
