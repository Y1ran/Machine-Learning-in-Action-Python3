# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:06:48 2018

@author: Administrator
"""

from math import log
from numpy import *

def cal_entropy(data):
    '''计算样本实例的熵'''
    entries_num = len(data)
    label_count = {} #字典存储每个类别出现的次数
 
    for vec in data:
        cur_label = vec[-1] 
    # 将样本标签提取出来，并计数
        label_count[cur_label] = label_count.get(cur_label,0) + 1
    Entropy = 0.0
    # 对每一个类别，计算样本中取到该类的概率
    # 最后将概率带入，求出熵
    for key in label_count:
        prob = float(label_count[key]) / entries_num
        Entropy += prob * math.log(prob, 2) #此处使用numpy.math
    return (0-Entropy)


        
        