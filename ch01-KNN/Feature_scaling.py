# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:41:25 2018

@author: Administrator
"""

from numpy import *

def Norm_feature(data_set):
    minVal = data_set.min(0)
    maxVal = data_set.max(0)
    ranges = maxVal - minVal     # 计算极差
    # 下一步将初始化一个与原始数据矩阵同尺寸的矩阵
    # 利用tile函数实现扩充向量，并进行元素间的对位运算
    norm_set = zeros(shape(data_set))
    rows = data_set.shape[0]
    norm_set = (data_set - tile(minVal, (rows, 1))) / tile(ranges, (rows,1))
    
    return norm_set, ranges, minVal
# 返回极差与最小值留待后续备用
    

