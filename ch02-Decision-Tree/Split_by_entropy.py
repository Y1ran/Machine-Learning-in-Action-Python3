# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:35:15 2018

@author: Administrator
"""

from Cal_Entropy import *

def Split_Data(dataset, axis, value):
    '''
    使用传入的axis以及value划分数据集
    axis代表在每个列表中的第X位，value为用来划分的特征值
    '''
    new_subset = []
    # 利用循环将不符合value的特征值划分入另一集合
    # 相当于将value单独提取出来（或作为叶节点）
    for vec in dataset:
        if vec[axis] == value:
            feature_split = vec[:axis]
            feature_split.extend(vec[axis + 1:])
            new_subset.append(feature_split)
    # extend将VEC中的元素一一纳入feature_split
    # append则将feature_split作为列表结合进目标集合
            
    return new_subset

def Split_by_entropy(dataset):
    '''
    使用熵原则进行数据集划分
    @信息增益:info_gain = old -new
    @最优特征：best_feature
    @类别集合：uniVal
    '''
    feature_num = len(dataset[0]) - 1
    ent_old = cal_entropy(dataset)
    best_gain = 0.0
    best_feature = -1
    # ENT_OLD代表划分前集合的熵，ENT_NEW代表划分后的熵
    # best_gain将在迭代每一次特征的时候更新，最终选出最优特征
    for i in range(feature_num):
        feature_list = [x[i] for x in dataset]
        uniVal = set(feature_list)
        ent_new = 0.0
        # 使用set剔除重复项，保留该特征对应的不同取值
        for value in uniVal:
            sub_set = Split_Data(dataset, i, value)
            prob = len(sub_set) / float(len(dataset))
            # 使用熵计算函数求出划分后的熵值
            ent_new += prob * (0 - cal_entropy(sub_set))
        
        # 由ent_old - ent_new选出划分对应的最优特征
        Info_gain = ent_old - ent_new
        if(Info_gain > best_gain):
            best_gain = Info_gain
            best_feature = i
            
    return best_feature