# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:42:04 2018

@author: Administrator
"""
import operator
from Split_by_entropy import *

def Majority_vote(classList):
    '''
    使用多数表决法：若集合中属于第K类的节点最多，则此分支集合
            划分为第K类
    '''
    classcount = {}
    for vote in classList:
        classcount[vote] = classcount.get(vote,0) + 1
    sorted_count = sorted(classcount.items(), key = operator.itemgetter(1),\
                          reverse = True)
    # 获取每一类出现的节点数（没出现默认为0）并进行排序
    # 返回最大项的KEY所对应的类别
    return sorted_count[0][0]

def Create_Tree(dataset,labels):

    classList = [x[-1] for x in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #
    if len(dataset[0]) == 1:
        return Majority_vote(classList)
    
    best_feature = Split_by_entropy(dataset)
    best_labels = labels[best_feature]
    
    myTree = {best_labels:{}}
    # 此位置书上写的有误，书上为del(labels[bestFeat])
    # 相当于操作原始列表内容，导致原始列表内容发生改变
    # 按此运行程序，报错'no surfacing'is not in list
    # 以下代码已改正
    
    # 复制当前特征标签列表，防止改变原始列表的内容
    subLabels=labels[:]
    # 删除属性列表中当前分类数据集特征
    del(subLabels[best_feature])

    # 使用列表推导式生成该特征对应的列
    f_val = [x[best_feature] for x in dataset]
    uni_val = set(f_val)
    for value in uni_val:
        # 递归创建子树并返回
        myTree[best_labels][value] = Create_Tree(Split_Data(dataset\
              ,best_feature,value), subLabels)
    
    return myTree



