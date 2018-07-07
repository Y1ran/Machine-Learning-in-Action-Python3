# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:50:09 2018

@author: Administrator
"""
def Word2Vec_bag(wordList, input_set):
    '''
    基于多项式而非伯努利模型的贝叶斯方法
    称为词袋模型，具有大于1的权重
    @wordList：为前一个函数的输出值（包含单词）
    @input_set：输入需要分类的集合
    '''
    return_vec = [0] * len(wordList) 
    # 创建与词汇表等长的列表向量
    for word in input_set:
        if word in wordList:
            return_vec[wordList.index(word)] += 1 # 出现的单词赋1
        # else: print("the word %s is not in list" % word)
    return return_vec
