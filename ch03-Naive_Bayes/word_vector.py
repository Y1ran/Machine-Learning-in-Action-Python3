# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:56:38 2018

@author: Administrator
"""

from numpy import *

def loadDataSet():
    '''
    postingList: 进行词条切分后的文档集合
    classVec:类别标签   
    使用伯努利模型的贝叶斯分类器只考虑单词出现与否（0，1）
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def Create_wordVec(dataset):
    word_set = set([])
    for doc in dataset:
        word_set = word_set | set(doc) # 通过对两个集合取并，找出所有非重复的单词
    return list(word_set)

def Words2Vec(wordList, input_set):
    '''
    @wordList：为前一个函数的输出值（包含单词）
    @input_set：输入需要分类的集合
    函数输出：包含0，1的布尔型向量（对应Wordlist中的单词出现与否）
    '''
    return_vec = [0] * len(wordList) 
    # 创建与词汇表等长的列表向量
    for word in input_set:
        if word in wordList:
            return_vec[wordList.index(word)] = 1 # 出现的单词赋1
        else: print("the word %s is not in list" % word)
    return return_vec
