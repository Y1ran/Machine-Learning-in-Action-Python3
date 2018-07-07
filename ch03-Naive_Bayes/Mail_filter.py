# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:52:50 2018

@author: Administrator
"""
import re
from classifier_test import *
from Train_bayes import *
from word_bag import *
from numpy import *

def text_parser(string_inp):
    tokens = re.split(r'\W*', string_inp)
    return [tok.lower() for tok in tokens if len(tok) > 2]

def Spam_filter(filename):
    '''
    处理数据长字符串
    对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
    # 将分割后的字符串中所有的大些字母变成小写lower(),并且只
    # 保留单词长度大于3的单词
    '''
    #新建三个列表
    docList = [];classList = [];fullTest = []
    #i 由1到26
    for i in range(1,26):
        #打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = text_parser(open(filename + '/spam/%d.txt' % i).read())
        #将得到的字符串列表添加到docList
        docList.append(wordList)
        #将字符串列表中的元素添加到fullTest
        fullTest.extend(wordList)
        #类列表添加标签1
        classList.append(1)
        #打开并取得另外一个类别为0的文件，然后进行处理
        wordList = text_parser(open(filename + '/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    #将所有邮件中出现的字符串构建成字符串列表
    vocabList=Create_wordVec(docList)
    #构建一个大小为50的整数列表和一个空列表
    trainingSet = list(range(50)); testSet = []
    #随机选取1~50中的10个数，作为索引，构建测试集
    for i in range(10):
        #随机选取1~50中的一个整型数
        randIndex=int(random.uniform(0,len(trainingSet)))
        #将选出的数的列表索引值添加到testSet列表中
        testSet.append(trainingSet[randIndex])
        #从整数列表中删除选出的数，防止下次再次选出
        #同时将剩下的作为训练集
        del(trainingSet[randIndex])
    #新建两个列表
    trainMat=[];trainClasses=[]
    #遍历训练集中的吗每个字符串列表

    for docIndex in trainingSet:
        #将字符串列表转为词条向量，然后添加到训练矩阵中
        trainMat.append(Word2Vec_bag(vocabList, docList[docIndex]))
        #将该邮件的类标签存入训练类标签列表中
        trainClasses.append(classList[docIndex])
    #计算贝叶斯函数需要的概率值并返回
    p1V,p0V,pSpam = train_bayes(array(trainMat), array(trainClasses))
    errorCount = 0
    #遍历测试集中的字符串列表
   
    for docIndex in testSet:
        #同样将测试集中的字符串列表转为词条向量
            
        wordVector = Word2Vec_bag(vocabList,docList[docIndex])
        # print(wordVector)
        #对测试集中字符串向量进行预测分类，分类结果不等于实际结果
        if classify_bayes(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1
        print('the error rate is:',float(errorCount)/ len(testSet))