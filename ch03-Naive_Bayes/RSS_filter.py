# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 21:56:52 2018

@author: Administrator
"""

#实例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
#RSS源分类器及高频词去除函数

import operator
from numpy import *
from Mail_filter import *
from Train_bayes import *
from word_bag import *
from classifier_test import *

def calMostFreq(vocabList,fullTest):
    #导入操作符
    import operator
    #创建新的字典
    freqDict={}
    #遍历词条列表中的每一个词
    for token in vocabList:
        #将单词/单词出现的次数作为键值对存入字典
        freqDict[token]=fullTest.count(token)
    #按照键值value(词条出现的次数)对字典进行排序，由大到小
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    
    #返回出现次数最多的前30个单词
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    #新建三个列表
    docList=[];classList=[];fullTest=[]
    #获取条目较少的RSS源的条目数
    minLen = 100
    #遍历每一个条目
    for i in range(minLen):
        #解析和处理获取的相应数据
        wordList = text_parser(feed1['entries'][i]['summary'])
        #添加词条列表到docList
        docList.append(wordList)
        #添加词条元素到fullTest
        fullTest.extend(wordList)
        #类标签列表添加类1
        classList.append(1)
        #同上
        wordList = text_parser(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        #此时添加类标签0
        classList.append(0)
    #构建出现的所有词条列表
    vocabList=Create_wordVec(docList)
    #找到出现的单词中频率最高的30个单词
    top30Words=calMostFreq(vocabList,fullTest)
    #遍历每一个高频词，并将其在词条列表中移除
    #这里移除高频词后错误率下降，如果继续移除结构上的辅助词
    #错误率很可能会继续下降
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    #下面内容与函数spaTest完全相同
    trainingSet=list(range(2*minLen));testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(Word2Vec_bag(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p1V,p0V,pSpam=train_bayes(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=Word2Vec_bag(vocabList,docList[docIndex])
        if classify_bayes(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1 
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    #利用RSS源分类器获取所有出现的词条列表，以及每个分类中每个单词出现的概率
    vocabList,p0V,p1V=localWords(ny,sf)
    #创建两个元组列表
    topNY=[];topSF=[]
    #遍历每个类中各个单词的概率值
    for i in range(len(p0V)):
        #往相应元组列表中添加概率值大于阈值的单词及其概率值组成的二元列表
        if(p0V[i]>-6.0):topSF.append((vocabList[i],p0V[i]))
        if(p1V[i]>-6.0):topNY.append((vocabList[i],p1V[i]))
    # 对列表按照每个二元列表中的概率值项进行排序，排序规则由大到小
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=true)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    #遍历列表中的每一个二元条目列表
    for item in sortedSF:
        #打印每个二元列表中的单词字符串元素
        print(item[0])
    #解析同上
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=true)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedNY:
        print(item[0])