# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:20:42 2018

@author: Administrator
"""
from word_vector import *
from Train_bayes import *
from numpy import *

def classify_bayes(test_vec, p0, p1, p_pos):
    '''
    @vec2Classify:待测试分类的词条向量
    @p0:类别0所有文档中各个词条出现的频数p(wi|c0)
    @p1:类别1所有文档中各个词条出现的频数p(wi|c1)
    @p_pos:类别为1的文档占文档总数比例
    '''
    p1 = sum(test_vec * p1) + log( p_pos)
    p0 = sum(test_vec * p0) + log( 1.0 - p_pos)
    # 原公式为乘积P(w|c)P(c)，log取对数后乘积变为相加
    # print(p1,p0)
    if p1 > p0:
        return 1
    else:
        return 0
    
#分类测试整体函数        
def Test_classify():
    #由数据集获取文档矩阵和类标签向量
    listOPosts,listClasses=loadDataSet()
    #统计所有文档中出现的词条，存入词条列表
    myVocabList=Create_wordVec(listOPosts)
    #创建新的列表
    trainMat=[]
    for postinDoc in listOPosts:
        #将每篇文档利用words2Vec函数转为词条向量，存入文档矩阵中
        trainMat.append(Words2Vec(myVocabList,postinDoc))\
    #将文档矩阵和类标签向量转为NumPy的数组形式，方便接下来的概率计算
    #调用训练函数，得到相应概率值
    p1V,p0V,pAb=train_bayes(array(trainMat),array(listClasses))
    #测试文档
    testEntry=['love','my','dalmation']
    #将测试文档转为词条向量，并转为NumPy数组的形式
    thisDoc=array(Words2Vec(myVocabList,testEntry))
    #利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry,'classified as:',classify_bayes(thisDoc,p0V,p1V,pAb))
    #第二个测试文档
    testEntry1=['stupid','garbage']
    #同样转为词条向量，并转为NumPy数组的形式
    thisDoc1=array(Words2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classify_bayes(thisDoc1,p0V,p1V,pAb))